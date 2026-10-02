import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline
from rpde_baselines.model.fno import FNO3d

class HeteroFNO(nn.Module):
    def __init__(self, base_fno):
        super().__init__()
        self.base = base_fno
        # The base FNO has fc1 (width -> 128) and fc2 (128 -> dim_out).
        # We will keep the base exactly as is, but add a parallel fc2 for variance!
        self.fc2_var = nn.Linear(128, base_fno.dim_out)
        # Initialize variance to predict small values initially
        nn.init.constant_(self.fc2_var.weight, 0.0)
        nn.init.constant_(self.fc2_var.bias, -5.0) # log_var = -5 means var = 0.0067

    def forward(self, x):
        # We must reimplement the forward pass of FNO3d to branch at the end
        grid = self.base.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.base.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.base.padding, 0, self.base.padding, 0, self.base.padding])

        for i in range(self.base.n_layers):
            x1 = self.base.spectral_convs[i](x)
            x2 = self.base.convs[i](x)
            x = x1 + x2
            x = self.base.bns[i](x)
            if i < self.base.n_layers - 1:
                x = F.gelu(x)

        x = x[..., :-self.base.padding, :-self.base.padding, :-self.base.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.base.fc1(x)
        x = F.gelu(x)
        
        # Branch here!
        mu = self.base.fc2(x)
        logvar = self.fc2_var(x)
        
        mu = mu.reshape(*mu.shape[:-1], self.base.shape_out[-1], self.base.shape_out[0] // self.base.shape_in[0])
        mu = mu.permute(0, 1, 5, 2, 3, 4).reshape(mu.shape[0], *self.base.shape_out)
        
        logvar = logvar.reshape(*logvar.shape[:-1], self.base.shape_out[-1], self.base.shape_out[0] // self.base.shape_in[0])
        logvar = logvar.permute(0, 1, 5, 2, 3, 4).reshape(logvar.shape[0], *self.base.shape_out)
        
        return mu, logvar

def nll_loss_fn(mu, logvar, target, scm):
    # Gaussian NLL: 0.5 * [ (target - mu)^2 / exp(logvar) + logvar ]
    var = torch.exp(logvar)
    sq_err = (target - mu)**2
    nll = 0.5 * (sq_err / var + logvar)
    return (nll * scm).sum() / scm.sum()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--steps",type=int,default=5000)
    ap.add_argument("--bs",type=int,default=4)
    ap.add_argument("--tag",default="hetero")
    a=ap.parse_args()

    DEV=f"cuda:{a.gpu}"; C=2; IN=20
    MI=torch.tensor([0.154960856,-0.000513992854,0.0]).to(DEV); SI=torch.tensor([0.0968056545,0.015960684,1.0]).to(DEV)
    MT=torch.tensor([0.154962569,-0.000517793698,0.0]).to(DEV); ST=torch.tensor([0.0968104079,0.0159636438,1.0]).to(DEV)

    print("Loading dataset...")
    X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
    meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
    ntraj=len(lens)

    starts_tr=[]
    for i in range(ntraj):
        for t0 in range(off[i],off[i]+lens[i]-39):
            starts_tr.append(t0)
    starts_tr=np.array(starts_tr)
    print("train windows %d (ALL %d trajectories)"%(len(starts_tr),ntraj),flush=True)

    rng=np.random.default_rng(42)

    def batch(idx):
        w=np.stack([np.asarray(X[s:s+40]) for s in idx])
        z=np.zeros(w.shape[:-1]+(1,),np.float32)
        w=np.concatenate([w,z],-1)
        return torch.from_numpy(w[:,:IN]).to(DEV), torch.from_numpy(w[:,IN:]).to(DEV)

    print("Building HeteroFNO based on sim_real_fno.pth...")
    # 1. Load baseline model to get the FNO architecture and pretrained weights
    from load_baseline import MODEL_KWARGS
    #
    base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
    #
    #
    
    # 2. Wrap it in HeteroFNO
    model = HeteroFNO(base_model).to(DEV)
    model.train()
    
    # Freeze the base FNO layers for the first 1000 steps to let the variance head warm up?
    # No, let's train end-to-end but with a lower learning rate.
    
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    
    t0=time.time()
    for step in range(1,a.steps+1):
        idx=rng.choice(starts_tr,size=a.bs,replace=False)
        x,y=batch(idx)
        scm = (y[..., :C] != 0.0).float()
        
        # Forward
        x_norm = (x - MI) / SI
        mu, logvar = model(x_norm)
        mu = mu * ST + MT
        
        # The variance predicts the scale of the error in un-normalized space.
        # But wait, if we scale mu by ST, we should scale variance by ST^2!
        # logvar is predicted in normalized space.
        # So var_unnorm = exp(logvar) * (ST^2)
        # logvar_unnorm = log(exp(logvar) * ST^2) = logvar + 2*log(ST)
        logvar_unnorm = logvar[..., :C] + 2 * torch.log(ST[:C])
        
        # Loss
        loss = nll_loss_fn(mu[..., :C], logvar_unnorm, y[..., :C], scm)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if step % 100 == 0:
            print(f"step {step:5d} NLL: {float(loss):.4f} | {time.time()-t0:.0f}s", flush=True)

    out_path = f"{B}/local_harness/ft_{a.tag}_final.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
