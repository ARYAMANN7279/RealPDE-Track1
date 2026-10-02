import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--arm", type=str, required=True, choices=["A", "B", "C", "D", "E", "F"])
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
sys.path.insert(0, B)
import r12_eval as EV

KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline

torch.manual_seed(args.seed)
np.random.seed(args.seed)
import random; random.seed(args.seed)
DEV = f"cuda:{args.gpu}"

if args.arm == "A" and args.seed == 0:
    EV.selftest()

meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
RE = np.array([int(n.split("_")[0]) for n in names])
AOA = np.array([float(n.split("_")[1].split(".")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

starts_tr = []
traj_idx = []
for i in range(ntraj):
    for t0 in range(off[i], off[i]+lens[i]-39):
        if i not in VT:
            starts_tr.append(t0)
            traj_idx.append(i)
starts_tr = np.array(starts_tr)
traj_idx = np.array(traj_idx)

sub, _ = EV.val_starts(900)

model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
model = model.to(DEV)

if args.arm == "D":
    def _mk(p):
        def hook(mod, inp, out): return F.dropout(out, p=p, training=mod.training)
        return hook
    for m in model.modules():
        if "SpectralConv" in type(m).__name__:
            m.register_forward_hook(_mk(0.1))
    model.load_state_dict(torch.load(f"{B}/data/comp_real/sim_real_fno.pth", map_location=DEV), strict=True)
    print("Arm D: strict loading verified with forward hooks.")

class EMA():
    def __init__(self, beta):
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.state_dict().values(), ma_model.state_dict().values()):
            if current_params.is_complex():
                ma_params.mul_(self.beta).add_(current_params.detach(), alpha=1-self.beta)
            elif current_params.dtype.is_floating_point:
                ma_params.mul_(self.beta).add_(current_params.detach().float(), alpha=1-self.beta)
            else:
                ma_params.copy_(current_params.detach())

if args.arm in ["B", "C"]:
    ema_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
    ema_model = ema_model.to(DEV)
    ema = EMA(0.999 if args.arm == "B" else 0.9999)

last_pooled = None
if args.arm == "F":
    aux_head = torch.nn.Linear(model.fno.width, 2).to(DEV)
    aux_opt = torch.optim.Adam(aux_head.parameters(), lr=3e-5)
    def aux_hook(mod, inp, out):
        global last_pooled
        last_pooled = out.mean(dim=(2,3,4))
    for name, m in model.named_modules():
        if name == "fno.convs.3":
            m.register_forward_hook(aux_hook)
    
num_update = 12000
batch_size = 16
eval_freq = 500

opt_cls = torch.optim.AdamW if args.arm == "E" else torch.optim.Adam
wd = 1e-3 if args.arm == "E" else 1e-6
opt = opt_cls(model.parameters(), lr=3e-5, weight_decay=wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_update)

mi = EV.MI.view(1,1,1,1,3).to(DEV)
si = EV.SI.view(1,1,1,1,3).to(DEV)
mt = EV.MT.view(1,1,1,1,3).to(DEV)
st = EV.ST.view(1,1,1,1,3).to(DEV)

def fwd(x):
    xt_norm = (x - mi) / si
    out = model(xt_norm)
    return out * st + mt

def evaluate():
    eval_m = ema_model if args.arm in ["B", "C"] else model
    eval_m.eval()
    def _fwd(x):
        xt_norm = (x - mi) / si
        out = eval_m(xt_norm)
        return out * st + mt
    return EV.eval_model(_fwd, sub, bs=16)

def rel_l2(p,t):
    p=p[...,:2].reshape(p.shape[0],-1); t=t[...,:2].reshape(t.shape[0],-1)
    return (torch.linalg.norm(p-t,dim=1)/torch.linalg.norm(t,dim=1).clamp(min=1e-8)).mean()

def tke_l2(p,t):
    def ke(z):
        u,v=z[...,0],z[...,1]
        return 0.5*(((u-u.mean(1,keepdim=True))**2).mean(1)+((v-v.mean(1,keepdim=True))**2).mean(1))
    pk=ke(p[...,:2]).reshape(p.shape[0],-1); tk=ke(t[...,:2]).reshape(t.shape[0],-1)
    return (torch.linalg.norm(pk-tk,dim=1)/torch.linalg.norm(tk,dim=1).clamp(min=1e-8)).mean()

print(f"Starting Arm {args.arm} Seed {args.seed} Training...")
tr_rng = np.random.default_rng(args.seed + 100)
best_dacc, best_step = -100, -1
results = []
t0 = time.time()

for step in range(1, num_update + 1):
    model.train()
    if args.arm == "F": aux_head.train()
    
    idx_choice = tr_rng.choice(len(starts_tr), size=batch_size, replace=False)
    idx = starts_tr[idx_choice]
    tr_id = traj_idx[idx_choice]
    
    x, y = EV.window(idx)
    xt = torch.from_numpy(x).to(DEV)
    yt = torch.from_numpy(y).to(DEV)
    
    p = fwd(xt)
    loss = rel_l2(p, yt) + 0.15 * tke_l2(p, yt)
    
    if args.arm == "F":
        aux_target = torch.stack([torch.tensor(RE[tr_id])/15000.0, torch.tensor(AOA[tr_id])/10.0], dim=1).float().to(DEV)
        aux_pred = aux_head(last_pooled)
        loss = loss + 0.1 * F.mse_loss(aux_pred, aux_target)
        
    opt.zero_grad(set_to_none=True)
    if args.arm == "F": aux_opt.zero_grad(set_to_none=True)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    opt.step()
    if args.arm == "F": aux_opt.step()
    sched.step()
    
    if args.arm in ["B", "C"]:
        ema.update_model_average(ema_model, model)
    
    if step % eval_freq == 0:
        r = evaluate()
        dt = time.time() - t0
        print(f"Step {step:5d} | Loss: {loss.item():.5f} | rel_l2: {r['rel_l2']:.4f} | tke: {r['tke']:.4f} | mvpe: {r['mvpe']:.4f} | d_acc: {r['d_acc']:.4f} | time: {dt:.1f}s", flush=True)
        results.append((step, r['rel_l2'], r['tke'], r['mvpe'], r['d_acc']))
        if r['d_acc'] > best_dacc:
            best_dacc = r['d_acc']
            best_step = step
            save_m = ema_model if args.arm in ["B", "C"] else model
            torch.save(save_m.state_dict(), f"{B}/train_es/r13_{args.arm}_s{args.seed}.pth")

print(f"Arm {args.arm} Seed {args.seed} completed. Best d_acc = {best_dacc:.4f} at step {best_step}")
res_dict = {"results": results, "best_step": best_step, "best_dacc": best_dacc}
with open(f"{B}/r13_{args.arm}_s{args.seed}.json", "w") as f:
    json.dump(res_dict, f)
