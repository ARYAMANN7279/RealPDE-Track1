import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))

import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

from u_fno import UFNO

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--wtke", type=float, default=0.33)
ap.add_argument("--tag", default="nll_v1")
a = ap.parse_args()

DEV = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
C = 2; IN = 20; OUT = 20

# Load Data
print("Loading data...")
X_npy = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off = meta["off"]; lens = meta["lens"]; names = meta["names"]
ntraj = len(lens); vidx = set(range(0, ntraj, 5))
starts_tr = []; starts_va = []
for i in range(ntraj):
    for t0 in range(off[i], off[i]+lens[i]-39):
        (starts_va if i in vidx else starts_tr).append(t0)
starts_tr = np.array(starts_tr)
starts_va = np.array(starts_va)
print(f"train windows {len(starts_tr)} | val windows {len(starts_va)}", flush=True)

rng = np.random.default_rng(0)
va_sub = rng.choice(starts_va, size=min(600, len(starts_va)), replace=False)

MI=torch.tensor([0.154960856,-0.000513992854,0.0]).to(DEV)
SI=torch.tensor([0.0968056545,0.015960684,1.0]).to(DEV)
MT=torch.tensor([0.154962569,-0.000517793698,0.0]).to(DEV)
ST=torch.tensor([0.0968104079,0.0159636438,1.0]).to(DEV)

def batch(idx):
    w = np.stack([np.asarray(X_npy[s:s+40]) for s in idx])
    return torch.from_numpy(w[:, :IN]).to(DEV), torch.from_numpy(w[:, IN:]).to(DEV)

# 1. Model
model = UFNO(in_c=40, out_c=40, w=64).to(DEV)

def tke_l2(p, t):
    def ke(z):
        u, v = z[...,0], z[...,1]
        return 0.5 * (((u - u.mean(1, keepdim=True))**2).mean(1) + ((v - v.mean(1, keepdim=True))**2).mean(1))
    pk = ke(p).reshape(p.shape[0], -1)
    tk = ke(t).reshape(t.shape[0], -1)
    return (torch.linalg.norm(pk - tk, dim=1) / torch.linalg.norm(tk, dim=1).clamp(min=1e-8)).mean()

@torch.no_grad()
def evaluate():
    model.eval()
    P = []; T = []
    for i in range(0, len(va_sub), 16):
        x, y = batch(va_sub[i:i+16])
        x_norm = (x - MI[:2]) / SI[:2]
        x_flat = x_norm.permute(0, 1, 4, 2, 3).reshape(x.shape[0], -1, 32, 64)
        mu, logvar = model(x_flat)
        mu_res = mu.view(mu.shape[0], 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        mu_res = mu_res * ST[:2] + MT[:2]

        mu_pad = torch.zeros(mu_res.shape[:-1] + (1,), device=DEV)
        p_corr = torch.cat([mu_res, mu_pad], dim=-1)
        
        y_pad = torch.zeros(y.shape[:-1] + (1,), device=DEV)
        y_val = torch.cat([y, y_pad], dim=-1)
        
        P.append(p_corr.cpu().numpy()); T.append(y_val.cpu().numpy())
    model.train()
    P = np.concatenate(P, 0).astype(np.float32)
    T = np.concatenate(T, 0).astype(np.float32)
    dm = S.rel_l2_per_sample(P, T, C)
    tk = S.tke_rel_l2_per_sample(P, T, C)
    mv = S.mvpe_rel_l2_per_sample(P, T)
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))

def proj(r): return 0.306*r[0] + 0.163*r[1] + 0.218*r[2]

base = evaluate()
b0 = proj(base)
print("Untrained val: rel_l2 %.2f tke %.2f mvpe %.2f"%base, flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

best = (b0, base, 0)
t0 = time.time()
model.train()

for step in range(1, a.steps+1):
    idx = rng.choice(starts_tr, size=a.bs, replace=False)
    x, y = batch(idx)
    
    x_norm = (x - MI[:2]) / SI[:2]
    x_flat = x_norm.permute(0, 1, 4, 2, 3).reshape(x.shape[0], -1, 32, 64)
    y_flat = y.permute(0, 1, 4, 2, 3).reshape(y.shape[0], -1, 32, 64)
    
    mu_norm, logvar = model(x_flat)
    
    mu_norm_res = mu_norm.view(mu_norm.shape[0], 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
    mu_res = mu_norm_res * ST[:2] + MT[:2]
    mu_unnorm_flat = mu_res.permute(0, 1, 4, 2, 3).reshape(y.shape[0], -1, 32, 64)
    
    # Gaussian NLL
    # Note: since y is unnormalized, we want logvar to predict the variance in unnormalized space.
    # Therefore, we compute NLL on the unnormalized space.
    nll_loss = ((y_flat - mu_unnorm_flat)**2 / (2 * torch.exp(logvar)) + 0.5 * logvar).mean()
    
    # TKE loss
    tke_loss = tke_l2(mu_res, y)
    
    loss = nll_loss + a.wtke * tke_loss
    
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    
    if step % 100 == 0 or step == a.steps:
        r = evaluate()
        flag = ""
        if proj(r) > best[0]:
            best = (proj(r), r, step)
            torch.save(model.state_dict(), f"{B}/train_es/{a.tag}_best.pth")
            flag = " *saved"
        print("step %5d loss %.4f (nll %.4f tke %.4f) | val rel_l2 %.2f tke %.2f mvpe %.2f | scr %.3f | %.0fs%s" % (
            step, float(loss), float(nll_loss), float(tke_loss), r[0], r[1], r[2], proj(r), time.time() - t0, flag), flush=True)

print("BEST at step %d: rel_l2 %.2f tke %.2f mvpe %.2f  (score %.3f)" % (
    best[2], best[1][0], best[1][1], best[1][2], best[0]), flush=True)
