"""Measure bound-head TRANSFER under Reynolds EXTRAPOLATION.

The whole gap to 79 is the head's transfer loss: its local edge over constants is
+0.0540 but only +0.0054 (~10%) survives on the live board. If transfer were ~75%
we would be at 78.98; at 100%, 79.18. No measured signal can get there (they cap
at ~78.9 even assuming perfect transfer), so improving TRANSFER is the only route.

Why our local numbers lie: the held-out split is random BY TRAJECTORY, so train and
test share Reynolds regimes. The rules say the private test uses "unseen parameter
regimes (e.g., novel angles of attack or Reynolds numbers)". This builds that split
-- train on interior Re, test on the LOWEST and HIGHEST Re -- and asks which head
capacity keeps the most edge when it must extrapolate. Overfit capacity should show
a big random-split edge that collapses here; that collapse is the thing to minimise.
"""
import json, os, sys, collections
import numpy as np, torch, torch.nn as nn
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sys.path.insert(0, f"{B}/train_maxsoup")
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
from head_common import feats, global_scale
SIG = S.SIGMA_GLOBAL; DEV = "cuda:0" if torch.cuda.is_available() else "cpu"; C = 2
torch.manual_seed(1234); np.random.seed(1234)

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(DEV) for x in (MI, SI, MT, ST)]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
W = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40,32,64,1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = W[:, :20], W[:, 20:]; SCM = (Y[..., :C] != 0.0)

RE = np.array([int(names[i].split("_")[0]) for i in wt])
uniq_re = sorted(set(RE.tolist()))
lo_re, hi_re = uniq_re[:3], uniq_re[-3:]
EXTRAP = np.isin(RE, lo_re + hi_re)            # unseen regimes
TRAIN = ~EXTRAP
RANDOM_EV = np.isin(wt, sorted(set(range(0, ntraj, 5))))   # the old, easy split
print("Re extrapolation split: train on %d interior Re, test on %s + %s"
      % (len(uniq_re) - 6, lo_re, hi_re))
print("  train windows %d | extrapolation windows %d | (old random split ev=%d)"
      % (TRAIN.sum(), EXTRAP.sum(), RANDOM_EV.sum()))

base, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
base.load_state_dict(torch.load(f"{LH}/soup_final_candidate.pth", map_location=DEV))
base = base.to(DEV).eval()
P = []
with torch.no_grad():
    for i in range(0, len(Xin), 32):
        xb = torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
        P.append((base((xb - mi)/si)*st + mt).cpu().numpy())
P = np.concatenate(P, 0).astype(np.float32)
ERR = np.abs(P[..., :C] - Y[..., :C]).astype(np.float32)
SCALE = global_scale(ERR, SCM)
for ci in range(C): ERR[..., ci] *= SCALE[ci]
LOGE = np.log(ERR + 1e-6).astype(np.float32)
FT = feats(P); NF = FT.shape[-1]
mu_ = FT[TRAIN].reshape(-1, NF).mean(0); sd_ = FT[TRAIN].reshape(-1, NF).std(0) + 1e-6
FT = (FT - mu_) / sd_

def make_head(kind):
    if kind == "linear":
        return nn.Conv2d(NF, 2, 1)
    if kind == "tiny":
        return nn.Sequential(nn.Conv2d(NF,16,3,padding=1), nn.GELU(), nn.Conv2d(16,2,1))
    if kind == "small":
        return nn.Sequential(nn.Conv2d(NF,32,3,padding=1), nn.GELU(),
                             nn.Conv2d(32,32,3,padding=2,dilation=2), nn.GELU(), nn.Conv2d(32,2,1))
    w = 128   # "full" == the shipped architecture
    return nn.Sequential(nn.Conv2d(NF,w,3,padding=1), nn.GELU(),
        nn.Conv2d(w,w,3,padding=2,dilation=2), nn.GELU(),
        nn.Conv2d(w,w,3,padding=4,dilation=4), nn.GELU(),
        nn.Conv2d(w,w,3,padding=8,dilation=8), nn.GELU(),
        nn.Conv2d(w,w,3,padding=1), nn.GELU(), nn.Conv2d(w,2,1))

def train_head(kind, epochs=14):
    net = make_head(kind).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)
    idx_all = np.where(TRAIN)[0]
    for ep in range(epochs):
        idx = idx_all.copy(); np.random.default_rng(1234+ep).shuffle(idx)
        for i in range(0, len(idx), 8):
            j = idx[i:i+8]
            f = torch.from_numpy(np.ascontiguousarray(FT[j])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            p = net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2)
            t = torch.from_numpy(np.ascontiguousarray(LOGE[j])).to(DEV)
            m = torch.from_numpy(SCM[j].astype(np.float32)).to(DEV)
            loss = ((p-t).abs()*m).sum()/m.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sch.step()
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(P), 8):
            f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            out.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out, 0), sum(p.numel() for p in net.parameters())

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def E_const_on(mask, hu=0.0129, hv=0.0098):
    eu = ERR[mask][...,0][SCM[mask][...,0]]; ev_ = ERR[mask][...,1][SCM[mask][...,1]]
    return float((np.exp(-2*hu/SIG)*(eu<=hu)).sum()+(np.exp(-2*hv/SIG)*(ev_<=hv)).sum())/(eu.size+ev_.size)

def E_head_on(MU, fitmask, evmask, NB=24):
    num=0.0; tot=0
    for ci in range(C):
        mf = SCM[...,ci] & fitmask[:,None,None,None]; me = SCM[...,ci] & evmask[:,None,None,None]
        sf = MU[...,ci][mf]; ef = ERR[...,ci][mf]
        q = np.quantile(sf, np.linspace(0,1,NB+1)[1:-1]); b = np.digitize(sf, q)
        LUT = np.array([best_h(ef[b==k][::3]) if (b==k).sum()>200 else 0.012 for k in range(NB)], np.float32)
        h = LUT[np.digitize(MU[...,ci][me], q)]; ee = ERR[...,ci][me]
        num += float((np.exp(-2*h/SIG)*(ee<=h)).sum()); tot += ee.size
    return num/tot

print("\n%-8s %9s | %-22s | %-24s | %s" % ("head","params","RANDOM split (easy)","Re-EXTRAP split (real)","edge kept"))
print("-"*104)
ec_rand = E_const_on(RANDOM_EV); ec_ext = E_const_on(EXTRAP)
res={}
for kind in ("linear","tiny","small","full"):
    MU, npar = train_head(kind)
    e_rand = E_head_on(MU, ~RANDOM_EV, RANDOM_EV)
    e_ext  = E_head_on(MU, TRAIN, EXTRAP)
    edge_r = e_rand-ec_rand; edge_e = e_ext-ec_ext
    keep = (edge_e/edge_r*100) if edge_r>0 else 0.0
    res[kind]=(edge_r,edge_e,keep,npar)
    print("%-8s %9d | edge %+.4f          | edge %+.4f            | %5.1f%%"%(kind,npar,edge_r,edge_e,keep))
print("\nconstants baseline: E %.4f (random)  %.4f (extrapolation)"%(ec_rand,ec_ext))
print("\nshipped head realises ~10%% of its random-split edge on the live board.")
print("If a head keeps a LARGER FRACTION under Re-extrapolation, it should transfer better.")
best=max(res.items(), key=lambda kv: kv[1][1])
print("\nBEST under extrapolation: %s (edge %+.4f, keeps %.1f%%)"%(best[0],best[1][1],best[1][2]))
W_REAL=0.684538; E_now=0.502026; E_const_real=E_now-0.0054
for kind,(er,ee,keep,npar) in res.items():
    proj=E_const_real+ee
    print("  %-8s projected real E %.4f -> final %.2f"%(kind,proj,78.4566+(proj-E_now)*100*W_REAL*0.217))
