"""Train a REAL error predictor: raw input window -> log|error| of the frozen model.

sec27: the binding constraint is the uncertainty FEATURE, not the LUT. Our head reaches
log-space corr 0.54/0.50 (u/v) using 13 hand-crafted features; competitors imply
0.87-0.92. This is a deterministic supervised problem with ~60M labelled elements, so
the ceiling is corr -> 1, not an aleatoric floor.

Input : the 20-frame input window (40ch) + the frozen model's prediction (40ch)
Output: log|err| for u,v over 20 output frames (40ch)
Arch  : 2D U-Net over (32,64) with time folded into channels -- the error field is
        spatially structured and strongly time-correlated, so this is the natural form.

Reports the metric that actually matters (sec27.2): captured fraction of per-element
headroom, which needs NO local->real calibration.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline
SIG = 0.0563870259
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--width", type=int, default=64); ap.add_argument("--split", default="re_lohi")
ap.add_argument("--tag", default="e1")
a = ap.parse_args()
DEV = f"cuda:{a.gpu}"
torch.manual_seed(1234); np.random.seed(1234)
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]

CACHE = f"{B}/train_mvpe/runs/errnet_cache.npz"
meta = json.load(open(f"{LH}/tr_meta.json")); off,lens,names = meta["off"],meta["lens"],meta["names"]
ntraj = len(lens)
if not os.path.exists(CACHE):
    print("[cache] building predictions once...", flush=True)
    X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
    wins, wt = [], []
    for i in range(ntraj):
        for t0 in range(off[i], off[i]+lens[i]-39, 10):
            wins.append(t0); wt.append(i)
    wins = np.array(wins); wt = np.array(wt)
    model,_ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
    model.load_state_dict(torch.load(f"{LH}/soup_final_candidate.pth", map_location=DEV))
    model = model.to(DEV).eval()
    XI = np.zeros((len(wins),20,32,64,2), np.float32)
    PR = np.zeros((len(wins),20,32,64,2), np.float32)
    ER = np.zeros((len(wins),20,32,64,2), np.float32)
    SC = np.zeros((len(wins),20,32,64,2), bool)
    with torch.no_grad():
        for i in range(0, len(wins), 32):
            w = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40,32,64,1),np.float32)],-1)
                          for s in wins[i:i+32]]).astype(np.float32)
            xb = torch.from_numpy(w[:,:20]).to(DEV)
            p = (model((xb-MI)/SI)*ST+MT).cpu().numpy()
            XI[i:i+32] = w[:,:20,...,:2]; PR[i:i+32] = p[...,:2]
            ER[i:i+32] = np.abs(p[...,:2]-w[:,20:,...,:2]); SC[i:i+32] = (w[:,20:,...,:2]!=0.0)
            if i % 640 == 0: print("   %d/%d" % (i,len(wins)), flush=True)
    np.savez(CACHE, XI=XI, PR=PR, ER=ER, SC=SC, wt=wt)
    print("[cache] wrote", CACHE, flush=True)
d = np.load(CACHE)
XI,PR,ER,SC,wt = d["XI"],d["PR"],d["ER"],d["SC"],d["wt"]
RE = np.array([int(names[i].split("_")[0]) for i in wt])
VAL = np.isin(RE, [3750,5025,25425,26700]) if a.split=="re_lohi" else np.isin(wt, list(range(0,ntraj,5)))
TR = ~VAL
print("[data] %d windows | train %d | val %d (%s)" % (len(XI), TR.sum(), VAL.sum(), a.split), flush=True)

def flat(z): return torch.from_numpy(z).permute(0,1,4,2,3).reshape(z.shape[0],-1,32,64)
LOGE = np.log(ER+1e-6).astype(np.float32)
mu_,sd_ = LOGE[TR][SC[TR]].mean(), LOGE[TR][SC[TR]].std()
print("[data] log|err| mean %.3f sd %.3f" % (mu_,sd_), flush=True)

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return F.gelu(s.n2(s.c2(x)))
class UNet(nn.Module):
    def __init__(s,ci,co,w=64):
        super().__init__()
        s.e1=Blk(ci,w); s.e2=Blk(w,2*w); s.e3=Blk(2*w,4*w)
        s.b =Blk(4*w,4*w)
        s.d3=Blk(8*w,2*w); s.d2=Blk(4*w,w); s.d1=Blk(2*w,w)
        s.out=nn.Conv2d(w,co,1); s.pool=nn.AvgPool2d(2)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(s.pool(e1)); e3=s.e3(s.pool(e2)); b=s.b(s.pool(e3))
        u=F.interpolate(b,size=e3.shape[-2:],mode="bilinear",align_corners=False)
        d3=s.d3(torch.cat([u,e3],1))
        u=F.interpolate(d3,size=e2.shape[-2:],mode="bilinear",align_corners=False)
        d2=s.d2(torch.cat([u,e2],1))
        u=F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False)
        d1=s.d1(torch.cat([u,e1],1))
        return s.out(d1)
net = UNet(80,40,a.width).to(DEV)
print("[net] params %.2fM" % (sum(p.numel() for p in net.parameters())/1e6), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
idx_tr = np.where(TR)[0]; idx_va = np.where(VAL)[0]
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
        total_steps=a.epochs*max(1,len(idx_tr)//a.bs), pct_start=0.15)

def batch(ix):
    x = torch.cat([flat(XI[ix]), flat(PR[ix])],1).to(DEV)
    y = torch.from_numpy((LOGE[ix]-mu_)/sd_).permute(0,1,4,2,3).reshape(len(ix),-1,32,64).to(DEV)
    m = torch.from_numpy(SC[ix].astype(np.float32)).permute(0,1,4,2,3).reshape(len(ix),-1,32,64).to(DEV)
    return x,y,m

def captured(pred_log, true_err, scm, NB=24):
    """fraction of per-element headroom -- calibration-free (sec27.2)"""
    tot=Eor=Ebc=Ep=0.0
    for ci in range(2):
        e = true_err[...,ci][scm[...,ci]]; s = pred_log[...,ci][scm[...,ci]]
        if e.size==0: continue
        tot += e.size; Eor += float(np.exp(-2*e/SIG).sum())
        es = np.sort(e); g = np.linspace(0.0005,0.05,600); k = np.searchsorted(es,g,side="right")
        Ebc += float(np.max(np.exp(-2*g/SIG)*k))
        q = np.quantile(s, np.linspace(0,1,NB+1)[1:-1]); b = np.digitize(s,q)
        for kk in range(NB):
            ss = np.sort(e[b==kk])
            if ss.size==0: continue
            c = np.searchsorted(ss,g,side="right"); Ep += float(np.max(np.exp(-2*g/SIG)*c))
    return 100*(Ep-Ebc)/(Eor-Ebc), Ebc/tot, Eor/tot

@torch.no_grad()
def evaluate():
    net.eval(); P=[]
    for i in range(0,len(idx_va),16):
        x,_,_ = batch(idx_va[i:i+16])
        P.append(net(x).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    net.train()
    P = np.concatenate(P,0)
    E_ = ER[idx_va]; S_ = SC[idx_va]
    cu = np.corrcoef(np.log(E_[...,0][S_[...,0]]+1e-9), P[...,0][S_[...,0]])[0,1]
    cv = np.corrcoef(np.log(E_[...,1][S_[...,1]]+1e-9), P[...,1][S_[...,1]])[0,1]
    fr,_,_ = captured(P, E_, S_)
    return cu, cv, fr

print("\n%5s %9s %8s %8s %9s %8s" % ("epoch","loss","corr_u","corr_v","captured","sec"), flush=True)
t0=time.time(); best=-1
for ep in range(1, a.epochs+1):
    perm = idx_tr.copy(); np.random.default_rng(ep).shuffle(perm); tl=0.0; nb=0
    for i in range(0, len(perm)-a.bs+1, a.bs):
        x,y,m = batch(perm[i:i+a.bs])
        p = net(x); loss = ((p-y).abs()*m).sum()/m.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        tl+=float(loss); nb+=1
    if ep%2==0 or ep==a.epochs:
        cu,cv,fr = evaluate()
        flag=""
        if fr>best: best=fr; torch.save({"sd":net.state_dict(),"mu":mu_,"sd_":sd_,"w":a.width},
                                        f"{B}/train_mvpe/runs/errnet_{a.tag}.pth"); flag=" *"
        print("%5d %9.4f %8.3f %8.3f %8.1f%% %8.0f%s"%(ep,tl/nb,cu,cv,fr,time.time()-t0,flag), flush=True)
print("\nBEST captured %.1f%%  (shipped head: 18-22%% ceiling, corr 0.54/0.50)"%best, flush=True)
print("targets: 33%% = benslash2 (final 79.66) | 43%% = np-user (final 80.09)", flush=True)
