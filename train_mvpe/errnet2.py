"""Uncertainty head, take 2 -- attack the structural limit, not the capacity.

errnet v1 raised log-space corr 0.54->0.66 but captured stayed at 22%, the same
ceiling as the old feature. Reason: a POINT estimate of log|err| plus a global 24-bin
lookup assumes every element shares one conditional error SHAPE. The optimal half-width
depends on the whole conditional distribution, not its centre.

Modes:
  logerr    v1 baseline: point estimate + global LUT.
  quantile  predict K quantiles of |err| per element (pinball loss). Then the optimal
            h is available IN CLOSED FORM per element, with NO LUT and no binning:
                h* = argmax_j  exp(-2 q_j / sigma) * tau_j
            since F(q_j) = tau_j by construction. This removes both the shared-shape
            assumption and the 24-bin discretisation at once.
  directh   predict h directly against a smooth surrogate of the SPS integrand.

--rich adds engineered channels (gradients, vorticity, temporal std, KE) on top of the
raw window + prediction, since those are what the old hand-crafted head actually used.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
SIG = 0.0563870259
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--width", type=int, default=96)
ap.add_argument("--mode", default="quantile", choices=["logerr","quantile","directh"])
ap.add_argument("--rich", action="store_true")
ap.add_argument("--split", default="re_lohi"); ap.add_argument("--tag", default="q1")
a = ap.parse_args()
DEV = f"cuda:{a.gpu}"
torch.manual_seed(1234); np.random.seed(1234)
TAUS = np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.92,0.96,0.99], np.float32)
NQ = len(TAUS)

d = np.load(f"{B}/train_mvpe/runs/errnet_cache.npz")
XI,PR,ER,SC,wt = d["XI"],d["PR"],d["ER"],d["SC"],d["wt"]
meta = json.load(open(f"{LH}/tr_meta.json")); names = meta["names"]; ntraj = len(meta["lens"])
RE = np.array([int(names[i].split("_")[0]) for i in wt])
VAL = np.isin(RE,[3750,5025,25425,26700]) if a.split=="re_lohi" else np.isin(wt,list(range(0,ntraj,5)))
TR = ~VAL
print("[data] %d windows | train %d | val %d (%s) | mode=%s rich=%s w=%d"
      % (len(XI),TR.sum(),VAL.sum(),a.split,a.mode,a.rich,a.width), flush=True)

def rich_feats(p):
    """engineered channels from the prediction, (N,20,32,64,2) -> (N,20,32,64,K)"""
    u,v = p[...,0], p[...,1]
    gux=np.gradient(u,axis=3); guy=np.gradient(u,axis=2)
    gvx=np.gradient(v,axis=3); gvy=np.gradient(v,axis=2)
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2); vo=np.abs(gvx-guy)
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tvv=np.repeat(v.std(axis=1,keepdims=True),v.shape[1],axis=1)
    ke=0.5*(u*u+v*v)
    dev=np.abs(u-u.mean(axis=1,keepdims=True)); dv=np.abs(v-v.mean(axis=1,keepdims=True))
    return np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(tv+1e-6),np.log(tvv+1e-6),
                     np.log(ke+1e-9),np.log(dev+1e-6),np.log(dv+1e-6)],-1).astype(np.float32)
RF = None
if a.rich:
    print("[feat] computing rich channels...", flush=True)
    RF = np.concatenate([rich_feats(PR[i:i+256]) for i in range(0,len(PR),256)],0)
    m_,s_ = RF[TR].reshape(-1,RF.shape[-1]).mean(0), RF[TR].reshape(-1,RF.shape[-1]).std(0)+1e-6
    RF = ((RF-m_)/s_).astype(np.float32)
    print("[feat] rich shape", RF.shape, flush=True)
CI = 80 + (RF.shape[-1]*20 if RF is not None else 0)
LOGE = np.log(ER+1e-6).astype(np.float32)
mu_,sd_ = float(LOGE[TR][SC[TR]].mean()), float(LOGE[TR][SC[TR]].std())

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return F.gelu(s.n2(s.c2(x)))
class UNet(nn.Module):
    def __init__(s,ci,co,w):
        super().__init__()
        s.e1=Blk(ci,w); s.e2=Blk(w,2*w); s.e3=Blk(2*w,4*w); s.b=Blk(4*w,4*w)
        s.d3=Blk(8*w,2*w); s.d2=Blk(4*w,w); s.d1=Blk(2*w,w); s.out=nn.Conv2d(w,co,1)
        s.pool=nn.AvgPool2d(2)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(s.pool(e1)); e3=s.e3(s.pool(e2)); b=s.b(s.pool(e3))
        u=F.interpolate(b,size=e3.shape[-2:],mode="bilinear",align_corners=False); d3=s.d3(torch.cat([u,e3],1))
        u=F.interpolate(d3,size=e2.shape[-2:],mode="bilinear",align_corners=False); d2=s.d2(torch.cat([u,e2],1))
        u=F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False); d1=s.d1(torch.cat([u,e1],1))
        return s.out(d1)
CO = 40*(NQ if a.mode=="quantile" else 1)
net = UNet(CI,CO,a.width).to(DEV)
print("[net] in %d out %d params %.2fM"%(CI,CO,sum(p.numel() for p in net.parameters())/1e6), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
idx_tr=np.where(TR)[0]; idx_va=np.where(VAL)[0]
sch = torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,
        total_steps=a.epochs*max(1,len(idx_tr)//a.bs),pct_start=0.15)
tq = torch.tensor(TAUS,device=DEV).view(1,1,NQ,1,1)

def flat(z): return torch.from_numpy(np.ascontiguousarray(z)).permute(0,1,4,2,3).reshape(z.shape[0],-1,32,64)
def batch(ix):
    parts=[flat(XI[ix]),flat(PR[ix])]
    if RF is not None: parts.append(flat(RF[ix]))
    x=torch.cat(parts,1).to(DEV)
    e=torch.from_numpy(np.ascontiguousarray(ER[ix])).permute(0,1,4,2,3).reshape(len(ix),-1,32,64).to(DEV)
    m=torch.from_numpy(SC[ix].astype(np.float32)).permute(0,1,4,2,3).reshape(len(ix),-1,32,64).to(DEV)
    return x,e,m

def loss_fn(p,e,m):
    if a.mode=="logerr":
        y=(torch.log(e+1e-6)-mu_)/sd_; return ((p-y).abs()*m).sum()/m.sum()
    if a.mode=="quantile":
        q=p.reshape(-1,40,NQ,32,64); tgt=e.reshape(-1,40,1,32,64); mm=m.reshape(-1,40,1,32,64)
        q=F.softplus(q)*0.05
        d_=tgt-q; pin=torch.maximum(tq*d_,(tq-1.0)*d_)
        return (pin*mm).sum()/(mm.sum()*NQ)
    h=F.softplus(p)*0.05
    return -((torch.exp(-2*h/SIG)*torch.sigmoid((h-e)/0.0015))*m).sum()/m.sum()

def captured_from(hpred, err, scm):
    """h is chosen PER ELEMENT -- no LUT, no binning"""
    tot=Eor=Ebc=Ep=0.0
    for ci in range(2):
        e=err[...,ci][scm[...,ci]]; h=hpred[...,ci][scm[...,ci]]
        if e.size==0: continue
        tot+=e.size; Eor+=float(np.exp(-2*e/SIG).sum())
        es=np.sort(e); g=np.linspace(0.0005,0.05,700); k=np.searchsorted(es,g,side="right")
        Ebc+=float(np.max(np.exp(-2*g/SIG)*k))
        Ep+=float((np.exp(-2*h/SIG)*(e<=h)).sum())
    return 100*(Ep-Ebc)/(Eor-Ebc)
def captured_lut(s_, err, scm, NB=24):
    tot=Eor=Ebc=Ep=0.0
    for ci in range(2):
        e=err[...,ci][scm[...,ci]]; s=s_[...,ci][scm[...,ci]]
        if e.size==0: continue
        tot+=e.size; Eor+=float(np.exp(-2*e/SIG).sum())
        es=np.sort(e); g=np.linspace(0.0005,0.05,700); k=np.searchsorted(es,g,side="right")
        Ebc+=float(np.max(np.exp(-2*g/SIG)*k))
        q=np.quantile(s,np.linspace(0,1,NB+1)[1:-1]); b=np.digitize(s,q)
        for kk in range(NB):
            ss=np.sort(e[b==kk])
            if ss.size==0: continue
            c=np.searchsorted(ss,g,side="right"); Ep+=float(np.max(np.exp(-2*g/SIG)*c))
    return 100*(Ep-Ebc)/(Eor-Ebc)

@torch.no_grad()
def evaluate():
    net.eval(); H=[]; S=[]
    for i in range(0,len(idx_va),8):
        x,_,_=batch(idx_va[i:i+8]); p=net(x)
        if a.mode=="quantile":
            q=F.softplus(p.reshape(-1,40,NQ,32,64))*0.05
            val=torch.exp(-2*q/SIG)*tq                       # objective at each quantile
            j=val.argmax(dim=2,keepdim=True)
            h=torch.gather(q,2,j).squeeze(2)
            S.append(torch.gather(q,2,torch.full_like(j,NQ//2)).squeeze(2)
                     .reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
        elif a.mode=="directh":
            h=F.softplus(p)*0.05; S.append(h.reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
        else:
            h=p; S.append(p.reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
        H.append(h.reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    net.train()
    H=np.concatenate(H,0); S=np.concatenate(S,0); E_=ER[idx_va]; M_=SC[idx_va]
    cu=np.corrcoef(np.log(E_[...,0][M_[...,0]]+1e-9),S[...,0][M_[...,0]])[0,1]
    cv=np.corrcoef(np.log(E_[...,1][M_[...,1]]+1e-9),S[...,1][M_[...,1]])[0,1]
    direct = captured_from(H,E_,M_) if a.mode!="logerr" else float("nan")
    return cu,cv,direct,captured_lut(S,E_,M_)

print("\n%5s %9s %8s %8s %11s %10s %7s"%("epoch","loss","corr_u","corr_v","direct-h","via LUT","sec"),flush=True)
t0=time.time(); best=-1
for ep in range(1,a.epochs+1):
    perm=idx_tr.copy(); np.random.default_rng(ep).shuffle(perm); tl=0.0; nb=0
    for i in range(0,len(perm)-a.bs+1,a.bs):
        x,e,m=batch(perm[i:i+a.bs]); p=net(x); loss=loss_fn(p,e,m)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        tl+=float(loss.detach()); nb+=1
    if ep%2==0 or ep==a.epochs:
        cu,cv,dh,lu=evaluate(); sc=max(dh if dh==dh else -1, lu); flag=""
        if sc>best:
            best=sc; torch.save({"sd":net.state_dict(),"mode":a.mode,"w":a.width,"rich":a.rich,
                                 "mu":mu_,"sd":sd_,"taus":TAUS}, f"{B}/train_mvpe/runs/errnet_{a.tag}.pth"); flag=" *"
        print("%5d %9.5f %8.3f %8.3f %10.1f%% %9.1f%% %7.0f%s"
              %(ep,tl/nb,cu,cv,dh,lu,time.time()-t0,flag),flush=True)
print("\nBEST %.1f%%   (old head 18-22%% | benslash2 33%% | np-user 43%%)"%best,flush=True)
