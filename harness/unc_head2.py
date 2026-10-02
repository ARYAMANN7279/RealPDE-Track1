"""LEARNED PER-ELEMENT UNCERTAINTY HEAD, trained to maximise SPS directly.

Why: constant bounds are maxed (E~0.49 real). Top teams reach E~0.59 = 76% of
oracle; we are at 63%. The whole gap is per-element bound quality.

The SPS objective per element is   exp(-2h/sigma) * 1[|err| <= h]
The indicator is non-differentiable, so train on a smooth surrogate
   exp(-2h/sigma) * sigmoid((h - |err|)/tau)
with tau annealed down. h = softplus(net(features)) so h > 0 always.

Features are all inference-available: the FNO prediction, its spatial gradients,
vorticity, local temporal variance of the INPUT, KE, and normalised position.
Trained on train_real trajectories, validated on held-out ones.
"""
import json,os,sys,time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; DEV="cuda:2"
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); sims=sorted(set(sim.tolist()))
SCM=(Y[...,:C]!=0.0)
tr=np.isin(sim,sims[0::2]); te=~tr
print("train %d / test %d windows"%(tr.sum(),te.sum()),flush=True)

# ---- calibrate errors to the REAL scale (two wide anchors) ----
EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
m0,m1=SCM[...,0],SCM[...,1]; SUB=40
eu=EL[...,0][m0][::SUB]; ev=EL[...,1][m1][::SUB]
au=AP[...,0][m0][::SUB]; av=AP[...,1][m1][::SUB]; NT=eu.size+ev.size
best=None
for al in np.linspace(1.0,6.0,26):
    for be in np.linspace(-12,2,29):
        su,sv=eu*(al+be*au),ev*(al+be*av)
        Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
        hu,hv=0.05*au,0.05*av
        Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
        r=abs(Ec-0.4399)+abs(Ep-0.2076)
        if best is None or r<best[0]: best=(r,al,be)
_,AL,BE=best
print("calibration: alpha %.2f beta %.2f"%(AL,BE),flush=True)
ERR=(EL*(AL+BE*AP)).astype(np.float32)      # real-scale |error|, (N,20,32,64,2)

def feats(Pw):
    u,v=Pw[...,0],Pw[...,1]
    gux=np.gradient(u,axis=3); guy=np.gradient(u,axis=2)
    gvx=np.gradient(v,axis=3); gvy=np.gradient(v,axis=2)
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2)
    vo=np.abs(gvx-guy)
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tvv=np.repeat(v.std(axis=1,keepdims=True),v.shape[1],axis=1)
    ke=0.5*(u*u+v*v)
    dev=np.abs(u-u.mean(axis=1,keepdims=True))
    yy,xx=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,64),indexing="ij")
    yy=np.broadcast_to(yy,u.shape); xx=np.broadcast_to(xx,u.shape)
    tf=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    F_=np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(tv+1e-6),np.log(tvv+1e-6),
                 np.log(ke+1e-9),np.log(dev+1e-6),np.abs(u),np.abs(v),xx,yy,tf],axis=-1)
    return F_.astype(np.float32)
FT=feats(P)
NF=FT.shape[-1]
mu=FT[tr].reshape(-1,NF).mean(0); sd=FT[tr].reshape(-1,NF).std(0)+1e-6
FT=(FT-mu)/sd
print("features: %d channels"%NF,flush=True)

class Head(nn.Module):
    def __init__(s,nf,w=112):
        super().__init__()
        s.net=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
                            nn.Conv2d(w,w,3,padding=1,dilation=1),nn.GELU(),
                            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
                            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
                            nn.Conv2d(w,w,3,padding=1),nn.GELU(),
                            nn.Conv2d(w,2,1))
        s.b=nn.Parameter(torch.tensor([np.log(np.exp(0.013)-1),np.log(np.exp(0.010)-1)],dtype=torch.float32))
    def forward(s,x): return F.softplus(s.net(x)+s.b.view(1,2,1,1))+1e-5

def batches(mask,bs=8,shuffle=True):
    idx=np.where(mask)[0]
    if shuffle: np.random.default_rng(int(time.time())%1000).shuffle(idx)
    for i in range(0,len(idx),bs): yield idx[i:i+bs]

net=Head(NF).to(DEV)
opt=torch.optim.Adam(net.parameters(),lr=2e-3)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=22,eta_min=1e-4)
BEST={"E":-1.0}
def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
def eval_E(mask):
    tot=0.0; cnt=0.0
    with torch.no_grad():
        for idx in batches(mask,bs=8,shuffle=False):
            f=to_t(FT[idx]).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            h=net(f).reshape(len(idx),20,2,32,64).permute(0,1,3,4,2)
            e=to_t(ERR[idx]); m=to_t(SCM[idx].astype(np.float32))
            ok=(e<=h).float()*m
            tot+=float((torch.exp(-2*h/SIG)*ok).sum()); cnt+=float(m.sum())
    return tot/cnt
print("\nE before training (init ~ constant bounds): %.4f"%eval_E(te),flush=True)
TAU=[0.010]*3+[0.005]*3+[0.0025]*4+[0.0012]*4+[0.0006]*4+[0.0003]*4
step=0
for ep,tau in enumerate(TAU):
    for idx in batches(tr,bs=8):
        f=to_t(FT[idx]).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        h=net(f).reshape(len(idx),20,2,32,64).permute(0,1,3,4,2)
        e=to_t(ERR[idx]); m=to_t(SCM[idx].astype(np.float32))
        soft=torch.sigmoid((h-e)/tau)
        loss=-(torch.exp(-2*h/SIG)*soft*m).sum()/m.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); step+=1
    sched.step()
    Ete=eval_E(te); Etr=eval_E(tr)
    star=""
    if Ete>BEST["E"]:
        BEST["E"]=Ete
        torch.save({"sd":net.state_dict(),"mu":mu,"std":sd,"nf":NF},f"{B}/local_harness/unc_head_best.pth")
        star=" *"
    print("  ep %2d tau %.4f  E_train %.4f  E_test %.4f%s"%(ep,tau,Etr,Ete,star),flush=True)
print("\nBASELINE real-anchored: constant [0.0129,0.0098] -> E 0.4876 (sps 33.08, final 78.07)")
ck=torch.load(f"{B}/local_harness/unc_head_best.pth",map_location=DEV,weights_only=False)
net.load_state_dict(ck["sd"])
Efin=eval_E(te)
with torch.no_grad():
    _i=np.where(te)[0][:16]
    _f=to_t(FT[_i]).permute(0,1,4,2,3).reshape(-1,NF,32,64)
    _h=net(_f).cpu().numpy()
print("")
print("half-width: u med %.5f  v med %.5f   sd(log h_u) %.2f   (needs >~1.2)"%(
    float(np.median(_h[:,0])),float(np.median(_h[:,1])),float(np.std(np.log(_h[:,0]+1e-9)))))
W=0.6784
fin=lambda sps:0.306*94.168150+0.163*74.025866+0.218*92.836278+0.100*91.362375+0.217*sps
print("LEARNED HEAD E_test %.4f -> sps %.2f -> final %.2f  (%+.2f vs 78.07)"%(
    Efin,100*W*Efin,fin(100*W*Efin),fin(100*W*Efin)-78.068714))
torch.save({"sd":net.state_dict(),"mu":mu,"std":sd,"nf":NF},f"{B}/local_harness/unc_head.pth")
print("saved -> unc_head.pth")
