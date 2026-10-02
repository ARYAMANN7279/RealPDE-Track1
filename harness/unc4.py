"""Uncertainty head v4 = v3's learned error map + the CORRECT h mapping.

v3 learned sd(mu)=0.97 of real spatial structure but applied h = k*exp(mu), which
is wrong: for high-error elements the SPS optimum is to GIVE UP (small h), because
exp(-2h/sigma) decays faster than coverage grows. The optimal h vs mu curve
SATURATES rather than scaling.

So: bin the learned mu into quantiles and optimise h per bin on TRAIN (exactly, by
sorting each bin's errors), then apply that lookup to TEST. Non-parametric, so it
finds whatever the true shape is.
"""
import json,os,sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; DEV="cuda:2"; W=0.6784
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); sims=sorted(set(sim.tolist()))
SCM=(Y[...,:C]!=0.0); tr=np.isin(sim,sims[0::2]); te=~tr
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
ERR=(EL*(AL+BE*AP)).astype(np.float32)
def feats(Pw):
    u,v=Pw[...,0],Pw[...,1]
    gux=np.gradient(u,axis=3); guy=np.gradient(u,axis=2)
    gvx=np.gradient(v,axis=3); gvy=np.gradient(v,axis=2)
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2); vo=np.abs(gvx-guy)
    lap=np.abs(np.gradient(gux,axis=3)+np.gradient(guy,axis=2))
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tvv=np.repeat(v.std(axis=1,keepdims=True),v.shape[1],axis=1)
    ke=0.5*(u*u+v*v); dev=np.abs(u-u.mean(axis=1,keepdims=True))
    dv=np.abs(v-v.mean(axis=1,keepdims=True))
    yy,xx=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,64),indexing="ij")
    yy=np.broadcast_to(yy,u.shape); xx=np.broadcast_to(xx,u.shape)
    tf=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    return np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(lap+1e-6),np.log(tv+1e-6),
                     np.log(tvv+1e-6),np.log(ke+1e-9),np.log(dev+1e-6),np.log(dv+1e-6),
                     u,v,xx,yy,tf],axis=-1).astype(np.float32)
class Net(nn.Module):
    def __init__(s,nf,w=128):
        super().__init__()
        s.n=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
            nn.Conv2d(w,w,3,padding=8,dilation=8),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,2,1))
    def forward(s,x): return s.n(x)
ck=torch.load(f"{B}/local_harness/unc3_best.pth",map_location=DEV,weights_only=False)
NF=ck["nf"]; net=Net(NF).to(DEV); net.load_state_dict(ck["sd"]); net.eval()
FT=(feats(P)-ck["mu"])/ck["std"]
def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
def mu_of(mask,bs=8):
    out=[]; idx=np.where(mask)[0]
    with torch.no_grad():
        for i in range(0,len(idx),bs):
            j=idx[i:i+bs]
            f=to_t(FT[j]).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            out.append(net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out,0)
MUtr=mu_of(tr); MUte=mu_of(te)
print("learned mu: sd %.2f (v2 head had 0.36)"%MUtr[SCM[tr]].std(),flush=True)
NB=24
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.01
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
LUT=np.zeros((NB,C),np.float32); EDGES=[]
for ci in range(C):
    m=SCM[tr][...,ci]
    mu=MUtr[...,ci][m]; er=ERR[tr][...,ci][m]
    q=np.quantile(mu,np.linspace(0,1,NB+1)[1:-1]); EDGES.append(q)
    b=np.digitize(mu,q)
    for k in range(NB):
        sel=b==k
        LUT[k,ci]=best_h(er[sel][::3]) if sel.sum()>200 else 0.012
print("\noptimal h per mu-bin (channel u): SATURATING shape means 'give up' on hard elements")
print("  bin :", " ".join("%5d"%k for k in range(0,NB,3)))
print("  h_u :", " ".join("%5.4f"%LUT[k,0] for k in range(0,NB,3)))
def E_of(mask,MU):
    tot=0.0; cnt=0.0
    for ci in range(C):
        m=SCM[mask][...,ci]
        mu=MU[...,ci][m]; er=ERR[mask][...,ci][m]
        h=LUT[np.digitize(mu,EDGES[ci]),ci]
        ok=er<=h
        tot+=float((np.exp(-2*h/SIG)*ok).sum()); cnt+=float(m.sum())
    return tot/cnt
Etr=E_of(tr,MUtr); Ete=E_of(te,MUte)
RATIO=0.4876/0.5132
print("\n=== RESULTS (model scale) ===")
print("  constant baseline   E 0.5132   (real 0.4876, sps 33.08, final 78.07)")
print("  v2 surrogate head   E 0.5586")
print("  v3 regression+k*exp E 0.5401")
print("  v4 regression+LUT   E_train %.4f  E_test %.4f"%(Etr,Ete))
sps=100*W*Ete*RATIO
print("\n  v4 real estimate: E ~%.4f -> sps ~%.2f -> DELTA %+.2f -> final ~%.2f"%(
    Ete*RATIO,sps,0.217*(sps-33.078742),78.068714+0.217*(sps-33.078742)))
print("  top-20 real E 0.5896 ; doomduke2 0.6073")
np.savez(f"{B}/local_harness/unc4_lut.npz",LUT=LUT,e0=EDGES[0],e1=EDGES[1])
print("saved LUT -> unc4_lut.npz")
