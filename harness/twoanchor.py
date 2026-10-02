"""THE ALGORITHM: calibrate the error model against BOTH real anchors, then
optimise bounds under the corrected model.

One scale factor cannot fit both real anchors:
    constant [0.030,0.010] -> E 0.4399   (we match this exactly)
    proportional 0.05|pred| -> E 0.2076  (we predict 0.1565 -> too pessimistic)
The proportional bound is wide where |pred| is large, so under-predicting its E
means our error model is too pessimistic in high-|pred| regions. Fit a 2-param
correction  err_real ~ err_local * (alpha + beta*|pred|)  to BOTH anchors, then
re-optimise the bound family (which now includes a |pred| term).
"""
import json, os, sys
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda:2"; C=2; IN=20
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
mi,si,mt,st=[torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
Wall=np.stack([np.concatenate([np.asarray(X[s:s+40]),
        np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
mdl,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
ORIG=torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV); ORIG=ORIG.get("model_state_dict",ORIG)
mdl.load_state_dict(ORIG); mdl.to(DEV).eval(); o=[]
with torch.no_grad():
    for i in range(0,len(Xin),32):
        o.append((mdl((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
P=np.concatenate(o,0).astype(np.float32)
E_LOC=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
m0,m1=SCM[...,0],SCM[...,1]
eu,ev=E_LOC[...,0][m0],E_LOC[...,1][m1]; au,av=AP[...,0][m0],AP[...,1][m1]
NT=eu.size+ev.size
def E_two(alpha,beta):
    """E under err_real = err_local*(alpha+beta*|pred|), for both anchors."""
    su=eu*(alpha+beta*au); sv=ev*(alpha+beta*av)
    Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
    hu,hv=0.05*au,0.05*av
    Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
    return Ec,Ep
print("fitting alpha,beta to BOTH real anchors (0.4399 const / 0.2076 prop)...",flush=True)
best=None
for alpha in np.linspace(0.5,6.0,34):
    for beta in np.linspace(-12,4,33):
        Ec,Ep=E_two(alpha,beta)
        r=abs(Ec-0.4399)+abs(Ep-0.2076)
        if best is None or r<best[0]: best=(r,alpha,beta,Ec,Ep)
r,alpha,beta,Ec,Ep=best
print("  best alpha=%.3f beta=%.3f  -> E_const %.4f (0.4399)  E_prop %.4f (0.2076)  resid %.4f"%(
    alpha,beta,Ec,Ep,r))
one=None
for a_ in np.linspace(0.5,6.0,200):
    Ec_,Ep_=E_two(a_,0.0)
    rr=abs(Ec_-0.4399)+abs(Ep_-0.2076)
    if one is None or rr<one[0]: one=(rr,a_)
print("  best SINGLE-scale (beta=0) residual %.4f  -> two-param fit is %.1fx better"%(one[0],one[0]/max(r,1e-9)))

SC_u=alpha+beta*au; SC_v=alpha+beta*av
su,sv=eu*SC_u,ev*SC_v
n=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
WR=0.5*(1-n(inv(RL)))+0.3*(1-n(inv(TK)))+0.2*(1-n(inv(MV)))
fin=lambda sps:0.306*RL+0.163*TK+0.218*MV+0.100*TM+0.217*sps
BASE=fin(29.84)
def rep(tag,hu,hv):
    E=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
    cov=((su<=hu).sum()+(sv<=hv).sum())/NT
    sps=100*WR*E
    print("  %-34s E %.4f cov %.3f sps %6.2f -> EST REAL %.2f"%(tag,E,cov,sps,77.20+fin(sps)-BASE),flush=True)
    return E,sps
print("\n=== bound families re-optimised under the CORRECTED error model ===")
rep("current [0.030,0.010]",0.030,0.010)
gu=np.linspace(0.004,0.05,40); gv=np.linspace(0.002,0.03,40)
bu=max(gu,key=lambda h:float((np.exp(-2*h/SIG)*(su<=h)).mean()))
bv=max(gv,key=lambda h:float((np.exp(-2*h/SIG)*(sv<=h)).mean()))
rep("best constant [%.4f,%.4f]"%(bu,bv),bu,bv)
best_ab=None
for a_ in np.linspace(0.0,0.035,15):
    for b_ in np.linspace(0.0,0.30,16):
        hu=a_+b_*au; hv=a_*0.5+b_*av
        E=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
        if best_ab is None or E>best_ab[0]: best_ab=(E,a_,b_)
E_,a_,b_=best_ab
rep("h = %.4f + %.3f*|pred|"%(a_,b_),a_+b_*au,a_*0.5+b_*av)
print("\n  agent33 (identical model, real): E 0.5208  sps 35.33  final 78.64")
np.save(f"{B}/local_harness/twoanchor_ab.npy",np.array([alpha,beta,a_,b_]))
