"""Two untested mechanisms that could close the gap to agent33's E=0.5208.

M1. PER-LOCATION BIAS CORRECTION OF THE PREDICTION.
    If the model is systematically off at fixed locations (wake), subtracting a
    per-(pixel,channel) mean-error map fixes rel_l2, tke, mvpe AND sps at once.
M2. ASYMMETRIC BOUNDS. The scorer penalises (upper-lower) and asks only whether
    the target is inside. Nothing requires the interval to be centred on the
    prediction. If the residual is skewed, an off-centre interval captures more
    probability for the same width.
    (memory says a global bias shift FAILED live: 29.84 -> 23.65 -- but that map
     was computed on the WRONG dataset, so it is worth retesting properly.)
Both fit on half the held-out trajectories, scored on the other half.
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
def run(sd):
    mdl.load_state_dict(sd); mdl.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            o.append((mdl((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
ORIG=torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV); ORIG=ORIG.get("model_state_dict",ORIG)
P=run(ORIG)
e0=np.abs(P[...,:C]-Y[...,:C])
def Ec(e,lam):
    eu=e[...,0][SCM[...,0]]/lam; ev=e[...,1][SCM[...,1]]/lam
    return float((np.exp(-2*0.030/SIG)*(eu<=0.030)).sum()+(np.exp(-2*0.010/SIG)*(ev<=0.010)).sum())/(eu.size+ev.size)
lo,hi=0.02,20.0
for _ in range(60):
    m_=(lo+hi)/2
    if Ec(e0,m_)>0.4399: hi=m_
    else: lo=m_
LAM=(lo+hi)/2
print("calibration OK (E=%.4f)"%Ec(e0,LAM),flush=True)
h=len(wins)//2; fit=np.zeros(len(wins),bool); fit[:h]=True; ev=~fit
n=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
Wf=lambda rl,tk,mv:0.5*(1-n(inv(rl)))+0.3*(1-n(inv(tk)))+0.2*(1-n(inv(mv)))
fin=lambda rl,tk,mv,sps:0.306*rl+0.163*tk+0.218*mv+0.100*TM+0.217*sps
BASE=fin(RL,TK,MV,29.84)
def acc(Q,m):
    dm=S.rel_l2_per_sample(Q[m],Y[m],C);tk=S.tke_rel_l2_per_sample(Q[m],Y[m],C);mv=S.mvpe_rel_l2_per_sample(Q[m],Y[m])
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
a0=acc(P,ev)
print("\nbaseline (original, held-out half): rel_l2 %.2f tke %.2f mvpe %.2f"%a0,flush=True)

# ---------- M1: per-location bias correction of the PREDICTION ----------
resid=(P[...,:C]-Y[...,:C])
biasmap=np.zeros((32,64,C),np.float32)
for ci in range(C):
    r=resid[fit][...,ci]; M=SCM[fit][...,ci]
    s=(r*M).sum(axis=(0,1)); c=M.sum(axis=(0,1)).clip(1)
    biasmap[:,:,ci]=s/c
print("\n=== M1: per-location bias correction ===")
print("  bias map: u mean %.5f max|.| %.5f | v mean %.5f max|.| %.5f"%(
    biasmap[...,0].mean(),np.abs(biasmap[...,0]).max(),biasmap[...,1].mean(),np.abs(biasmap[...,1]).max()))
for strength in [0.0,0.5,1.0]:
    Q=P.copy(); Q[...,:C]=P[...,:C]-strength*biasmap[None,None]
    a=acc(Q,ev); d=[a[i]-a0[i] for i in range(3)]
    ERR=np.abs(Q[...,:C]-Y[...,:C])/LAM
    HL=np.zeros((32,64,C),np.float32)
    for ci in range(C):
        E=ERR[fit][...,ci]; M=SCM[fit][...,ci]
        for i in range(32):
            for j in range(64): HL[i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]])
    hh=np.broadcast_to(HL[None,None],ERR[ev].shape); ok=(ERR[ev]<=hh)&SCM[ev]
    E=float((np.exp(-2*hh/SIG)*ok).sum()/SCM[ev].sum())
    rl,tk,mv=RL+d[0],TK+d[1],MV+d[2]; sps=100*Wf(rl,tk,mv)*E
    print("  strength %.1f: rel_l2 %+.2f tke %+.2f mvpe %+.2f | E %.4f sps %.2f -> EST REAL %.2f"%(
        strength,d[0],d[1],d[2],E,sps,77.20+fin(rl,tk,mv,sps)-BASE),flush=True)

# ---------- M2: asymmetric per-location bounds ----------
print("\n=== M2: asymmetric per-location bounds (lower/upper fitted separately) ===")
ERR=np.abs(P[...,:C]-Y[...,:C])/LAM
R=(Y[...,:C]-P[...,:C])/LAM          # signed residual, calibrated
LOW=np.zeros((32,64,C),np.float32); UPP=np.zeros((32,64,C),np.float32)
for ci in range(C):
    r=R[fit][...,ci]; M=SCM[fit][...,ci]
    for i in range(32):
        for j in range(64):
            v=r[:,:,i,j][M[:,:,i,j]]
            if v.size<20: LOW[i,j,ci]=UPP[i,j,ci]=0.01; continue
            qs=np.quantile(v,np.linspace(0.005,0.995,60))
            best=(-1,0.01,0.01)
            for lo_ in qs[:30]:
                for up_ in qs[30:]:
                    if up_<=lo_: continue
                    val=np.exp(-(up_-lo_)/SIG)*np.mean((v>=lo_)&(v<=up_))
                    if val>best[0]: best=(val,lo_,up_)
            LOW[i,j,ci]=best[1]; UPP[i,j,ci]=best[2]
r=R[ev]; M=SCM[ev]
lo_b=np.broadcast_to(LOW[None,None],r.shape); up_b=np.broadcast_to(UPP[None,None],r.shape)
ok=(r>=lo_b)&(r<=up_b)&M
Easym=float((np.exp(-(up_b-lo_b)/SIG)*ok).sum()/M.sum())
sps=100*Wf(RL,TK,MV)*Easym
print("  E %.4f  coverage %.3f  sps %.2f -> EST REAL %.2f"%(
    Easym,float(ok.sum()/M.sum()),sps,77.20+fin(RL,TK,MV,sps)-BASE))
print("  (symmetric per-location on same split gave E~0.4806, sps 32.60)")
print("\n  agent33 target: E=0.5208, sps 35.33")
np.savez(f"{B}/local_harness/mech.npz",bias=biasmap,low=LOW,upp=UPP)
