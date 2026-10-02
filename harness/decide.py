"""ONE table: every model x every bounds policy, ranked by estimated REAL score.

Method (all deltas, anchored to the known-true 77.20 submission):
  accuracy : measured directly on 17 held-out trajectories -> DELTA vs original
             -> applied to the real 94.17 / 74.03 / 92.84
  sps      : E measured on errors calibrated so the original+current-bounds config
             reproduces its real sps of 29.84 exactly -> sps = 100*W_real_new*E
  final    : the externally-fitted formula
Also: search what bound policy would be needed to hit agent33's E (the 35.33 team).
"""
import json, os, sys, itertools
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
MODELS={"original":ORIG,
        "soup(all6)":torch.load(f"{B}/local_harness/soup_best.pth",map_location=DEV),
        "single w015":torch.load(f"{B}/local_harness/ft_w015_best.pth",map_location=DEV)}
PR={k:run(v) for k,v in MODELS.items()}
def acc(P):
    dm=S.rel_l2_per_sample(P,Y,C); tk=S.tke_rel_l2_per_sample(P,Y,C); mv=S.mvpe_rel_l2_per_sample(P,Y)
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
ACC={k:acc(v) for k,v in PR.items()}
# calibrate lambda on ORIGINAL + current bounds so it reproduces real sps 29.84
e0=np.abs(PR["original"][...,:C]-Y[...,:C])
def Ec(e,lam,hu=0.030,hv=0.010):
    eu=e[...,0][SCM[...,0]]/lam; ev=e[...,1][SCM[...,1]]/lam
    return float((np.exp(-2*hu/SIG)*(eu<=hu)).sum()+(np.exp(-2*hv/SIG)*(ev<=hv)).sum())/(eu.size+ev.size)
lo,hi=0.02,20.0
for _ in range(60):
    m_=(lo+hi)/2
    if Ec(e0,m_)>0.4399: hi=m_
    else: lo=m_
LAM=(lo+hi)/2
assert abs(Ec(e0,LAM)-0.4399)<0.005
print("calibration OK: original+current bounds -> E %.4f (real 0.4399, sps 29.84)\n"%Ec(e0,LAM),flush=True)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
n=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
def W(rl,tk,mv): return 0.5*(1-n(inv(rl)))+0.3*(1-n(inv(tk)))+0.2*(1-n(inv(mv)))
def fin(rl,tk,mv,sps): return 0.306*rl+0.163*tk+0.218*mv+0.100*TM+0.217*sps
BASE=fin(RL,TK,MV,29.84)
half=len(wins)//2; fitm=np.zeros(len(wins),bool); fitm[:half]=True   # fit bounds on half, score on other half
out=[]
for mk,P in PR.items():
    ERR=np.abs(P[...,:C]-Y[...,:C])/LAM
    a=ACC[mk]; d=[a[i]-ACC["original"][i] for i in range(3)]
    rl,tk,mv=RL+d[0],TK+d[1],MV+d[2]; Wn=W(rl,tk,mv)
    HL=np.zeros((32,64,C),np.float32)
    for ci in range(C):
        E=ERR[fitm][...,ci]; M=SCM[fitm][...,ci]
        for i in range(32):
            for j in range(64): HL[i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]])
    hk=np.zeros((1,1,C),np.float32)
    for ci in range(C): hk[0,0,ci]=best_h(ERR[fitm][...,ci][SCM[fitm][...,ci]][::17])
    ev=~fitm
    def sc(H):
        h=np.broadcast_to(H[None,None],ERR[ev].shape); ok=(ERR[ev]<=h)&SCM[ev]
        return float((np.exp(-2*h/SIG)*ok).sum()/SCM[ev].sum()), float(ok.sum()/SCM[ev].sum())
    pol={"current [.030,.010]":np.array([[[0.030,0.010]]],np.float32)*np.ones((32,64,1),np.float32),
         "best constant":hk*np.ones((32,64,1),np.float32)}
    for mm in [1.00,1.15,1.30]:
        pol["per-location x%.2f"%mm]=HL*mm
    for pk,H in pol.items():
        E,cv=sc(H); sps=100*Wn*E
        out.append((mk,pk,rl,tk,mv,sps,cv,fin(rl,tk,mv,sps)-BASE))
out.sort(key=lambda r:-r[7])
print("%-13s %-21s %7s %7s %7s %7s %6s %9s"%("model","bounds","rel_l2","tke","mvpe","sps","cov","EST REAL"))
print("-"*92)
for mk,pk,rl,tk,mv,sps,cv,d in out:
    print("%-13s %-21s %7.2f %7.2f %7.2f %7.2f %6.3f %9.2f"%(mk,pk,rl,tk,mv,sps,cv,77.20+d))
print("-"*92)
print("known-true baseline: original + current bounds = 77.20 (sps 29.84)")
print("demonstrated by others on the IDENTICAL model: agent33 sps 35.33 -> 78.64")
# what would it take to reach agent33's E?
Eneed=35.33/(100*W(RL,TK,MV))
print("\nagent33 implies E=%.4f on the original model. Our best original-model E above:"%Eneed)
best_orig=max([o for o in out if o[0]=="original"],key=lambda r:r[5])
print("  %s -> sps %.2f (E=%.4f).  Shortfall %.4f in E."%(best_orig[1],best_orig[5],best_orig[5]/(100*W(RL,TK,MV)),Eneed-best_orig[5]/(100*W(RL,TK,MV))))
