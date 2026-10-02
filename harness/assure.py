"""FINAL ASSURANCE: score the LITERAL zip artifact, and re-test every bounds
family under the CORRECTED two-anchor calibration (earlier verdicts on several
of these were made under the old, wrong single-scale calibration).

Split is disjoint BY TRAJECTORY. alpha/beta fitted on the FIT half only, then
applied to the EVAL half -> no calibration leakage into the comparison.
Self-check: original + current bounds must reproduce 77.20.
"""
import json, os, sys, zipfile, importlib.util, shutil
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; IN=20
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[];wt=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0); wt.append(i)
wt=np.array(wt)
Wall=np.stack([np.concatenate([np.asarray(X[s:s+40]),
        np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
FITT=set(vidx[0::2]); fit=np.isin(wt,list(FITT)); ev=~fit
print("trajectory-disjoint split: fit %d / eval %d windows"%(fit.sum(),ev.sum()),flush=True)

# ---- run the ACTUAL zip ----
ZIP=f"{B}/submissions/submission_original_perloc_v2.zip"
WD="/tmp/_assure_zip"
shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(ZIP) as z: z.extractall(WD)
sys.path.insert(0,WD)
sp=importlib.util.spec_from_file_location("subm",os.path.join(WD,"submission.py"))
subm=importlib.util.module_from_spec(sp); sp.loader.exec_module(subm)
PL,LO,UP=[],[],[]
for i in range(0,len(Xin),48):
    r=subm.predict(Xin[i:i+48],metadata={})
    PL.append(r["prediction"]); LO.append(r["lower"]); UP.append(r["upper"])
P=np.concatenate(PL,0).astype(np.float32); LOW=np.concatenate(LO,0).astype(np.float32); UPP=np.concatenate(UP,0).astype(np.float32)
print("zip predict() ran: pred%s"%(str(P.shape)),flush=True)

# ---- corrected calibration, fitted on FIT half ONLY ----
EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
mf0,mf1=SCM[fit][...,0],SCM[fit][...,1]
eu,ev_=EL[fit][...,0][mf0],EL[fit][...,1][mf1]; au,av=AP[fit][...,0][mf0],AP[fit][...,1][mf1]
NT=eu.size+ev_.size
best=None
for al in np.linspace(1.0,6.0,51):
    for be in np.linspace(-12,2,57):
        su,sv=eu*(al+be*au),ev_*(al+be*av)
        Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
        hu,hv=0.05*au,0.05*av
        Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
        r=abs(Ec-0.4399)+abs(Ep-0.2076)
        if best is None or r<best[0]: best=(r,al,be)
r,AL,BE=best
print("two-anchor fit (FIT half only): alpha=%.3f beta=%.3f resid %.4f"%(AL,BE,r),flush=True)
SCALE=AL+BE*AP
ERR=EL*SCALE                      # errors on the real scale
n=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
WR=0.5*(1-n(inv(RL)))+0.3*(1-n(inv(TK)))+0.2*(1-n(inv(MV)))
fin=lambda sps:0.306*RL+0.163*TK+0.218*MV+0.100*TM+0.217*sps
BASE=fin(29.84)
def sc(H):
    """H broadcastable to (N,T,32,64,C) half-widths; scored on EVAL half."""
    h=np.broadcast_to(H,ERR[ev].shape); ok=(ERR[ev]<=h)&SCM[ev]
    E=float((np.exp(-2*h/SIG)*ok).sum()/SCM[ev].sum()); cov=float(ok.sum()/SCM[ev].sum())
    return E,cov,100*WR*E,77.20+fin(100*WR*E)-BASE
# self-check
cur=np.array([[[0.030,0.010]]],np.float32)*np.ones((32,64,1),np.float32)
E,cv,sps,f=sc(cur[None,None])
print("\nSELF-CHECK original+current bounds -> %.2f (must be ~77.20)  delta %+.2f"%(f,f-77.20))
assert abs(f-77.20)<0.4,"self-check failed"

H_ZIP=((UPP-LOW)/2.0)[...,:C]               # exactly what the zip emits
res=[]
res.append(("*** THE ZIP (per-location x1.15) ***",)+sc(H_ZIP[ev]))
# multiplier sweep on the zip's own map
base_map=H_ZIP[0,0]/1.0
for m_ in [0.85,1.00,1.10,1.15,1.25,1.40,1.60]:
    res.append(("per-location x%.2f"%(m_*1.15/1.15),)+sc((base_map*(m_))[None,None]))
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
# alternatives, ALL fitted on FIT half under the corrected calibration
hk=np.zeros((1,1,1,1,C),np.float32)
for ci in range(C): hk[...,ci]=best_h(ERR[fit][...,ci][SCM[fit][...,ci]][::17])
res.append(("best constant [%.4f,%.4f]"%(hk[...,0].item(),hk[...,1].item()),)+sc(hk))
HT=np.zeros((1,20,1,1,C),np.float32)
for t in range(20):
    for ci in range(C):
        HT[0,t,0,0,ci]=best_h(ERR[fit][:,t][...,ci][SCM[fit][:,t][...,ci]][::7])
res.append(("per-timestep",)+sc(HT))
HTL=np.zeros((1,20,32,64,C),np.float32)
for ci in range(C):
    for t in range(20):
        E_=ERR[fit][:,t][...,ci]; M_=SCM[fit][:,t][...,ci]
        for i in range(32):
            for j in range(64): HTL[0,t,i,j,ci]=best_h(E_[:,i,j][M_[:,i,j]])
res.append(("per-timestep x location",)+sc(HTL))
# |pred|-dependent family
bestab=None
for a_ in np.linspace(0.002,0.030,15):
    for b_ in np.linspace(0.0,0.25,11):
        h=a_+b_*AP[fit]; ok=(ERR[fit]<=h)&SCM[fit]
        E=float((np.exp(-2*h/SIG)*ok).sum()/SCM[fit].sum())
        if bestab is None or E>bestab[0]: bestab=(E,a_,b_)
_,a_,b_=bestab
res.append(("h = %.4f + %.3f|pred|"%(a_,b_),)+sc(a_+b_*AP[ev]))
print("\n%-38s %8s %8s %8s %9s"%("policy","E","coverage","sps","EST REAL"))
print("-"*78)
for tag,E,cv,sps,f in sorted(res,key=lambda r:-r[4]):
    print("%-38s %8.4f %8.3f %8.2f %9.2f"%(tag,E,cv,sps,f))
print("-"*78)
print("known-true 77.20 | agent33 (identical model, bounds only) 78.64")
