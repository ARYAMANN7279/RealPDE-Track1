"""FAST score assurance for submission_ROBUST.zip (= proven model + [0.0129,0.0098]).
Calibration fit on a 500k-element subsample; scoring on the full eval half.
Hard self-check: proven bounds must reproduce the known 77.20."""
import json,os,sys
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); sims=sorted(set(sim.tolist()))
SCM=(Y[...,:C]!=0.0); EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
n_=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
WR=0.5*(1-n_(inv(RL)))+0.3*(1-n_(inv(TK)))+0.2*(1-n_(inv(MV)))
fin=lambda sps:0.306*RL+0.163*TK+0.218*MV+0.100*TM+0.217*sps
BASE=fin(29.84)
res=[]
for lab,fs in (("split A",sims[0::2]),("split B",sims[1::2])):
    fit=np.isin(sim,fs); ev=~fit
    m0,m1=SCM[fit][...,0],SCM[fit][...,1]
    eu=EL[fit][...,0][m0][::120]; evv=EL[fit][...,1][m1][::120]
    au=AP[fit][...,0][m0][::120]; av=AP[fit][...,1][m1][::120]
    NT=eu.size+evv.size
    best=None
    for al in np.linspace(1.5,5.5,21):
        for be in np.linspace(-11,0,23):
            su,sv=eu*(al+be*au),evv*(al+be*av)
            Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
            hu,hv=0.05*au,0.05*av
            Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
            r=abs(Ec-0.4399)+abs(Ep-0.2076)
            if best is None or r<best[0]: best=(r,al,be)
    rr,AL,BE=best
    ERR=EL[ev]*(AL+BE*AP[ev]); M=SCM[ev]
    def sc(hu,hv):
        h=np.zeros((1,1,1,1,C),np.float32); h[...,0]=hu; h[...,1]=hv
        hh=np.broadcast_to(h,ERR.shape); ok=(ERR<=hh)&M
        E=float((np.exp(-2*hh/SIG)*ok).sum()/M.sum()); cov=float(ok.sum()/M.sum())
        return E,cov,100*WR*E,77.20+fin(100*WR*E)-BASE
    chk=sc(0.030,0.010); rob=sc(0.0129,0.0098)
    print("=== %s  alpha %.2f beta %.2f  anchor-resid %.4f ==="%(lab,AL,BE,rr),flush=True)
    print("   SELF-CHECK proven [0.030,0.010] -> %.2f   (known 77.20, delta %+.2f)"%(chk[3],chk[3]-77.20),flush=True)
    ok = abs(chk[3]-77.20)<0.5
    print("   %s"%("self-check PASSED" if ok else "*** SELF-CHECK FAILED - estimate void ***"),flush=True)
    print("   ROBUST [0.0129,0.0098] -> EST %.2f   sps %.2f  coverage %.3f"%(rob[3],rob[2],rob[1]),flush=True)
    if ok: res.append(rob[3])
if res:
    print("\n"+"="*56)
    print("ROBUST estimate: %s  -> mean %.2f, spread %.2f"%(" / ".join("%.2f"%v for v in res),np.mean(res),max(res)-min(res)))
    print("banked 77.20   gain %+.2f"%(np.mean(res)-77.20))
    print("agent33 (identical model, bounds only, REAL): 78.64")
