"""SCORE ASSURANCE for submission_ROBUST.zip.

Scores the LITERAL zip output on held-out competition trajectories, under the
two-anchor-corrected error model, in BOTH split directions.
Hard self-check: the proven zip (original model + [0.030,0.010]) must reproduce
its known real score of 77.20. If it does not, the estimate is void.
"""
import json,os,sys,zipfile,shutil,importlib.util
import numpy as np
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
W=np.stack([np.concatenate([np.asarray(X[s:s+40]),np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=W[:,:IN],W[:,IN:]; SCM=(Y[...,:C]!=0.0)
def run_zip(name,tag):
    WD=f"{B}/_tmp/{tag}"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
    with zipfile.ZipFile(f"{B}/submissions/{name}") as z: z.extractall(WD)
    sys.path.insert(0,WD)
    sp=importlib.util.spec_from_file_location("z_"+tag,os.path.join(WD,"submission.py"))
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    P=[];LO=[];UP=[]
    for i in range(0,len(Xin),64):
        r=m.predict(Xin[i:i+64],metadata={})
        P.append(r["prediction"]);LO.append(r["lower"]);UP.append(r["upper"])
    return (np.concatenate(P,0).astype(np.float32),np.concatenate(LO,0).astype(np.float32),
            np.concatenate(UP,0).astype(np.float32))
Pp,LOp,UPp=run_zip("submission_fno_plain_sps.zip","zprov")
Pr,LOr,UPr=run_zip("submission_ROBUST.zip","zrob")
print("model outputs identical: %s"%np.array_equal(Pp,Pr),flush=True)
n_=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
Wf=lambda rl,tk,mv:0.5*(1-n_(inv(rl)))+0.3*(1-n_(inv(tk)))+0.2*(1-n_(inv(mv)))
fin=lambda rl,tk,mv,sps:0.306*rl+0.163*tk+0.218*mv+0.100*TM+0.217*sps
BASE=fin(RL,TK,MV,29.84)
def acc(P,m):
    dm=S.rel_l2_per_sample(P[m],Y[m],C);tk=S.tke_rel_l2_per_sample(P[m],Y[m],C);mv=S.mvpe_rel_l2_per_sample(P[m],Y[m])
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
res={}
for lab,fitset in (("split A",vidx[0::2]),("split B",vidx[1::2])):
    fit=np.isin(wt,list(fitset)); ev=~fit
    EL=np.abs(Pp[...,:C]-Y[...,:C]); AP=np.abs(Pp[...,:C])
    m0,m1=SCM[fit][...,0],SCM[fit][...,1]
    eu=EL[fit][...,0][m0]; evv=EL[fit][...,1][m1]; au=AP[fit][...,0][m0]; av=AP[fit][...,1][m1]
    NT=eu.size+evv.size; best=None
    for al in np.linspace(1.0,6.0,51):
        for be in np.linspace(-12,2,57):
            su,sv=eu*(al+be*au),evv*(al+be*av)
            Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
            hu,hv=0.05*au,0.05*av
            Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
            r=abs(Ec-0.4399)+abs(Ep-0.2076)
            if best is None or r<best[0]: best=(r,al,be)
    _,AL,BE=best
    A0=acc(Pp,ev)
    def est(P,LO,UP,tag):
        SC=AL+BE*np.abs(P[...,:C])
        ERR=np.abs(P[...,:C]-Y[...,:C])*SC
        h=((UP-LO)/2.0)[...,:C][ev]
        ok=(ERR[ev]<=h)&SCM[ev]
        E=float((np.exp(-2*h/SIG)*ok).sum()/SCM[ev].sum()); cov=float(ok.sum()/SCM[ev].sum())
        a=acc(P,ev); d=[a[i]-A0[i] for i in range(3)]
        rl,tk,mv=RL+d[0],TK+d[1],MV+d[2]
        sps=100*Wf(rl,tk,mv)*E
        return rl,tk,mv,sps,cov,77.20+fin(rl,tk,mv,sps)-BASE
    p=est(Pp,LOp,UPp,"proven"); r=est(Pr,LOr,UPr,"robust")
    res[lab]=(p,r)
    print("\n=== %s (alpha=%.2f beta=%.2f) ==="%(lab,AL,BE))
    print("  SELF-CHECK proven zip -> %.2f  (known 77.20, delta %+.2f)"%(p[5],p[5]-77.20))
    assert abs(p[5]-77.20)<0.45,"SELF-CHECK FAILED"
    print("  ROBUST: rel_l2 %.2f tke %.2f mvpe %.2f sps %.2f cov %.3f -> EST %.2f"%r)
vals=[res[k][1][5] for k in res]
print("\n"+"="*62)
print("ROBUST estimate: %.2f and %.2f  -> mean %.2f, spread %.2f"%(vals[0],vals[1],np.mean(vals),abs(vals[0]-vals[1])))
print("banked 77.20 | agent33 (identical model, bounds only, REAL) 78.64")
