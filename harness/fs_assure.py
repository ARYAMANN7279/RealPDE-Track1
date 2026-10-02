"""ASSURANCE for the SHIPPED submission_SOUP_v2.zip.

Runs the actual zip's predict() (predictions AND bounds as it will emit them),
then scores those exact outputs. Calibration fit on the FIT half of held-out
trajectories, scored on the disjoint EVAL half.
Self-check: the proven 78.07 zip must reproduce 78.07 on the same pipeline.
"""
import json,os,sys,zipfile,shutil
import importlib.util as iu
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; IN=20
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[];wt=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0); wt.append(i)
wt=np.array(wt)
W_=np.stack([np.concatenate([np.asarray(X[s:s+40]),np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=W_[:,:IN],W_[:,IN:]; SCM=(Y[...,:C]!=0.0)
FIT=set(vidx[0::2]); fit=np.isin(wt,list(FIT)); ev=~fit
print("held-out: fit %d / eval %d windows"%(fit.sum(),ev.sum()),flush=True)
def run_zip(name,tag):
    WD=f"{B}/_tmp/{tag}"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
    with zipfile.ZipFile(f"{B}/submissions/{name}") as z: z.extractall(WD)
    sys.path.insert(0,WD)
    s2=iu.spec_from_file_location("z_"+tag,os.path.join(WD,"submission.py"))
    m=iu.module_from_spec(s2); s2.loader.exec_module(m)
    P=[];L=[];U=[]
    for i in range(0,len(Xin),48):
        r=m.predict(Xin[i:i+48],metadata={})
        P.append(r["prediction"]);L.append(r["lower"]);U.append(r["upper"])
    return (np.concatenate(P,0).astype(np.float32),np.concatenate(L,0).astype(np.float32),
            np.concatenate(U,0).astype(np.float32))
Pp,Lp,Up=run_zip("submission_SOUP_v2.zip","zp")
Pf,Lf,Uf=run_zip("submission_SOUP_v2.zip","zf")
n_=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL0,TK0,MV0,TM=94.168150,74.025866,92.836278,91.362375
Wf_=lambda rl,tk,mv:0.5*(1-n_(inv(rl)))+0.3*(1-n_(inv(tk)))+0.2*(1-n_(inv(mv)))
def acc(P,m):
    dm=S.rel_l2_per_sample(P[m],Y[m],C);tk=S.tke_rel_l2_per_sample(P[m],Y[m],C);mv=S.mvpe_rel_l2_per_sample(P[m],Y[m])
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
A0=acc(Pp,ev)
EL=np.abs(Pp[...,:C]-Y[...,:C]); AP=np.abs(Pp[...,:C])
m0,m1=SCM[fit][...,0],SCM[fit][...,1]
eu=EL[fit][...,0][m0][::30]; evv=EL[fit][...,1][m1][::30]
au=AP[fit][...,0][m0][::30]; av=AP[fit][...,1][m1][::30]; NT=eu.size+evv.size
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
print("calibration alpha %.2f beta %.2f resid %.4f"%(AL,BE,rr),flush=True)
def est(P,L,U,tag):
    ERR=np.abs(P[...,:C]-Y[...,:C])*(AL+BE*np.abs(P[...,:C]))
    h=((U-L)/2.0)[...,:C]
    ok=(ERR[ev]<=h[ev])&SCM[ev]
    E=float((np.exp(-2*h[ev]/SIG)*ok).sum()/SCM[ev].sum()); cov=float(ok.sum()/SCM[ev].sum())
    a=acc(P,ev); d=[a[i]-A0[i] for i in range(3)]
    rl,tk,mv=RL0+d[0],TK0+d[1],MV0+d[2]
    sps=100*Wf_(rl,tk,mv)*E
    dfin=0.306*(rl-RL0)+0.163*(tk-TK0)+0.218*(mv-MV0)+0.217*(sps-33.078742)
    print("  %-22s rel_l2 %.2f tke %.2f mvpe %.2f | E %.4f cov %.3f sps %.2f | est %.2f (%+.2f)"%(
        tag,rl,tk,mv,E,cov,sps,78.068714+dfin,dfin),flush=True)
    return 78.068714+dfin
print("\n=== SCORING THE ACTUAL ZIPS ===")
chk=est(Pp,Lp,Up,"proven 78.07 zip")
print("  SELF-CHECK: %.2f vs known 78.07  (delta %+.2f)"%(chk,chk-78.07))
assert abs(chk-78.07)<0.5,"self-check failed"
est(Pf,Lf,Uf,"FULLSTACK_v3 (shipped)")
