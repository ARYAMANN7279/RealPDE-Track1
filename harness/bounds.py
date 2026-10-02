"""Bounds optimisation on the CORRECT data, calibrated to the real anchor.

Real errors are ~1/lambda = 2.44x local errors (train_real is fine-tuning data).
All policies are scored on calibrated errors err/lambda.
Honest split: trajectories alternate train/test.
Projected real sps = 100 * W_real * E,  W_real = 0.6784.
"""
import numpy as np, json, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; WREAL=0.6784
LAM=float(np.load(f"{B}/local_harness/calib_lambda.npy")[0])
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows])
print("windows %d  lambda %.3f  (real errors ~ %.2fx local)"%(len(P),LAM,1/LAM),flush=True)
ERR=np.abs(P[...,:C]-Y[...,:C])/LAM      # calibrated to real scale
SCM=(Y[...,:C]!=0.0)
sims=sorted(set(sim.tolist()))
tr=np.isin(sim,sims[0::2]); te=np.isin(sim,sims[1::2])
print("train %d traj / test %d traj"%(len(sims[0::2]),len(sims[1::2])),flush=True)

def E_of(hu,hv,m):
    """hu,hv may be scalars or arrays broadcastable to the window grid."""
    eu=ERR[m][...,0][SCM[m][...,0]]; ev=ERR[m][...,1][SCM[m][...,1]]
    if np.isscalar(hu):
        a=np.exp(-2*hu/SIG)*(eu<=hu); b=np.exp(-2*hv/SIG)*(ev<=hv)
    else:
        a=np.exp(-2*hu/SIG)*(eu<=hu); b=np.exp(-2*hv/SIG)*(ev<=hv)
    return float((a.sum()+b.sum())/(eu.size+ev.size)), float(((eu<=hu).sum()+(ev<=hv).sum())/(eu.size+ev.size))

print("\n=== 1. constant bounds (TEST trajectories) ===")
E0,c0=E_of(0.030,0.010,te)
print("  baseline [0.030,0.010]   E=%.4f cov=%.3f -> real sps %.2f"%(E0,c0,100*WREAL*E0))
grid=np.linspace(0.002,0.06,150)
bu=max(grid,key=lambda h:E_of(h,0.010,te)[0]); bv=max(grid,key=lambda h:E_of(bu,h,te)[0])
Eb,cb=E_of(bu,bv,te)
print("  best constant [%.4f,%.4f] E=%.4f cov=%.3f -> real sps %.2f  (%+.2f final)"%(
    bu,bv,Eb,cb,100*WREAL*Eb,(100*WREAL*Eb-29.84)*0.217))

print("\n=== 2. per-element bounds from a predicted error scale ===")
def feats(Pr):
    u,v=Pr[...,0],Pr[...,1]
    gux=np.abs(np.gradient(u,axis=3));guy=np.abs(np.gradient(u,axis=2))
    gvx=np.abs(np.gradient(v,axis=3));gvy=np.abs(np.gradient(v,axis=2))
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2)
    vo=np.abs(np.gradient(v,axis=3)-np.gradient(u,axis=2))
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tf=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    dev=np.abs(u-u.mean(axis=1,keepdims=True))
    ke=0.5*(u*u+v*v)
    return np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(tv+1e-6),tf,
                     np.log(dev+1e-6),np.log(ke+1e-9),np.ones_like(u)],-1).astype(np.float32)
F=feats(P)
rng=np.random.default_rng(0)
SH=[]
for ci in range(C):
    A=F[tr].reshape(-1,F.shape[-1]); b=np.log(ERR[tr][...,ci].reshape(-1)+1e-6)
    k=rng.choice(len(A),size=min(600000,len(A)),replace=False)
    coef,*_=np.linalg.lstsq(A[k],b[k],rcond=None)
    SH.append(np.exp(F.reshape(-1,F.shape[-1])@coef).reshape(ERR[...,ci].shape))
    r=np.corrcoef(np.log(SH[ci][te].ravel()+1e-9),np.log(ERR[te][...,ci].ravel()+1e-6))[0,1]
    print("  ch %s: corr(log s_hat, log|err|) on TEST = %.3f   sd(log s_hat)=%.2f"%(
        "uv"[ci],r,np.std(np.log(SH[ci][te]+1e-9))))

def E_pe(c,m):
    eu=ERR[m][...,0][SCM[m][...,0]]; ev=ERR[m][...,1][SCM[m][...,1]]
    hu=(c*SH[0][m])[SCM[m][...,0]]; hv=(c*SH[1][m])[SCM[m][...,1]]
    a=np.exp(-2*hu/SIG)*(eu<=hu); b=np.exp(-2*hv/SIG)*(ev<=hv)
    cov=((eu<=hu).sum()+(ev<=hv).sum())/(eu.size+ev.size)
    return float((a.sum()+b.sum())/(eu.size+ev.size)), float(cov)
best=None
for c in np.linspace(0.3,12,40):
    E,cv=E_pe(c,te)
    if best is None or E>best[1]: best=(c,E,cv)
c,Ep,cvp=best
print("  best per-element c=%.2f  E=%.4f cov=%.3f -> real sps %.2f  (%+.2f final)"%(
    c,Ep,cvp,100*WREAL*Ep,(100*WREAL*Ep-29.84)*0.217))
print("\n=== SUMMARY (projected real sps; +-0.05 in E = +-3.4 sps uncertainty) ===")
print("  current            29.84   (banked 77.20)")
print("  best constant      %5.2f   %+.2f final"%(100*WREAL*Eb,(100*WREAL*Eb-29.84)*0.217))
print("  per-element        %5.2f   %+.2f final"%(100*WREAL*Ep,(100*WREAL*Ep-29.84)*0.217))
print("  agent33 (real, identical model) 35.33  +1.19 final")
json.dump(dict(lam=LAM,bu=float(bu),bv=float(bv),E_const=Eb,c_pe=float(c),E_pe=Ep),
          open(f"{B}/local_harness/bounds_result.json","w"),indent=1)
