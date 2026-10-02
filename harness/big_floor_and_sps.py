import importlib.util, os, sys, json, numpy as np, torch
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
EVAL=f"{H}/real_eval_heldout_big"
sys.path.insert(0,KIT);sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
SIGMA=scoring.SIGMA_GLOBAL
mi=torch.tensor([0.154960856,-0.000513992854,0.0]);si=torch.tensor([0.0968056545,0.015960684,1.0])
mt=torch.tensor([0.154962569,-0.000517793698,0.0]);st=torch.tensor([0.0968104079,0.0159636438,1.0])
X=np.load(f"{EVAL}/inputs.npz")["input"];Y=np.load(f"{EVAL}/targets.npz")["target"];meta=json.load(open(f"{EVAL}/meta.json"))
sims=np.array([m["sim_id"] for m in meta])
CACHE=f"{EVAL}/fno_tta_cache.npz"
if not os.path.exists(CACHE):
    m=load_baseline(f"{H}/fno_model/sim_real_fno_fp16.pth",device="cpu")[0]
    P1,P2=[],[]
    with torch.no_grad():
        for i in range(0,X.shape[0],16):
            xb=torch.from_numpy(X[i:i+16]).float()
            y1=(m((xb-mi)/si)*st+mt).numpy()
            x2=torch.flip(xb,dims=[2]).clone();x2[...,1]=-x2[...,1]
            y2=m((x2-mi)/si)*st+mt;y2=torch.flip(y2,dims=[2]).clone();y2[...,1]=-y2[...,1]
            P1.append(y1);P2.append(y2.numpy())
    P1=np.concatenate(P1,0).astype(np.float32);P2=np.concatenate(P2,0).astype(np.float32)
    np.savez(CACHE,P1=P1,P2=P2)
    print("cached FNO TTA")
d=np.load(CACHE);P1,P2=d["P1"],d["P2"];pred=(0.5*(P1+P2)).astype(np.float32)
C=2
# floor with fixed shipped bounds
fh=np.zeros((1,1,1,1,3),np.float32);fh[...,0]=0.107537;fh[...,1]=0.010307
rel=float(np.mean(scoring.rel_l2_per_sample(pred,Y,C)));tke=float(np.mean(scoring.tke_rel_l2_per_sample(pred,Y,C)));mvpe=scoring.mvpe_rel_l2(pred,Y)
sps,cov=scoring.aggregate_sps(pred,Y,C,lower=pred-fh,upper=pred+fh)
print(f"\n=== FNO floor on {X.shape[0]} big held-out windows ===")
print(f"L2={scoring.score_error(rel):.2f} TKE={scoring.score_error(tke):.2f} MVPE={scoring.score_error(mvpe):.2f} SPS={scoring.score_sps(sps):.2f} cov={cov*100:.1f}%")
print(f"raw: rel={rel:.4f} tke={tke:.4f} mvpe={mvpe:.4f}")
# TKE per trajectory (does the steady-3750 skew wash out?)
print("\nper-trajectory TKE score:")
for s in sorted(set(sims.tolist())):
    mk=sims==s
    tk=float(np.mean(scoring.tke_rel_l2_per_sample(pred[mk],Y[mk],C)))
    print(f"   {s:16s} n={mk.sum():3d}  TKE_raw={tk:8.3f}  score={scoring.score_error(tk):.2f}")
# SPS optimization (separable), honest tune/test split
Ws=(0.5*(1-np.clip(scoring.rel_l2_per_sample(pred,Y,C),0,None)/(0.5+scoring.rel_l2_per_sample(pred,Y,C)))
    +0.3*(1-np.nan_to_num(scoring.tke_rel_l2_per_sample(pred,Y,C)/(0.5+scoring.tke_rel_l2_per_sample(pred,Y,C))))
    +0.2*(1-np.nan_to_num(scoring.mvpe_rel_l2_per_sample(pred,Y)/(0.5+scoring.mvpe_rel_l2_per_sample(pred,Y))))).astype(np.float32)
abserr=np.abs(Y[...,:C]-pred[...,:C]).astype(np.float32);scored=(Y[...,:C]!=0.0)
Wpix=np.broadcast_to(Ws.reshape(-1,1,1,1,1),abserr.shape)
uniq=sorted(set(sims.tolist()));tune=np.isin(sims,uniq[::2]);test=~tune
def sps_const(hu,hv,mask):
    ns=np.count_nonzero(scored&mask.reshape(-1,1,1,1,1));tot=0.0;cv=0
    for ch,h in ((0,hu),(1,hv)):
        ins=(abserr[...,ch]<=h)&scored[...,ch]&mask.reshape(-1,1,1,1)
        tot+=np.exp(-2*h/SIGMA)*np.sum(Wpix[...,ch]*ins,dtype=np.float64);cv+=np.count_nonzero(ins)
    return 100*min(max(tot/ns if ns else 0,0),1),100*cv/ns if ns else 0
def best_ch(ch,mask):
    m=mask.reshape(-1,1,1,1);e=abserr[...,ch][scored[...,ch]&m];w=Wpix[...,ch][scored[...,ch]&m]
    o=np.argsort(e);e=e[o];G=np.cumsum(w[o]);f=np.exp(-2*e/SIGMA)*G;return float(e[np.argmax(f)])
print(f"\n=== SPS bound opt (tune={sorted(uniq[::2])} test={sorted(uniq[1::2])}) ===")
st,ct=sps_const(0.107537,0.010307,test);print(f"fixed(shipped)  test SPS={st:.2f} cov={ct:.1f}%")
hu=best_ch(0,tune);hv=best_ch(1,tune);st,ct=sps_const(hu,hv,test)
print(f"const-opt       test SPS={st:.2f} cov={ct:.1f}%  (hu={hu:.3f} hv={hv:.3f})")
