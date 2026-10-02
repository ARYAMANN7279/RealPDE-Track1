"""SPS bound optimization, exploiting channel-separability + sorted-cumsum.
SPS*n_scored = sum over (pixel,channel) of W[sample]*exp(-2h/sigma)*(|t-p|<=h)*scored.
Separable per channel -> optimize u,v independently. Honest tune/test traj split.
"""
import importlib.util, os, numpy as np
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
SIGMA=scoring.SIGMA_GLOBAL
d=np.load(f"{H}/real_eval_heldout/fno_tta_cache.npz",allow_pickle=True)
P1,P2,Y,sims=d["P1"],d["P2"],d["Y"],d["sims"]
pred=(0.5*(P1+P2)).astype(np.float32); disag=np.abs(P1-P2).astype(np.float32)
C=2
dm=scoring.rel_l2_per_sample(pred,Y,C);tke=scoring.tke_rel_l2_per_sample(pred,Y,C);mvpe=scoring.mvpe_rel_l2_per_sample(pred,Y)
def pm(x):x=x/(0.5+x);return np.where(np.isfinite(x),1.0-x,0.0)
Ws=(0.5*pm(dm)+0.3*pm(tke)+0.2*pm(mvpe)).astype(np.float32)   # (N,)
N=pred.shape[0]
abserr=np.abs(Y[...,:C]-pred[...,:C]).astype(np.float32)      # (N,T,H,W,C)
scored=(Y[...,:C]!=0.0)
dis=disag[...,:C]
Wpix=np.broadcast_to(Ws.reshape(-1,1,1,1,1),abserr.shape)
uniq=sorted(set(sims.tolist()));tune=np.isin(sims,uniq[::2]);test=~tune

def n_scored(mask): return int(np.count_nonzero(scored & mask.reshape(-1,1,1,1,1)))

def sps_const(hu,hv,mask):
    m=mask.reshape(-1,1,1,1,1)
    ns=np.count_nonzero(scored&m)
    tot=0.0
    for ch,h in ((0,hu),(1,hv)):
        ins=(abserr[...,ch]<=h)&scored[...,ch]&mask.reshape(-1,1,1,1)
        tot+=np.exp(-2*h/SIGMA)*np.sum(Wpix[...,ch]*ins,dtype=np.float64)
    w=tot/ns if ns else 0.0
    # coverage across both channels
    cov=0
    for ch,h in ((0,hu),(1,hv)):
        cov+=np.count_nonzero((abserr[...,ch]<=h)&scored[...,ch]&mask.reshape(-1,1,1,1))
    return 100*min(max(w,0),1), 100*cov/ns if ns else 0

def best_const_channel(ch,mask):
    m=mask.reshape(-1,1,1,1)
    e=abserr[...,ch][scored[...,ch]&m]; w=Wpix[...,ch][scored[...,ch]&m]
    order=np.argsort(e); e=e[order]; w=w[order]
    G=np.cumsum(w)                      # G(e_i)=sum of W for |err|<=e_i
    f=np.exp(-2*e/SIGMA)*G              # objective at h=e_i
    i=np.argmax(f); return float(e[i])

def sps_adapt(au,bu,av,bv,mask):
    m=mask.reshape(-1,1,1,1); ns=np.count_nonzero(scored&mask.reshape(-1,1,1,1,1))
    tot=0.0;cov=0
    for ch,(a,b) in ((0,(au,bu)),(1,(av,bv))):
        h=a+b*dis[...,ch]
        ins=(abserr[...,ch]<=h)&scored[...,ch]&m
        tot+=np.sum(Wpix[...,ch]*np.exp(-2*h/SIGMA)*ins,dtype=np.float64)
        cov+=np.count_nonzero(ins)
    w=tot/ns if ns else 0.0
    return 100*min(max(w,0),1),100*cov/ns if ns else 0

# verify fast-const vs official on whole set
fh=np.zeros((1,1,1,1,3),np.float32);fh[...,0]=0.107537;fh[...,1]=0.010307
off=scoring.score_sps(scoring.aggregate_sps(pred,Y,C,lower=pred-fh,upper=pred+fh)[0])
mine=sps_const(0.107537,0.010307,np.ones(N,bool))[0]
print(f"VERIFY: official={off:.2f} mine={mine:.2f} (diff={abs(off-mine):.2f})\n")
print(f"tune={sorted(uniq[::2])}  test={sorted(uniq[1::2])}")
print(f"{'strategy':16s}{'tuneSPS':>9s}{'testSPS':>9s}{'testcov':>9s}")
st,ct=sps_const(0.107537,0.010307,test);print(f"{'fixed(shipped)':16s}{sps_const(0.107537,0.010307,tune)[0]:9.2f}{st:9.2f}{ct:8.1f}%")
hu=best_const_channel(0,tune);hv=best_const_channel(1,tune)
st,ct=sps_const(hu,hv,test);print(f"{'const-opt':16s}{sps_const(hu,hv,tune)[0]:9.2f}{st:9.2f}{ct:8.1f}%  (hu={hu:.3f} hv={hv:.3f})")
best=(-1,)
for au in np.linspace(0,0.1,6):
    for bu in [0,1,2,3,5]:
        for av in np.linspace(0,0.02,6):
            for bv in [0,1,2,3,5]:
                s,_=sps_adapt(au,bu,av,bv,tune)
                if s>best[0]:best=(s,au,bu,av,bv)
_,au,bu,av,bv=best;st,ct=sps_adapt(au,bu,av,bv,test)
print(f"{'adaptive-TTA':16s}{best[0]:9.2f}{st:9.2f}{ct:8.1f}%  (au={au:.3f} bu={bu} av={av:.3f} bv={bv})")
