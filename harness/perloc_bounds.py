"""Optimal STATIC per-(H,W,channel) bound: tight in freestream, wide in wake.
Physically-grounded (the wake location is stable), so it should generalize.
Optimize each location's bound for SPS on TUNE trajectories, evaluate on TEST.
Bounds-only => zero accuracy risk. FNO (reliable proxy).
"""
import importlib.util, os, json, numpy as np
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness";KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
SIGMA=scoring.SIGMA_GLOBAL
P=np.load(f"{H}/real_eval_v3/fno_tta_cache.npz")["P1"].astype(np.float32)
Y=np.load(f"{H}/real_eval_v3/targets.npz")["target"];meta=json.load(open(f"{H}/real_eval_v3/meta.json"))
sims=np.array([m["sim_id"] for m in meta]);C=2
N,T,Hh,W,_=P.shape
uniq=sorted(set(sims.tolist())); tune=np.isin(sims,uniq[::2]); test=~sims.astype(bool) if False else ~np.isin(sims,uniq[::2])
# per-window accuracy weight W (bound-independent)
Ws=(0.5*(1-scoring.rel_l2_per_sample(P,Y,C)/(0.5+scoring.rel_l2_per_sample(P,Y,C)))
   +0.3*(1-np.nan_to_num(scoring.tke_rel_l2_per_sample(P,Y,C)/(0.5+scoring.tke_rel_l2_per_sample(P,Y,C))))
   +0.2*(1-np.nan_to_num(scoring.mvpe_rel_l2_per_sample(P,Y)/(0.5+scoring.mvpe_rel_l2_per_sample(P,Y))))).astype(np.float32)
ab=np.abs(Y[...,:C]-P[...,:C]).astype(np.float32)   # (N,T,H,W,C)
scored=(Y[...,:C]!=0.0)

def score_half(half, sel):  # half: (H,W,C) static; sel: window mask
    h3=np.zeros((half.shape[0],half.shape[1],3),np.float32); h3[...,:half.shape[2]]=half
    h=h3[None,None]
    s,cov=scoring.aggregate_sps(P[sel],Y[sel],C,lower=P[sel]-h,upper=P[sel]+h)
    return scoring.score_sps(s),cov*100

# optimize per-location bound on TUNE: for each (h,w,ch), maximize exp(-2b/sig)*sum(W*[|err|<=b]) over tune (window,time)
# candidate bounds = a grid; vectorized.
grid=np.linspace(0.002,0.12,40).astype(np.float32)
abt=ab[tune]                      # (Ntune,T,H,W,C)
Wt=Ws[tune][:,None,None,None]     # (Ntune,1,1,1)
half=np.zeros((Hh,W,C),np.float32)
for ch in range(C):
    a=abt[...,ch]                 # (Ntune,T,H,W)
    Wexp=np.broadcast_to(Wt,a.shape)
    # for each grid bound g: covered = a<=g ; score_g(loc)=exp(-2g/sig)*sum_over(win,time)(W*covered)
    best_val=np.full((Hh,W),-1.0,np.float32); best_b=np.full((Hh,W),grid[0],np.float32)
    for g in grid:
        cov=(a<=g)
        num=np.einsum('ntij,ntij->ij', Wexp, cov.astype(np.float32))  # (H,W)
        val=np.exp(-2*g/SIGMA)*num
        upd=val>best_val; best_val[upd]=val[upd]; best_b[upd]=g
    half[...,ch]=best_b
# evaluate
hc=np.zeros((Hh,W,C),np.float32); hc[...,0]=0.030; hc[...,1]=0.010
print(f"constant [0.030,0.010]  TEST SPS={score_half(hc,test)[0]:.2f}  TUNE={score_half(hc,tune)[0]:.2f}")
print(f"per-location optimal     TEST SPS={score_half(half,test)[0]:.2f} cov={score_half(half,test)[1]:.0f}%  TUNE={score_half(half,tune)[0]:.2f}")
oer=(ab+1e-4); oer3=np.zeros((N,T,Hh,W,3),np.float32); oer3[...,:C]=oer
so,_=scoring.aggregate_sps(P[test],Y[test],C,lower=P[test]-oer3[test],upper=P[test]+oer3[test])
print(f"oracle                   TEST SPS={scoring.score_sps(so):.2f}")
# show the learned bound map (freestream vs wake)
print(f"\nlearned bound u-channel: min={half[...,0].min():.3f} max={half[...,0].max():.3f} median={np.median(half[...,0]):.3f}")
print(f"  (tight where freestream, wide in wake — if median<<0.030, it found the freestream)")
np.save(f"{H}/real_eval_v3/perloc_half.npy", half)
