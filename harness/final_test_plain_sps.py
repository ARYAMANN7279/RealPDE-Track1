"""Definitive end-to-end test of submission_fno_plain_sps.zip:
extract fresh (as Codabench would), import, run the REAL predict() on the full
891-window held-out set, score with the official scoring.py. Plus robustness:
fresh-subprocess semantics, N=1, 64x128 resolution, finiteness, multiple calls.
"""
import importlib.util, os, sys, time, zipfile, shutil, tempfile
import numpy as np

H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
ZIP="/Users/aryamannsrivastava/Desktop/sem7/UGP/submissions/submission_fno_plain_sps.zip"
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)

work=tempfile.mkdtemp(prefix="finaltest_")
with zipfile.ZipFile(ZIP) as z: z.extractall(work)
sys.path.insert(0, work); cwd=os.getcwd(); os.chdir(work)
s=importlib.util.spec_from_file_location("submission", os.path.join(work,"submission.py"))
sub=importlib.util.module_from_spec(s); s.loader.exec_module(sub); os.chdir(cwd)

X=np.load(f"{H}/real_eval_v3/inputs.npz")["input"]; Y=np.load(f"{H}/real_eval_v3/targets.npz")["target"]
print(f"=== END-TO-END on full held-out {X.shape} (real packaged zip) ===")
t0=time.time(); out=sub.predict(X, {}); dt=time.time()-t0
pred=out["prediction"]; lo=out["lower"]; hi=out["upper"]
print(f"ran predict() in {dt:.0f}s ({dt/X.shape[0]*1000:.0f}ms/sample, CPU single-forward)")
c=scoring.measured_channels(Y)
rel=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(pred,Y,c))))
tke=scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(pred,Y,c))))
mvpe=scoring.score_error(scoring.mvpe_rel_l2(pred,Y))
spsv,cov=scoring.aggregate_sps(pred,Y,c,lower=lo,upper=hi); sps=scoring.score_sps(spsv)
print(f"  L2={rel:.2f} TKE={tke:.2f} MVPE={mvpe:.2f} SPS={sps:.2f} cov={cov*100:.0f}%")
# project final with solved 6-pt weights (time uses ~real GPU value, not CPU)
w=np.array([0.516,0.096,0.093,0.033,0.362]); TIME_REAL=92.66
proj=w@np.array([rel,tke,mvpe,mvpe*0+TIME_REAL,sps]) if False else 0.516*rel+0.096*tke+0.093*mvpe+0.033*TIME_REAL+0.362*sps
print(f"  projected final (6-pt formula, real time~92.66): {proj:.2f}")

print("\n=== ROBUSTNESS ===")
# finiteness / bound order on full set
print(f"  all finite: {bool(np.all(np.isfinite(pred)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)))}")
print(f"  lower<=upper everywhere: {bool(np.all(lo<=hi))}")
print(f"  shape correct (N,20,32,64,3): {pred.shape==(X.shape[0],20,32,64,3)}")
# N=1
o1=sub.predict(X[:1], {}); print(f"  N=1 works: {o1['prediction'].shape==(1,20,32,64,3)}")
# metadata=None (rule: {} on scored calls, but be safe)
oN=sub.predict(X[:2], None); print(f"  metadata=None works: {oN['prediction'].shape==(2,20,32,64,3)}")
# 64x128 native resolution (competition scores 32x64, but robustness)
try:
    Xbig=np.repeat(np.repeat(X[:2],2,axis=2),2,axis=3)
    ob=sub.predict(Xbig,{}); print(f"  64x128 input: OK shape={ob['prediction'].shape}")
except Exception as e:
    print(f"  64x128 input: FAILS ({type(e).__name__}: {str(e)[:60]}) -- only matters if eval isn't 32x64")
# determinism: two calls identical
a=sub.predict(X[:4],{})["prediction"]; b=sub.predict(X[:4],{})["prediction"]
print(f"  deterministic (2 calls identical): {bool(np.array_equal(a,b))}")
shutil.rmtree(work, ignore_errors=True)
