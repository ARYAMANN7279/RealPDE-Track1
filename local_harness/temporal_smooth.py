"""Does temporal smoothing of the FNO's 20-frame output reduce its spurious
frame-to-frame jitter enough to help TKE WITHOUT hurting L2/MVPE?
Runs instantly from the cached FNO predictions (no re-inference). Reports the
full metric set for a range of smoothing strengths, per-trajectory too.
"""
import importlib.util, os, json, numpy as np
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
E=f"{H}/real_eval_heldout_big"
d=np.load(f"{E}/fno_tta_cache.npz");pred0=(0.5*(d["P1"]+d["P2"])).astype(np.float32)
Y=np.load(f"{E}/targets.npz")["target"];meta=json.load(open(f"{E}/meta.json"))
sims=np.array([m["sim_id"] for m in meta]);C=2

def smooth_time(a, w):
    """moving-average along time axis (axis=1), window w (odd), edge-padded."""
    if w<=1: return a
    k=w//2
    ap=np.pad(a,((0,0),(k,k),(0,0),(0,0),(0,0)),mode="edge")
    out=np.zeros_like(a)
    for i in range(w): out+=ap[:,i:i+a.shape[1]]
    return (out/w).astype(np.float32)

def gauss_time(a, sigma):
    if sigma<=0: return a
    T=a.shape[1]; xs=np.arange(-3,4)
    k=np.exp(-0.5*(xs/sigma)**2); k/=k.sum()
    ap=np.pad(a,((0,0),(3,3),(0,0),(0,0),(0,0)),mode="edge")
    out=np.zeros_like(a)
    for i,ki in enumerate(k): out+=ki*ap[:,i:i+T]
    return out.astype(np.float32)

def metrics(pred):
    rel=float(np.mean(scoring.rel_l2_per_sample(pred,Y,C)))
    tke=float(np.mean(scoring.tke_rel_l2_per_sample(pred,Y,C)))
    mvpe=scoring.mvpe_rel_l2(pred,Y)
    return scoring.score_error(rel),scoring.score_error(tke),scoring.score_error(mvpe)

def ftd(a): return float(np.abs(np.diff(a[...,:2],axis=1)).mean())
print(f"target frame-to-frame |diff|: {ftd(Y):.5f}   FNO raw: {ftd(pred0):.5f}  (ratio {ftd(pred0)/ftd(Y):.1f}x)\n")
print(f"{'post-process':18s}{'L2':>7s}{'TKE':>7s}{'MVPE':>7s}{'acc3':>7s}{'ftd':>9s}")
def row(name,p):
    l,t,m=metrics(p);print(f"{name:18s}{l:7.2f}{t:7.2f}{m:7.2f}{(l+t+m)/3:7.2f}{ftd(p):9.5f}")
row("none (raw FNO)",pred0)
for w in (3,5,7):
    row(f"movavg w={w}", smooth_time(pred0,w))
for sg in (0.7,1.0,1.5,2.0):
    row(f"gauss s={sg}", gauss_time(pred0,sg))
# blend raw with smoothed (partial): keeps per-frame accuracy, trims jitter
for al in (0.3,0.5,0.7):
    p=(al*gauss_time(pred0,1.0)+(1-al)*pred0).astype(np.float32)
    row(f"blend g1*{al}", p)

# best-looking: check per-trajectory TKE for gauss s=1.0 vs raw
print("\nper-traj TKE score  raw -> gauss(s=1.0):")
g=gauss_time(pred0,1.0)
for s in sorted(set(sims.tolist())):
    mk=sims==s
    tr=scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(pred0[mk],Y[mk],C))))
    tg=scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(g[mk],Y[mk],C))))
    print(f"   {s:16s} {tr:6.2f} -> {tg:6.2f}")
