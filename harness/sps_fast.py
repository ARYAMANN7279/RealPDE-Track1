"""Fast, correct SPS bound optimization. Reuses the cached FNO TTA predictions.
Precomputes the bound-independent per-sample accuracy weight W, so each bound
candidate is a couple of array ops. Verified against the official aggregate_sps
on the shipped fixed bounds. Honest split: tune on 2 held-out trajectories,
report on the disjoint 2.
"""
import importlib.util, os, sys, numpy as np

H = "/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT = "/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
scoring = importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)
SIGMA = scoring.SIGMA_GLOBAL

d = np.load(f"{H}/real_eval_heldout/fno_tta_cache.npz", allow_pickle=True)
P1, P2, Y, sims = d["P1"], d["P2"], d["Y"], d["sims"]
pred = (0.5*(P1+P2)).astype(np.float32)
disag = np.abs(P1-P2).astype(np.float32)
C = 2
p = pred[..., :C]; t = Y[..., :C]; dis = disag[..., :C]
scored = t != 0.0

# per-sample accuracy weight W (bound-independent)
dm = scoring.rel_l2_per_sample(pred, Y, C); tke = scoring.tke_rel_l2_per_sample(pred, Y, C); mvpe = scoring.mvpe_rel_l2_per_sample(pred, Y)
def pm(x): x = x/(0.5+x); return np.where(np.isfinite(x), 1.0-x, 0.0)
Ws = 0.5*pm(dm) + 0.3*pm(tke) + 0.2*pm(mvpe)          # (N,)
Wpix = Ws.reshape((-1,)+(1,)*(p.ndim-1))               # broadcastable

abserr = np.abs(t - p)

def sps_fast(half, mask):
    # half: scalar-per-channel array broadcastable to p, or per-pixel; mask: window selector
    inside = abserr <= half
    nil = (2.0*half)/SIGMA * np.ones_like(p)
    elem = Wpix * np.exp(-nil) * inside * scored
    sel = mask.reshape((-1,)+(1,)*(p.ndim-1))
    num = np.sum(np.where(sel, elem, 0.0), dtype=np.float64)
    den = np.count_nonzero(scored & sel)
    w = num/den if den else 0.0
    cov = np.count_nonzero(inside & scored & sel)/den if den else 0.0
    return 100.0*min(max(w,0.0),1.0), cov*100

uniq = sorted(set(sims.tolist()))
tune = np.isin(sims, uniq[::2]); test = ~tune
allm = np.ones_like(tune)

def hc(hu, hv):
    a = np.zeros((1,)*(p.ndim-1)+(C,), np.float32); a[...,0]=hu; a[...,1]=hv; return a

# sanity: fast vs official on fixed bounds (whole set)
fh = hc(0.107537, 0.010307)
off = scoring.score_sps(scoring.aggregate_sps(pred, Y, C, lower=pred-fh, upper=pred+fh)[0])
fast = sps_fast(fh, allm)[0]
print(f"VERIFY fixed bounds: official={off:.3f}  fast={fast:.3f}  (match={abs(off-fast)<0.2})\n")

print(f"tune={sorted(uniq[::2])}  test={sorted(uniq[1::2])}\n")
print(f"{'strategy':18s} {'tuneSPS':>8s} {'testSPS':>8s} {'testcov':>8s}")
s_t,c_t = sps_fast(fh, test); print(f"{'fixed(shipped)':18s} {sps_fast(fh,tune)[0]:8.2f} {s_t:8.2f} {c_t:7.1f}%")

# const-opt: fine grid on tune, report test
grid = np.linspace(0.004, 0.25, 80)
best=(-1,)
for hu in grid:
    for hv in grid:
        s,_ = sps_fast(hc(hu,hv), tune)
        if s>best[0]: best=(s,hu,hv)
_,hu,hv = best; s_t,c_t = sps_fast(hc(hu,hv), test)
print(f"{'const-opt':18s} {best[0]:8.2f} {s_t:8.2f} {c_t:7.1f}%   (hu={hu:.3f} hv={hv:.3f})")

# adaptive: half = a + b*disagreement (per channel), grid on tune
def ha(au,bu,av,bv):
    h = np.empty_like(p); h[...,0]=au+bu*dis[...,0]; h[...,1]=av+bv*dis[...,1]; return h
best=(-1,)
for au in np.linspace(0.0,0.12,7):
    for bu in [0,0.5,1,2,3,5]:
        for av in np.linspace(0.0,0.03,7):
            for bv in [0,0.5,1,2,3,5]:
                s,_ = sps_fast(ha(au,bu,av,bv), tune)
                if s>best[0]: best=(s,au,bu,av,bv)
_,au,bu,av,bv = best; s_t,c_t = sps_fast(ha(au,bu,av,bv), test)
print(f"{'adaptive-TTA':18s} {best[0]:8.2f} {s_t:8.2f} {c_t:7.1f}%   (au={au:.3f} bu={bu} av={av:.3f} bv={bv})")
