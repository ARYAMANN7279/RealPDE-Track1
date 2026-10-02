"""Optimize SPS interval bounds for the FNO, HONESTLY: tune bound parameters on
one group of held-out trajectories, evaluate on a disjoint group. Low-capacity
(<=4 scalars) so it cannot overfit the way the XGBoost residual did.

Strategies compared:
  fixed      : the shipped constant half-bound [0.1075, 0.0103]
  const-opt  : best per-channel CONSTANT half-width (2 params)
  adaptive   : half = a + b*|y_tta1 - y_tta2|  (per-pixel, TTA disagreement; 4 params)
"""
import importlib.util, os, sys, json, numpy as np, torch

H = "/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT = "/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
scoring = importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)

CACHE = f"{H}/real_eval_heldout/fno_tta_cache.npz"
mi = torch.tensor([0.154960856,-0.000513992854,0.0]); si = torch.tensor([0.0968056545,0.015960684,1.0])
mt = torch.tensor([0.154962569,-0.000517793698,0.0]); st = torch.tensor([0.0968104079,0.0159636438,1.0])

def gen_cache():
    X = np.load(f"{H}/real_eval_heldout/inputs.npz")["input"]
    Y = np.load(f"{H}/real_eval_heldout/targets.npz")["target"]
    meta = json.load(open(f"{H}/real_eval_heldout/meta.json"))
    m = load_baseline(f"{H}/fno_model/sim_real_fno_fp16.pth", device="cpu")[0]
    P1s, P2s = [], []
    with torch.no_grad():
        for i in range(0, X.shape[0], 16):
            xb = torch.from_numpy(X[i:i+16]).float()
            y1 = (m((xb-mi)/si)*st+mt).numpy()
            x2 = torch.flip(xb, dims=[2]).clone(); x2[...,1] = -x2[...,1]
            y2 = m((x2-mi)/si)*st+mt; y2 = torch.flip(y2,dims=[2]).clone(); y2[...,1] = -y2[...,1]
            P1s.append(y1); P2s.append(y2.numpy())
    P1 = np.concatenate(P1s,0).astype(np.float32); P2 = np.concatenate(P2s,0).astype(np.float32)
    sims = np.array([mm["sim_id"] for mm in meta])
    np.savez(CACHE, P1=P1, P2=P2, Y=Y, sims=sims)
    print("cached", CACHE)

def sps_of(pred, Y, half, c=2):
    lo, hi = pred-half, pred+half
    val, cov = scoring.aggregate_sps(pred, Y, c, lower=lo, upper=hi)
    return scoring.score_sps(val), cov*100

def main():
    if not os.path.exists(CACHE): gen_cache()
    d = np.load(CACHE, allow_pickle=True)
    P1, P2, Y, sims = d["P1"], d["P2"], d["Y"], d["sims"]
    pred = 0.5*(P1+P2)
    disag = np.abs(P1-P2)  # (N,T,H,W,C) per-pixel TTA disagreement
    uniq = sorted(set(sims.tolist()))
    print("trajectories:", uniq)
    # honest split: odd-indexed trajs = tune, even = test
    tune_tr = set(uniq[::2]); test_tr = set(uniq[1::2])
    tune = np.array([s in tune_tr for s in sims]); test = ~tune
    print(f"tune windows {tune.sum()} ({sorted(tune_tr)}) | test windows {test.sum()} ({sorted(test_tr)})\n")

    def half_const(hu, hv):
        h = np.zeros((1,1,1,1,3), np.float32); h[...,0]=hu; h[...,1]=hv; return h
    def half_adapt(au,bu,av,bv,dis):
        h = np.zeros_like(dis); h[...,0]=au+bu*dis[...,0]; h[...,1]=av+bv*dis[...,1]; return h

    # fixed (shipped)
    fh = half_const(0.107537, 0.010307)
    print(f"fixed(shipped)  tune SPS={sps_of(pred[tune],Y[tune],fh)[0]:.2f}  test SPS={sps_of(pred[test],Y[test],fh)[0]:.2f} cov={sps_of(pred[test],Y[test],fh)[1]:.1f}%")

    # const-opt: grid per channel on TUNE, report TEST
    grid = np.linspace(0.005, 0.2, 40)
    best=None
    for hu in grid:
        for hv in grid:
            s,_=sps_of(pred[tune],Y[tune],half_const(hu,hv))
            if best is None or s>best[0]: best=(s,hu,hv)
    _,hu,hv=best
    ts,tc=sps_of(pred[test],Y[test],half_const(hu,hv))
    print(f"const-opt       tune SPS={best[0]:.2f} (hu={hu:.3f} hv={hv:.3f})  test SPS={ts:.2f} cov={tc:.1f}%")

    # adaptive on TUNE (coarse grid), report TEST
    best=None
    for au in [0.0,0.02,0.05,0.08]:
        for bu in [0.0,1.0,2.0,4.0]:
            for av in [0.0,0.005,0.01,0.02]:
                for bv in [0.0,1.0,2.0,4.0]:
                    h=half_adapt(au,bu,av,bv,disag[tune])
                    s,_=sps_of(pred[tune],Y[tune],h)
                    if best is None or s>best[0]: best=(s,au,bu,av,bv)
    _,au,bu,av,bv=best
    ht=half_adapt(au,bu,av,bv,disag[test])
    ts,tc=sps_of(pred[test],Y[test],ht)
    print(f"adaptive-TTA    tune SPS={best[0]:.2f} (au={au} bu={bu} av={av} bv={bv})  test SPS={ts:.2f} cov={tc:.1f}%")

if __name__=="__main__":
    main()
