"""Is the FNO's TKE=11.9 a real weakness or an eval artifact?
TKE(pixel) = 0.5*(var_t(u)+var_t(v)) over the T frames of a window.
Compare predicted vs target TKE fields on real held-out windows and decode
WHICH failure mode: over-smoothing (pred var << target), spurious temporal
noise (pred var >> target), or tiny-denominator (target nearly steady).
"""
import importlib.util, os, sys, numpy as np, torch

H = "/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT = "/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
scoring = importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)

X = np.load(f"{H}/real_eval_heldout/inputs.npz")["input"]
Y = np.load(f"{H}/real_eval_heldout/targets.npz")["target"]
import json
meta = json.load(open(f"{H}/real_eval_heldout/meta.json"))
N = 80
X, Y, meta = X[:N], Y[:N], meta[:N]

mi = torch.tensor([0.154960856,-0.000513992854,0.0]); si = torch.tensor([0.0968056545,0.015960684,1.0])
mt = torch.tensor([0.154962569,-0.000517793698,0.0]); st = torch.tensor([0.0968104079,0.0159636438,1.0])
m = load_baseline(f"{H}/fno_model/sim_real_fno_fp16.pth", device="cpu")[0]
with torch.no_grad():
    P = []
    for i in range(0, N, 16):
        xb = torch.from_numpy(X[i:i+16]).float()
        y = m((xb-mi)/si)*st+mt
        x2 = torch.flip(xb, dims=[2]).clone(); x2[...,1] = -x2[...,1]
        y2 = m((x2-mi)/si)*st+mt; y2 = torch.flip(y2,dims=[2]).clone(); y2[...,1] = -y2[...,1]
        P.append((0.5*(y+y2)).numpy())
    P = np.concatenate(P,0).astype(np.float32)

def tke_field(a):  # a: (N,T,H,W,C) -> (N,H,W) using u,v
    u,v = a[...,0], a[...,1]
    return 0.5*(u.var(axis=1)+v.var(axis=1))

tke_p, tke_t = tke_field(P), tke_field(Y)
print(f"windows: {N}")
print(f"target TKE: mean={tke_t.mean():.5f} median={np.median(tke_t):.5f} max={tke_t.max():.5f}")
print(f"pred   TKE: mean={tke_p.mean():.5f} median={np.median(tke_p):.5f} max={tke_p.max():.5f}")
print(f"ratio pred/target (mean of means): {tke_p.mean()/max(tke_t.mean(),1e-9):.2f}x")

# per-window rel-L2 of TKE (what the scorer does)
rel = np.linalg.norm((tke_p-tke_t).reshape(N,-1),axis=1)/np.linalg.norm(tke_t.reshape(N,-1),axis=1).clip(1e-8)
print(f"\nper-window TKE rel-L2: mean={rel.mean():.3f} median={np.median(rel):.3f} min={rel.min():.3f} max={rel.max():.3f}")
# is it dominated by a few near-steady windows (tiny denom)?
order = np.argsort(-rel)
print("worst 5 windows (rel, target_tke_norm, sim_id):")
for i in order[:5]:
    print(f"   rel={rel[i]:7.2f}  ||tke_t||={np.linalg.norm(tke_t[i]):.5f}  {meta[i]['sim_id']}")
print("best 5 windows:")
for i in order[-5:]:
    print(f"   rel={rel[i]:7.3f}  ||tke_t||={np.linalg.norm(tke_t[i]):.5f}  {meta[i]['sim_id']}")

# temporal smoothness: frame-to-frame diff magnitude, pred vs target
def ftd(a): return np.abs(np.diff(a[...,:2],axis=1)).mean()
print(f"\nframe-to-frame |diff|: pred={ftd(P):.5f}  target={ftd(Y):.5f}  (pred>>target => spurious temporal noise)")

# median-based robust TKE score (exclude near-steady windows)
keep = np.linalg.norm(tke_t.reshape(N,-1),axis=1) > np.percentile(np.linalg.norm(tke_t.reshape(N,-1),axis=1),25)
print(f"\nTKE score all windows: {scoring.score_error(rel.mean()):.2f}")
print(f"TKE score excl. bottom-25% steadiest: {scoring.score_error(rel[keep].mean()):.2f}")
