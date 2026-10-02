"""Map predicted log|error| -> the half-width that maximises the SPS integrand.
SOUP variant: NO temporal spectral correction.

Measured on the live board (v7 vs ROBUST) the spectral correction is net -0.09:
it buys rel_l2 -0.32 / tke +0.94 (= +0.055 final) but costs -1.47 on the time
subscore (= -0.147). The soup already fixes tke by +4.20 through training, so
the correction's marginal accuracy gain here would be smaller still while its
time cost is unchanged. It is therefore dropped from BOTH this fit and
submission.py -- they must agree, or the LUT is fitted to a prediction
distribution that never occurs at inference.

This is where the SPS trade-off lives. For each bin of predicted error we solve
    argmax_h  exp(-2h/sigma) * P(|err| <= h)
exactly, by sorting that bin's errors. Non-parametric, so it recovers whatever the
true shape is -- empirically h GROWS with predicted error but SUB-linearly, so
h = k*exp(mu) (which we tried first) is measurably worse."""
import json, os, sys
import numpy as np, torch
from importlib.machinery import SourceFileLoader
HERE = os.path.dirname(os.path.abspath(__file__))
C = SourceFileLoader("c", os.path.join(HERE, "00_config.py")).load_module()
C.seed_all(); sys.path.insert(0, HERE)
from head_common import feats, Net, calibrate, global_scale, anchor_scale, SIGMA_GLOBAL as SIG
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(os.path.join(C.WORK, "cache.npz"))
P = d["P"].astype(np.float32); Y = d["Y"].astype(np.float32)
# NO spectral gain applied -- see module docstring. gt is written as ones below so
# that any code path that still multiplies by it is an exact no-op.
g = np.ones((11, 2), np.float32)
SCM = (Y[..., :2] != 0.0)
# CRITICAL: use CLEAN errors with a GLOBAL per-channel scale. Do NOT use the
# per-element (alpha+beta*|pred|) transform -- it reorders elements 1.6x-2.75x
# across the field and destroys the ranking the LUT exists to encode. That bug
# cost 0.063 in E and made the head score BELOW plain constants (live: 76.61).
ERR = np.abs(P[..., :2] - Y[..., :2]).astype(np.float32)
# GATE 6: the SHIPPED artifact uses the anchor-refitted scale, not global_scale's
# Weibull fit (which mis-scales v by ~38%; see head_common.anchor_scale). Set
# USE_GLOBAL_SCALE=1 to reproduce the OLD, superseded LUT.
if os.environ.get("USE_GLOBAL_SCALE") == "1":
    SCALE = global_scale(ERR, SCM)
else:
    SCALE = anchor_scale()
for _ci in range(2):
    ERR[..., _ci] *= SCALE[_ci]
print("  global per-channel scale: u x%.3f  v x%.3f" % (SCALE[0], SCALE[1]))
HEAD_IN = os.environ.get("HEAD_IN", "head.pth")
ASSETS_OUT = os.environ.get("ASSETS_OUT", "head_assets.npz")
print("  head=%s -> %s" % (HEAD_IN, ASSETS_OUT))
ck = torch.load(os.path.join(C.WORK, HEAD_IN), map_location=dev, weights_only=False)
NF = ck["nf"]; net = Net(NF).to(dev); net.load_state_dict(ck["sd"]); net.eval()
FT = (feats(P) - ck["mu"]) / ck["std"]
MU = []
with torch.no_grad():
    for i in range(0, len(P), 8):
        f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(dev).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        MU.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
MU = np.concatenate(MU, 0)
def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
NB = 24; LUT = np.zeros((NB,2), np.float32); ED = np.zeros((2,NB-1), np.float32)
for ci in range(2):
    m = SCM[..., ci]; mu = MU[..., ci][m]; er = ERR[..., ci][m]
    q = np.quantile(mu, np.linspace(0,1,NB+1)[1:-1]); ED[ci] = q
    b = np.digitize(mu, q)
    for k in range(NB):
        s = b == k
        LUT[k, ci] = best_h(er[s][::3]) if s.sum() > 200 else 0.012
# GATE: thresholds calibrated against the two REAL artifacts, not guessed.
#   v6 (corrupted, sps 28.38): h_u min 0.00043, h_v min 0.00448
#   v7 (clean,     sps 33.41): h_u min 0.01116, h_v min 0.00252
# The corruption showed ONLY in u. v naturally runs ~6x smaller than u because the
# v field is ~6x smaller (std 0.0160 vs 0.0968), so v7's h_v 0.0025 is healthy and a
# threshold above it would reject a known-good artifact. Guard u tightly, v loosely.
assert LUT[:, 0].min() > 0.005, "LUT u has near-zero half-widths (min %.5f) -- corrupted ranking (v6 was 0.00043)" % LUT[:, 0].min()
assert LUT[:, 1].min() > 0.0015, "LUT v has near-zero half-widths (min %.5f) -- v7 good is 0.00252" % LUT[:, 1].min()
sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
np.savez_compressed(os.path.join(C.WORK, ASSETS_OUT), gt=g, LUT=LUT, ED=ED,
                    fmu=ck["mu"], fsd=ck["std"], nf=np.int32(NF),
                    **{"w_"+k: v for k, v in sd.items()})
print("LUT h_u %.4f .. %.4f  (median %.4f)" % (LUT[:,0].min(), LUT[:,0].max(), np.median(LUT[:,0])))
print("LUT h_v %.4f .. %.4f  (median %.4f) -> head_assets.npz" % (LUT[:,1].min(), LUT[:,1].max(), np.median(LUT[:,1])))
