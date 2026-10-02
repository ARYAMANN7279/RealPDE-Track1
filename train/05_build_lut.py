"""Map predicted log|error| -> the half-width that maximises the SPS integrand.

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
from head_common import feats, Net, calibrate, global_scale, SIGMA_GLOBAL as SIG
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(os.path.join(C.WORK, "cache.npz"))
P = d["P"].astype(np.float32); Y = d["Y"].astype(np.float32)
g = np.load(os.path.join(C.WORK, "spectral_gain.npy"))
Fq = np.fft.rfft(P[..., :2], axis=1) * g[None, :, None, None, :]
P[..., :2] = np.fft.irfft(Fq, n=20, axis=1)
SCM = (Y[..., :2] != 0.0)
# CRITICAL: use CLEAN errors with a GLOBAL per-channel scale. Do NOT use the
# per-element (alpha+beta*|pred|) transform -- it reorders elements 1.6x-2.75x
# across the field and destroys the ranking the LUT exists to encode. That bug
# cost 0.063 in E and made the head score BELOW plain constants (live: 76.61).
ERR = np.abs(P[..., :2] - Y[..., :2]).astype(np.float32)
SCALE = global_scale(ERR, SCM)
for _ci in range(2):
    ERR[..., _ci] *= SCALE[_ci]
print("  global per-channel scale: u x%.3f  v x%.3f" % (SCALE[0], SCALE[1]))
ck = torch.load(os.path.join(C.WORK, "head.pth"), map_location=dev, weights_only=False)
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
sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
np.savez_compressed(os.path.join(C.WORK, "head_assets.npz"), gt=g, LUT=LUT, ED=ED,
                    fmu=ck["mu"], fsd=ck["std"], nf=np.int32(NF),
                    **{"w_"+k: v for k, v in sd.items()})
print("LUT h_u %.4f .. %.4f -> head_assets.npz" % (LUT[:,0].min(), LUT[:,0].max()))
