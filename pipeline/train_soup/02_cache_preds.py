"""Run the MODEL SOUP over the prepared windows and cache predictions.

Same as train/02_cache_preds.py except the weights loaded are the soup
(weight-average of 6 fine-tunes of the Drive baseline) instead of the baseline.
load_baseline infers ARCHITECTURE FROM THE FILENAME, so the architecture is
still built from sim_real_fno.pth and the soup weights are loaded into it --
never pass soup_best.pth to load_baseline directly.
"""
import json, os, sys
import numpy as np, torch
from importlib.machinery import SourceFileLoader
HERE = os.path.dirname(os.path.abspath(__file__))
C = SourceFileLoader("c", os.path.join(HERE, "00_config.py")).load_module()
C.seed_all()
sys.path.insert(0, C.KIT); sys.path.insert(0, os.path.join(C.KIT, "_vendor"))
from load_baseline import load_baseline
MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
dev = "cuda" if torch.cuda.is_available() else "cpu"
F = np.load(os.path.join(C.WORK, "frames.npy"), mmap_mode="r")
m = json.load(open(os.path.join(C.WORK, "meta.json")))
HOR = C.IN_STEP + C.OUT_STEP
wins, wtraj = [], []
for i, (o, L) in enumerate(zip(m["off"], m["lens"])):
    for t0 in range(o, o + L - HOR + 1, HOR):
        wins.append(t0); wtraj.append(i)
W = np.stack([np.concatenate([np.asarray(F[s:s+HOR]),
     np.zeros((HOR, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
X, Y = W[:, :C.IN_STEP], W[:, C.IN_STEP:]

model, _ = load_baseline(C.CKPT_FP32, device=dev)      # architecture only
sd = torch.load(C.SOUP_SD, map_location=dev)
sd = sd.get("model_state_dict", sd)

# GATE: the soup must still carry intact complex spectral weights. Averaging via
# .float() silently discards the imaginary part and scored -7.34 once already.
ncplx = sum(1 for v in sd.values() if torch.is_tensor(v) and v.is_complex())
nzero = sum(1 for v in sd.values()
            if torch.is_tensor(v) and v.is_complex() and v.imag.abs().max().item() == 0.0)
assert ncplx == 16, "expected 16 complex tensors, found %d" % ncplx
assert nzero == 0, "%d complex tensors have a ZERO imaginary part -- soup is corrupt" % nzero
print("[soup] %d complex tensors, all imaginary parts non-zero" % ncplx)

model.load_state_dict(sd); model = model.to(dev).eval()
mi, si, mt, st = [torch.tensor(a).to(dev) for a in (MI, SI, MT, ST)]
out = []
with torch.no_grad():
    for i in range(0, len(X), 32):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i+32])).to(dev)
        out.append((model((xb - mi) / si) * st + mt).cpu().numpy())
P = np.concatenate(out, 0).astype(np.float32)
np.savez_compressed(os.path.join(C.WORK, "cache.npz"),
                    P=P.astype(np.float16), Y=Y.astype(np.float16),
                    wtraj=np.array(wtraj))
print("cached %d windows over %d trajectories -> cache.npz" % (len(P), len(m["names"])))
