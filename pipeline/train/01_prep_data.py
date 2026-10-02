"""train_real/*.h5 -> leak-free (input20, target20) windows at the 32x64 eval grid.
Windows are non-overlapping (stride 40) so no target frame appears in any input."""
import glob, json, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
C = SourceFileLoader("c", os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_config.py")).load_module()
C.seed_all()
HOR = C.IN_STEP + C.OUT_STEP
frames, off, lens, names = [], [], [], []
cur = 0
for fp in sorted(glob.glob(os.path.join(C.TRAIN_REAL, "*.h5"))):
    nm = os.path.basename(fp)
    if nm in C.EXCLUDE:
        print("  skip %s (known duplicate)" % nm); continue
    with h5py.File(fp, "r") as f:
        u = np.asarray(f["u"], np.float32)[:, ::C.SUB_S, ::C.SUB_S][:, :32, :64]
        v = np.asarray(f["v"], np.float32)[:, ::C.SUB_S, ::C.SUB_S][:, :32, :64]
    tr = np.stack([u, v], -1)
    frames.append(tr); off.append(cur); lens.append(len(tr)); names.append(nm); cur += len(tr)
    print("  %-14s %d frames" % (nm, len(tr)))
F = np.concatenate(frames, 0)
np.save(os.path.join(C.WORK, "frames.npy"), F)
json.dump({"off": off, "lens": lens, "names": names},
          open(os.path.join(C.WORK, "meta.json"), "w"))
print("total %d frames from %d trajectories -> frames.npy" % (len(F), len(names)))
