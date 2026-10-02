"""Run ONE submission zip's own predict() on the fixed 734-window set (_agent2/winset.npz,
the set §32 used) and save that zip's per-element bound geometry AND its own prediction.
Derived from _agent2/eval_zip.py; writes only under agents/bounds/cache734/.
One zip per process (module-level state).

Saves float32 (N,20,32,64,2), channels u,v:
  P_<tag>  prediction      C_<tag> = (lower+upper)/2 - prediction
  HD_<tag> = centre - lower        HU_<tag> = upper - centre
and TGT.npy once.  Prints a constant-fallback count (must be 0).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
OUT = f"{B}/agents/bounds/cache734"

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--keep-extract", action="store_true")
ap.add_argument("--windows", default="734", choices=["734", "r12_900"],
                help="734 = _agent2/winset.npz (§32 set); r12_900 = r12_eval.val_starts(900) (canonical ruler)")
a = ap.parse_args()
if a.windows == "r12_900":
    OUT = f"{B}/agents/bounds/cache900"
os.makedirs(OUT, exist_ok=True)

if a.windows == "734":
    starts = np.load(f"{B}/_agent2/winset.npz")["starts"]
else:
    sys.path.insert(0, B)
    import r12_eval  # noqa: E402
    starts = np.asarray(r12_eval.val_starts(900)[0])
    np.save(f"{OUT}/starts.npy", starts)
N = len(starts)
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")

d = f"{OUT}/x_{a.tag}"
if os.path.exists(d):
    shutil.rmtree(d)
os.makedirs(d)
subprocess.run(["unzip", "-q", a.zip, "-d", d], check=True)
sys.path.insert(0, d)
import submission  # noqa: E402

if hasattr(submission, "_TIME_BUDGET"):
    submission._TIME_BUDGET = 1e9          # local measurement only: never trip the fallback
print("[mod]", submission.__file__, flush=True)

shp = (N, 20, 32, 64, 2)
P = np.zeros(shp, np.float32)
C = np.zeros(shp, np.float32)
HD = np.zeros(shp, np.float32)
HU = np.zeros(shp, np.float32)
need_tgt = not os.path.exists(f"{OUT}/TGT.npy")
TGT = np.zeros(shp, np.float32) if need_tgt else None
nfb = 0
t0 = time.time()
for i in range(0, N, a.batch):
    j = min(i + a.batch, N)
    w = np.stack([np.concatenate(
        [np.asarray(X[s:s + 40]), np.zeros((40, 32, 64, 1), np.float32)], -1)
        for s in starts[i:j]]).astype(np.float32)
    r = submission.predict(w[:, :20])
    p = r["prediction"][..., :2].astype(np.float32)
    lo = r["lower"][..., :2].astype(np.float32)
    up = r["upper"][..., :2].astype(np.float32)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(up)) and np.all(lo <= up)
    ctr = 0.5 * (lo + up)
    P[i:j] = p
    C[i:j] = ctr - p
    HD[i:j] = ctr - lo
    HU[i:j] = up - ctr
    # constant-fallback detector: whole window at pred +- [0.0129, 0.0098]
    hh = 0.5 * (up - lo)
    fb = (np.abs(hh[..., 0] - 0.0129).max(axis=(1, 2, 3)) < 1e-6) & \
         (np.abs(hh[..., 1] - 0.0098).max(axis=(1, 2, 3)) < 1e-6)
    nfb += int(fb.sum())
    if need_tgt:
        TGT[i:j] = w[:, 20:, ..., :2]
print("[pred] %d windows %.1fs  constant-fallback windows: %d" % (N, time.time() - t0, nfb),
      flush=True)
for nm, A in (("P", P), ("C", C), ("HD", HD), ("HU", HU)):
    np.save(f"{OUT}/{nm}_{a.tag}.npy", A)
if need_tgt:
    np.save(f"{OUT}/TGT.npy", TGT)
if not a.keep_extract:
    shutil.rmtree(d, ignore_errors=True)
print("[done] %s  median h u %.6f v %.6f | median|c| u %.6f v %.6f"
      % (a.tag, np.median(HD[..., 0] + HU[..., 0]) / 2, np.median(HD[..., 1] + HU[..., 1]) / 2,
         np.median(np.abs(C[..., 0])), np.median(np.abs(C[..., 1]))), flush=True)
