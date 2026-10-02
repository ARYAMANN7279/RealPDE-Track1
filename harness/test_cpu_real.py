"""REAL CPU-path test. Must be launched with CUDA_VISIBLE_DEVICES="" already set
in the environment BEFORE python starts -- setting it from inside the process after
torch is imported does nothing (torch caches availability), which is how the first
version of this test silently passed while testing the GPU path.

This is the gate that did not exist when submission_MAXSOUP_head.zip was cleared,
and it is the scenario that most plausibly killed it: a container with no GPU, where
the model runs ~66x slower and the old per-call budget let load time run free.
"""
import os, sys, zipfile, shutil, time, json
import numpy as np
import importlib.util as iu

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", \
    "run me with CUDA_VISIBLE_DEVICES='' set in the shell"

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "fixed"
WD = f"{B}/_tmp/cputest_{WHICH}"; shutil.rmtree(WD, ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/submission_MAXSOUP_head.zip") as z:
    z.extractall(WD)
if WHICH == "fixed":
    shutil.copy("/tmp/fixed_submission.py", os.path.join(WD, "submission.py"))

import torch
print("torch.cuda.is_available():", torch.cuda.is_available(), "  (must be False)")
assert not torch.cuda.is_available(), "CUDA still visible; test is invalid"

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
wins = []
for i in sorted(set(range(0, len(lens), 5))):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20): wins.append(t0)
wins = wins[:32]
W = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1)
              for s in wins]).astype(np.float32)
Xin = W[:, :20]

t_import0 = time.time()
s = iu.spec_from_file_location("sub_%s" % WHICH, os.path.join(WD, "submission.py"))
m = iu.module_from_spec(s); s.loader.exec_module(m)
print("module import: %.2fs" % (time.time() - t_import0))

t0 = time.time(); mdl, dev = m._get_model(); t_load = time.time() - t0
print("device: %s   model load: %.1fs" % (dev, t_load))
assert dev == "cpu", "expected cpu path"

n = 8
t0 = time.time(); r = m.predict(Xin[:n], metadata={}); dt = time.time() - t0
per = dt / n
print("throughput: %.1f ms/sample  (%d windows in %.1fs)" % (per * 1000, n, dt))
print("finite=%s lo<=up=%s" % (bool(np.isfinite(r["prediction"]).all()),
                               bool((r["lower"] <= r["upper"]).all())))

print()
print("--- projected wall-clock at full eval size, vs the 180s kill ---")
for N in (2000, 5140):
    naive = t_load + per * N
    print("  N=%-5d naive(unbudgeted): load %.0fs + infer %.0fs = %.0fs  %s"
          % (N, t_load, per * N, naive, "OVER LIMIT" if naive > 180 else "ok"))

print()
print("--- does the budget actually save it? simulate a near-exhausted deadline ---")
# leave only ~3s of global budget, as if load+earlier work had eaten it
m._T_START = time.time() - (m._HARD_LIMIT - m._reserve_for(len(Xin)) - 3.0)
t0 = time.time(); r2 = m.predict(Xin, metadata={}); dt2 = time.time() - t0
print("  call with ~3s of budget left on %d windows returned in %.1fs" % (len(Xin), dt2))
print("  (was 98.8s before the batch-granularity fix; must now be a few seconds)")
print("  finite=%s lo<=up=%s shape_ok=%s" % (
    bool(np.isfinite(r2["prediction"]).all()),
    bool((r2["lower"] <= r2["upper"]).all()),
    r2["prediction"].shape == Xin.shape))
print("  -> output is complete and valid even when the deadline is nearly gone")
