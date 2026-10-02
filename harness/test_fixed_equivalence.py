"""Regression + CPU-path test for the fixed submission.py.

Three things must hold before this ships:
  1. EQUIVALENCE -- fixed predict() must return bit-identical prediction/lower/upper
     to the shipped version on the happy path. The fix is a timing/memory change and
     must not alter model behaviour at all.
  2. CPU PATH -- the run that actually failed. Force device to cpu, measure real
     throughput, and extrapolate whether load + inference + post-processing fits in
     180s. This is the gate that did not exist and would have caught the failure.
  3. BUDGET SEMANTICS -- the global deadline must be anchored at import (so load
     counts) and must actually trip.
"""
import os, sys, zipfile, shutil, time, json
import numpy as np
import importlib.util as iu

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
WD = f"{B}/_tmp/fixtest"; shutil.rmtree(WD, ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/submission_MAXSOUP_head.zip") as z:
    z.extractall(WD)

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
wins = []
for i in sorted(set(range(0, len(lens), 5))):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20): wins.append(t0)
wins = wins[:64]
W = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1)
              for s in wins]).astype(np.float32)
Xin = W[:, :20]

def load_mod(path, tag):
    s = iu.spec_from_file_location(tag, path); m = iu.module_from_spec(s); s.loader.exec_module(m)
    return m

print("=" * 70); print("1. EQUIVALENCE -- fixed vs shipped, same inputs"); print("=" * 70)
orig = load_mod(os.path.join(WD, "submission.py"), "orig")
shutil.copy("/tmp/fixed_submission.py", os.path.join(WD, "submission_fixed.py"))
fixed = load_mod(os.path.join(WD, "submission_fixed.py"), "fixed")

r_o = orig.predict(Xin, metadata={})
r_f = fixed.predict(Xin, metadata={})
allok = True
for k in ("prediction", "lower", "upper"):
    same = np.array_equal(r_o[k], r_f[k])
    mx = float(np.abs(r_o[k].astype(np.float64) - r_f[k].astype(np.float64)).max())
    print("  %-11s bit-identical=%-5s  max|diff|=%.3e" % (k, same, mx))
    allok &= same
print("  EQUIVALENCE:", "PASS" if allok else "FAIL")

print()
print("=" * 70); print("2. CPU PATH -- the case that actually failed"); print("=" * 70)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
cpu = load_mod(os.path.join(WD, "submission_fixed.py"), "cpu_fixed")
import torch
print("  torch.cuda.is_available():", torch.cuda.is_available())
t_load0 = time.time()
_m, _dev = cpu._get_model()
t_load = time.time() - t_load0
print("  device selected: %s   model load: %.1fs" % (_dev, t_load))

n_bench = 16
t0 = time.time(); r_cpu = cpu.predict(Xin[:n_bench], metadata={}); t_cpu = time.time() - t0
per = t_cpu / n_bench
print("  throughput: %.1f ms/sample  (%d windows in %.1fs)" % (per * 1000, n_bench, t_cpu))
print("  finite=%s  lo<=up=%s" % (bool(np.isfinite(r_cpu["prediction"]).all()),
                                  bool((r_cpu["lower"] <= r_cpu["upper"]).all())))
for N in (2000, 5140):
    est = t_load + per * N
    print("  N=%-5d -> load %.0fs + infer %.0fs = %.0fs total vs 180s limit  %s"
          % (N, t_load, per * N, est, "OK" if est < 180 else "WOULD BE KILLED -> budget must trip"))

print()
print("=" * 70); print("3. BUDGET SEMANTICS"); print("=" * 70)
print("  _HARD_LIMIT=%.0f  _RESERVE=%.0f  _CALL_CAP=%.0f" % (cpu._HARD_LIMIT, cpu._RESERVE, cpu._CALL_CAP))
print("  global deadline is %.0fs after MODULE IMPORT (load is charged against it)"
      % (cpu._HARD_LIMIT - cpu._RESERVE))
# force the global deadline into the past -> every chunk must bail immediately
saved = cpu._T_START
cpu._T_START = time.time() - 1e6
t0 = time.time(); r_trip = cpu.predict(Xin[:32], metadata={}); t_trip = time.time() - t0
cpu._T_START = saved
print("  global-deadline-exceeded call returned in %.2fs (should be fast: pure persistence)" % t_trip)
print("  finite=%s  lo<=up=%s  shapes=%s" % (
    bool(np.isfinite(r_trip["prediction"]).all()),
    bool((r_trip["lower"] <= r_trip["upper"]).all()),
    r_trip["prediction"].shape == Xin[:32].shape))
hw = (r_trip["upper"] - r_trip["lower"]) / 2
print("  fell back to constant band (h_u==0.0129): %s" % bool(np.allclose(hw[..., 0], 0.0129, atol=1e-6)))
