"""GPU-path equivalence + full-scale CPU survival test for the fixed submission.

(1) On GPU the fixed version must return bit-identical prediction/lower/upper to the
    shipped one -- the fix is timing/memory only and must not touch model behaviour.
(2) On CPU at a realistic N, the whole call must finish inside the 180s wall with a
    complete, valid result. This is the scenario that killed the real submission.
"""
import os, sys, zipfile, shutil, time, json
import numpy as np
import importlib.util as iu

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
MODE = sys.argv[1]                      # "gpu" or "cpu"
WD = f"{B}/_tmp/equiv_{MODE}"; shutil.rmtree(WD, ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/submission_MAXSOUP_head.zip") as z:
    z.extractall(WD)
shutil.copy("/tmp/fixed_submission.py", os.path.join(WD, "submission_fixed.py"))

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
wins = []
for i in sorted(set(range(0, len(lens), 5))):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20): wins.append(t0)

def make(nw):
    w = wins[:nw]
    W = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1)
                  for s in w]).astype(np.float32)
    return W[:, :20]

def load_mod(fn, tag):
    s = iu.spec_from_file_location(tag, os.path.join(WD, fn)); m = iu.module_from_spec(s)
    s.loader.exec_module(m); return m

if MODE == "gpu":
    import torch
    assert torch.cuda.is_available(), "need a GPU for this half"
    Xin = make(200)
    o = load_mod("submission.py", "o"); f = load_mod("submission_fixed.py", "f")
    ro = o.predict(Xin, metadata={}); rf = f.predict(Xin, metadata={})
    ok = True
    for k in ("prediction", "lower", "upper"):
        same = np.array_equal(ro[k], rf[k]); ok &= same
        print("  %-11s bit-identical=%s  max|diff|=%.3e"
              % (k, same, float(np.abs(ro[k].astype(np.float64) - rf[k].astype(np.float64)).max())))
    print("  GPU EQUIVALENCE:", "PASS" if ok else "FAIL")
    t0 = time.time(); f.predict(Xin, metadata={}); dt = (time.time() - t0) / len(Xin)
    print("  fixed GPU throughput: %.2f ms/sample -> N=5140 in %.0fs" % (dt * 1000, dt * 5140))
else:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "set CUDA_VISIBLE_DEVICES=''"
    import torch
    assert not torch.cuda.is_available()
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    Xin = make(N)
    print("  CPU full-call test at N=%d (import-anchored deadline is live)" % len(Xin))
    f = load_mod("submission_fixed.py", "fcpu")
    t0 = time.time(); r = f.predict(Xin, metadata={}); dt = time.time() - t0
    since_import = time.time() - f._T_START
    print("  predict() returned in %.1fs; %.1fs since module import (limit 180s)" % (dt, since_import))
    print("  finite=%s lo<=up=%s shape_ok=%s"
          % (bool(np.isfinite(r["prediction"]).all()),
             bool((r["lower"] <= r["upper"]).all()),
             r["prediction"].shape == Xin.shape))
    hw = (r["upper"] - r["lower"]) / 2
    frac_const = float(np.isclose(hw[..., 0], 0.0129, atol=1e-6).mean())
    print("  fraction of windows left on the constant band: %.2f (rest got the learned head)" % frac_const)
    print("  VERDICT:", "SURVIVES the 180s wall" if since_import < 180 else "STILL OVER -- more headroom needed")
