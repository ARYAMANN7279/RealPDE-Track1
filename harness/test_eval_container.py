"""Run a submission zip under the ACTUAL evaluation container's package set
(python 3.10 / torch 2.2.2 / numpy 1.26), with scipy/h5py/sklearn/pandas blocked
exactly as the kit's smoke test does.

Every previous gate ran under torch 2.12 on the VM. The validator warned about
that mismatch on every single run and it was dismissed as noise. This is the
first test in the real target environment.
"""
import sys, os, zipfile, shutil, time, json, traceback
import importlib.util as iu

# --- block the packages the eval image does not have, like smoke_test_kit does
class _Blocker:
    BLOCKED = {"scipy", "h5py", "sklearn", "pandas", "matplotlib", "cv2", "einops"}
    def find_module(self, name, path=None):
        root = name.split(".")[0]
        if root in self.BLOCKED and path is None:
            return self
        return None
    def load_module(self, name):
        raise ImportError("%s is not available in the evaluation container" % name)
sys.meta_path.insert(0, _Blocker())

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
ZIP = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

import numpy as np
import torch
print("env: python %s | torch %s | numpy %s" % (sys.version.split()[0], torch.__version__, np.__version__))
print("zip: %s" % ZIP)

WD = "%s/_tmp/evalctr_%s" % (B, os.path.basename(ZIP).replace(".zip", ""))
shutil.rmtree(WD, ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile("%s/submissions/%s" % (B, ZIP)) as z:
    z.extractall(WD)

ok = True

# 1. import the submission module exactly as the harness would
print("\n[1] import submission.py")
t0 = time.time()
try:
    sys.path.insert(0, WD)
    spec = iu.spec_from_file_location("submission", os.path.join(WD, "submission.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    print("    OK in %.1fs" % (time.time() - t0))
except Exception:
    ok = False
    print("    FAILED:"); traceback.print_exc()
    sys.exit(1)

# 2. load the checkpoint (the step most likely to break across torch versions)
print("\n[2] load checkpoint under torch 2.2.2")
t0 = time.time()
try:
    mdl, dev = m._get_model()
    print("    OK in %.1fs  device=%s" % (time.time() - t0, dev))
except Exception:
    ok = False
    print("    FAILED:"); traceback.print_exc()
    sys.exit(1)

# 3. real forward pass on real data
print("\n[3] predict() on %d real windows" % N)
X = np.load("%s/local_harness/tr_frames.npy" % B, mmap_mode="r")
meta = json.load(open("%s/local_harness/tr_meta.json" % B))
off, lens = meta["off"], meta["lens"]
wins = []
for i in sorted(set(range(0, len(lens), 5))):
    for t0_ in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0_)
wins = wins[:N]
W = np.stack([np.concatenate([np.asarray(X[s:s + 40]),
              np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin = W[:, :20]
t0 = time.time()
try:
    r = m.predict(Xin, metadata={})
    dt = time.time() - t0
    print("    OK in %.1fs (%.0f ms/sample)" % (dt, dt / len(Xin) * 1000))
except Exception:
    ok = False
    print("    FAILED:"); traceback.print_exc()
    sys.exit(1)

# 4. the contract the README spells out
print("\n[4] output contract")
p = r["prediction"]; lo = r.get("lower"); up = r.get("upper")
checks = [
    ("prediction shape == input shape", p.shape == Xin.shape),
    ("prediction finite", bool(np.isfinite(p).all())),
    ("has lower AND upper", lo is not None and up is not None),
]
if lo is not None and up is not None:
    checks += [
        ("lower.shape == prediction.shape", lo.shape == p.shape),
        ("upper.shape == prediction.shape", up.shape == p.shape),
        ("bounds finite", bool(np.isfinite(lo).all() and np.isfinite(up).all())),
        ("lower <= upper everywhere", bool((lo <= up).all())),
        ("dtypes float32", p.dtype == np.float32 and lo.dtype == np.float32),
    ]
for name, val in checks:
    print("    %-34s %s" % (name, "ok" if val else "*** FAIL ***"))
    ok &= bool(val)

# 5. second call in the same process -- 'predict may be called more than once and
#    every call must make the same choice about returning bounds'
print("\n[5] second call returns bounds consistently")
try:
    r2 = m.predict(Xin[:4], metadata={})
    same = (("lower" in r) == ("lower" in r2)) and (("upper" in r) == ("upper" in r2))
    print("    same bounds choice across calls: %s" % ("ok" if same else "*** FAIL ***"))
    ok &= same
except Exception:
    ok = False
    print("    FAILED:"); traceback.print_exc()

print("\nRESULT: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 2)
