"""Peak-memory test at the real eval scale. Never measured before.

At N=5140 the API forces four (N,20,32,64,3) float32 arrays live at once -- the
caller's input, prediction, lower, upper -- about 10 GB before counting torch.
If the eval container's memory limit is below that, the process is killed and the
platform reports 'Failed' with no traceback, exactly what we are seeing, and it
would be sensitive to host conditions (which fits: an identical-structure zip
scored fine on a previous day).
"""
import os, sys, zipfile, shutil, time, json, resource
import numpy as np
import importlib.util as iu

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5140
ZIP = sys.argv[2] if len(sys.argv) > 2 else "submission_MAXSOUP_v2.zip"
WD = f"{B}/_tmp/memtest"; shutil.rmtree(WD, ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/{ZIP}") as z: z.extractall(WD)

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)  # linux: KB

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
wins = []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20): wins.append(t0)
print("available distinct windows: %d; requested N=%d" % (len(wins), N))
reps = (N // len(wins)) + 1
wins = (wins * reps)[:N]

print("building input array (%.2f GB)..." % (N * 20 * 32 * 64 * 3 * 4 / 1e9))
Xin = np.empty((N, 20, 32, 64, 3), dtype=np.float32)
for j, s in enumerate(wins):
    Xin[j, ..., :2] = np.asarray(X[s:s + 20])
    Xin[j, ..., 2] = 0.0
print("  after input alloc: peak RSS %.2f GB" % rss_gb())

s = iu.spec_from_file_location("m", os.path.join(WD, "submission.py"))
m = iu.module_from_spec(s); s.loader.exec_module(m)
t0 = time.time()
r = m.predict(Xin, metadata={})
dt = time.time() - t0
print("  predict() done in %.1fs" % dt)
print("  PEAK RSS: %.2f GB" % rss_gb())
print("  output ok: finite=%s lo<=up=%s shape=%s" % (
    bool(np.isfinite(r["prediction"]).all()),
    bool((r["lower"] <= r["upper"]).all()),
    r["prediction"].shape == Xin.shape))
tot = sum(v.nbytes for v in r.values()) / 1e9
print("  returned arrays: %.2f GB  (+ %.2f GB caller input still live)" % (tot, Xin.nbytes / 1e9))
print("  => a container capped below ~%.0f GB would OOM-kill this" % (rss_gb() + 0.5))
