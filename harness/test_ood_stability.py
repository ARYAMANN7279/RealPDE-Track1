"""Last untested hypothesis: the 9-member soup is numerically fragile on inputs
outside our local train_real distribution.

Every local test so far fed the models train_real windows. The real eval uses
UNSEEN Reynolds numbers / angles of attack. If the new checkpoint produces
NaN/Inf or wildly extreme values there, the run can die or score zero while
looking perfect locally.

Compares v1's soup (proven, scored 78.4497) against the new soup on:
  * real windows                (control)
  * scaled/shifted windows      (proxy for a different Reynolds regime)
  * extreme-magnitude inputs    (stress)
  * degenerate inputs           (zeros, constant, single hot pixel)
Runs under the eval container's torch 2.2.2.
"""
import sys, os, json, zipfile, shutil
import numpy as np

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import torch
from load_baseline import load_baseline
print("torch", torch.__version__)

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)

def get_model(zipname):
    wd = f"{B}/_tmp/ood_{zipname.replace('.zip','')}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    with zipfile.ZipFile(f"{B}/submissions/{zipname}") as z:
        z.extract("sim_real_fno_fp16.pth", wd)
    m, _ = load_baseline(os.path.join(wd, "sim_real_fno_fp16.pth"), device="cpu")
    return m.eval()

def run(m, x):
    mi, si = torch.from_numpy(MI), torch.from_numpy(SI)
    mt, st = torch.from_numpy(MT), torch.from_numpy(ST)
    with torch.no_grad():
        xb = torch.from_numpy(np.ascontiguousarray(x))
        return (m((xb - mi) / si) * st + mt).numpy()

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off, lens = meta["off"], meta["lens"]
w = []
for i in sorted(set(range(0, len(lens), 5))):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20): w.append(t0)
w = w[:4]
base = np.stack([np.concatenate([np.asarray(X[s:s + 40]),
                 np.zeros((40, 32, 64, 1), np.float32)], -1) for s in w]).astype(np.float32)[:, :20]

cases = {}
cases["real (control)"] = base.copy()
for f in (1.5, 3.0, 10.0):
    c = base.copy(); c[..., :2] *= f; cases["scaled x%.1f" % f] = c
c = base.copy(); c[..., 0] += 0.5; cases["u shifted +0.5"] = c
cases["zeros"] = np.zeros_like(base)
c = np.zeros_like(base); c[..., 0] = 0.155; cases["constant u=0.155"] = c
c = np.zeros_like(base); c[:, :, 16, 32, 0] = 100.0; cases["single hot pixel 100"] = c
c = base.copy(); c[..., :2] *= 1e3; cases["scaled x1000"] = c

models = {"v1 soup (PROVEN)": "submission_SOUP_v1.zip",
          "new soup (FAILING)": "submission_BISECT_CKPT.zip"}
loaded = {k: get_model(v) for k, v in models.items()}

print("\n%-22s | %-20s | %-10s %-10s %-9s %s" % ("case", "model", "min", "max", "absmax", "finite"))
print("-" * 96)
bad_any = False
for cname, x in cases.items():
    for mname, mdl in loaded.items():
        y = run(mdl, x)
        fin = bool(np.isfinite(y).all())
        flag = "" if fin else "   <<< NON-FINITE"
        if not fin: bad_any = True
        print("%-22s | %-20s | %-10.4g %-10.4g %-9.4g %s%s"
              % (cname, mname, y.min(), y.max(), np.abs(y).max(), fin, flag))
    print()

print("any non-finite output:", bad_any)

# direct divergence between the two models on the control case
ya = run(loaded["v1 soup (PROVEN)"], base)
yb = run(loaded["new soup (FAILING)"], base)
print("control-case max|v1 - new| = %.6g   (models should differ slightly, not wildly)"
      % float(np.abs(ya - yb).max()))
