"""Compare the WORKING SOUP_v1 checkpoint against the FAILING new soup across the
FULL input distribution (all 81 trajectories, not the 200-window slice the gates used).

BISECT_CKPT is SOUP_v1's exact archive with only the checkpoint swapped, and it failed.
So if the platform is healthy the fault is in the weights. Structure, dtypes and
finiteness are already verified identical, which leaves numerical behaviour: a model
that emits NaN/Inf or absurd magnitudes on some input would make the scorer blow up,
which the platform reports as 'Failed' rather than as a bad score.
"""
import os, sys, zipfile, shutil, json
import numpy as np
import importlib.util as iu

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta.get("names", [])
wins, wtraj = [], []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wtraj.append(i)
wtraj = np.array(wtraj)
print("full sweep: %d windows over %d trajectories" % (len(wins), len(lens)))
W = np.stack([np.concatenate([np.asarray(X[s:s + 40]), np.zeros((40, 32, 64, 1), np.float32)], -1)
              for s in wins]).astype(np.float32)
Xin, Y = W[:, :20], W[:, 20:]

def run(zipname, tag):
    wd = f"{B}/_tmp/numstab_{tag}"; shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    with zipfile.ZipFile(f"{B}/submissions/{zipname}") as z: z.extractall(wd)
    s = iu.spec_from_file_location(tag, os.path.join(wd, "submission.py"))
    m = iu.module_from_spec(s); s.loader.exec_module(m)
    out = []
    CH = 256
    for i in range(0, len(Xin), CH):
        r = m.predict(Xin[i:i + CH], metadata={})
        out.append((r["prediction"], r["lower"], r["upper"]))
    P = np.concatenate([o[0] for o in out], 0)
    L = np.concatenate([o[1] for o in out], 0)
    U = np.concatenate([o[2] for o in out], 0)
    return P, L, U

for zipname, tag in (("submission_SOUP_v1.zip", "v1"), ("submission_BISECT_CKPT.zip", "new")):
    P, L, U = run(zipname, tag)
    finite = np.isfinite(P).all() and np.isfinite(L).all() and np.isfinite(U).all()
    print("\n=== %s ===" % zipname)
    print("  all finite            : %s" % finite)
    print("  lower <= upper        : %s" % bool((L <= U).all()))
    print("  prediction[...,2]==0  : %s" % bool((P[..., 2] == 0).all()))
    print("  |pred| max %.4f   mean %.5f   std %.5f" % (
        float(np.abs(P[..., :2]).max()), float(P[..., :2].mean()), float(P[..., :2].std())))
    print("  target |Y| max %.4f (for scale reference)" % float(np.abs(Y[..., :2]).max()))
    hw = (U - L) / 2
    print("  half-width u: min %.6f  max %.6f  any<=0: %s" % (
        float(hw[..., 0].min()), float(hw[..., 0].max()), bool((hw[..., 0] <= 0).any())))
    print("  half-width v: min %.6f  max %.6f" % (float(hw[..., 1].min()), float(hw[..., 1].max())))
    # per-trajectory worst case -- an instability may hit only some flow regimes
    worst = []
    for i in range(len(lens)):
        m_ = wtraj == i
        if not m_.any(): continue
        mx = float(np.abs(P[m_][..., :2]).max())
        worst.append((mx, i))
    worst.sort(reverse=True)
    print("  worst 5 trajectories by |pred| max: %s" % (
        ", ".join("traj%d=%.3f" % (i, v) for v, i in worst[:5])))
    if not finite:
        badw = np.where(~np.isfinite(P).all(axis=(1, 2, 3, 4)))[0]
        print("  !! non-finite in %d windows, first traj ids: %s"
              % (len(badw), sorted(set(wtraj[badw].tolist()))[:10]))
    np.save(f"{B}/_tmp/numstab_{tag}_stats.npy",
            np.array([float(np.abs(P[..., :2]).max()), float(P[..., :2].std())]))
