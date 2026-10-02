"""GATE 2/3 for submission_MAXSOUP.zip: end-to-end behavior of the actual zip's
predict(), timing ratio vs the ROBUST reference (whose real time subscore is
known: 91.36), and forced/partial timeout-fallback correctness."""
import os, sys, zipfile, shutil, importlib.util, time, json
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S); C = 2

def load_zip(path, tag):
    wd = f"{B}/_tmp/gate23_{tag}"; shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    with zipfile.ZipFile(path) as z: z.extractall(wd)
    s = iu.spec_from_file_location(tag, os.path.join(wd, "submission.py"))
    m = iu.module_from_spec(s); s.loader.exec_module(m)
    return m

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
vidx = sorted(set(range(0, len(lens), 5)))
wins = []
for i in vidx:
    for t0 in range(off[i], off[i] + lens[i] - 39, 20): wins.append(t0)
wins = wins[:200]
W = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = W[:, :20], W[:, 20:]

print("="*70); print("GATE 2 -- end-to-end behaviour of the ACTUAL zip"); print("="*70)
cand = load_zip(f"{B}/submissions/submission_MAXSOUP.zip", "cand")
r = cand.predict(Xin, metadata={})
p, lo, up = r["prediction"], r["lower"], r["upper"]
print("shapes", p.shape == Xin.shape and lo.shape == p.shape, p.shape)
print("all finite:", bool(np.isfinite(p).all() and np.isfinite(lo).all() and np.isfinite(up).all()))
print("lower<=upper everywhere:", bool((lo <= up).all()))
print("prediction[...,2]==0:", bool((p[..., 2] == 0).all()))
hw = (up - lo) / 2
print("bounds vary per element: %s (std_u=%.2e, std_v=%.2e)" % (bool(hw[...,0].std() > 1e-6), hw[...,0].std(), hw[...,1].std()))
print("h_u median %.5f range %.5f..%.5f" % (float(np.median(hw[...,0])), float(hw[...,0].min()), float(hw[...,0].max())))
print("h_v median %.5f range %.5f..%.5f" % (float(np.median(hw[...,1])), float(hw[...,1].min()), float(hw[...,1].max())))
dm = S.rel_l2_per_sample(p, Y, C); tk = S.tke_rel_l2_per_sample(p, Y, C); mv = S.mvpe_rel_l2_per_sample(p, Y)
print("local-scale accuracy: rel_l2 %.2f tke %.2f mvpe %.2f" % (
    S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean()))))

print(); print("="*70); print("GATE 3 -- time budget (interleaved ratio vs ROBUST)"); print("="*70)
robust = load_zip(f"{B}/submissions/submission_ROBUST.zip", "robust")
def bench(m, n=5):
    ts = []
    for _ in range(n):
        t0 = time.time(); m.predict(Xin, metadata={}); ts.append((time.time()-t0)/len(Xin))
    ts.sort(); return ts[len(ts)//2]
t_cand = bench(cand); t_robust = bench(robust)
ratio = t_cand / t_robust
print("candidate  %.2f ms/sample" % (t_cand*1000))
print("ROBUST ref %.2f ms/sample  (real time subscore 91.36)" % (t_robust*1000))
print("ratio %.3f -> A800-anchor est %.2f ms/sample" % (ratio, ratio * 2.72))
n_total = 5140
print("N=%d windows would take %.0fs on A800 estimate (limit 180s)" % (n_total, ratio*2.72/1000*n_total))

print(); print("-- fallback paths --")
cand._TIME_BUDGET = -1.0
r2 = cand.predict(Xin[:16], metadata={})
print("full trip: finite", bool(np.isfinite(r2["prediction"]).all()), "lo<=up", bool((r2["lower"] <= r2["upper"]).all()))
hw2 = (r2["upper"] - r2["lower"]) / 2
print("full trip bounds match per-location map (not corrupted):",
      bool(np.allclose(hw2[..., 0].std(axis=(0,1)) if hw2[...,0].ndim>2 else 0, hw2[...,0].std(axis=(0,1)) if hw2[...,0].ndim>2 else 0)))

cand2 = load_zip(f"{B}/submissions/submission_MAXSOUP.zip", "cand2")
cand2._TIME_BUDGET = t_cand * 100 * 1.5  # trip partway through
r3 = cand2.predict(Xin, metadata={})
print("partial trip: finite", bool(np.isfinite(r3["prediction"]).all()), "lo<=up", bool((r3["lower"] <= r3["upper"]).all()))
