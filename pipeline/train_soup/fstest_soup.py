"""GATE 2 + GATE 3 for the soup zip.

GATE 3 requires a RATIO to the bare FNO, not absolute ms: the VM is shared and the
same zip has measured 2.69 and 4.57 ms/sample minutes apart. Both zips are therefore
loaded in ONE process and timed INTERLEAVED (A,B,A,B,...), so drift hits both equally.
The median ratio is converted with the validated anchor: bare FNO = 6.59 ms/sample on
the A800, time_score = 100/(1+sqrt(t/0.72896)).

usage: fstest_soup.py <candidate.zip> <reference.zip>
"""
import os, sys, zipfile, shutil, time, json
import numpy as np
import importlib.util as iu

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S); C = 2

CAND = sys.argv[1]
REF  = sys.argv[2] if len(sys.argv) > 2 else f"{B}/submissions/submission_ROBUST.zip"

def load_zip(path, tag):
    wd = f"{B}/_tmp/{tag}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    with zipfile.ZipFile(path) as z: z.extractall(wd)
    sys.path.insert(0, wd)
    s2 = iu.spec_from_file_location("sub_" + tag, os.path.join(wd, "submission.py"))
    m = iu.module_from_spec(s2); s2.loader.exec_module(m)
    return m, wd

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
vidx = sorted(set(range(0, len(lens), 5)))
wins = []
for i in vidx:
    for t0 in range(off[i], off[i]+lens[i]-39, 20): wins.append(t0)
wins = wins[:200]
W_ = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40,32,64,1), np.float32)], -1)
               for s in wins]).astype(np.float32)
Xin, Y = W_[:, :20], W_[:, 20:]

mc, wdc = load_zip(CAND, "cand")
mr, wdr = load_zip(REF,  "ref")

print("=" * 68)
print("GATE 2 -- end-to-end behaviour of the ACTUAL zip")
print("=" * 68)
print("candidate:", os.path.basename(CAND))
r = mc.predict(Xin, metadata={})
p, lo, up = r["prediction"], r["lower"], r["upper"]
shapes_ok = (p.shape == Xin.shape and lo.shape == p.shape and up.shape == p.shape)
print("  shapes (N,20,32,64,3)      : %s  %s" % (shapes_ok, p.shape))
print("  all finite                 : %s" % bool(np.isfinite(p).all() and
                                                 np.isfinite(lo).all() and np.isfinite(up).all()))
print("  lower <= upper everywhere  : %s" % bool((lo <= up).all()))
print("  prediction[...,2] == 0     : %s" % bool((p[..., 2] == 0).all()))
hw = (up - lo) / 2
print("  bounds vary per element    : %s (std %.2e)" % (bool(hw[..., 0].std() > 1e-5), hw[..., 0].std()))
print("  h_u median %.5f  range %.5f .. %.5f" % (float(np.median(hw[..., 0])),
                                                 float(hw[..., 0].min()), float(hw[..., 0].max())))
print("  h_v median %.5f  range %.5f .. %.5f" % (float(np.median(hw[..., 1])),
                                                 float(hw[..., 1].min()), float(hw[..., 1].max())))
dm = S.rel_l2_per_sample(p, Y, C); tk = S.tke_rel_l2_per_sample(p, Y, C); mv = S.mvpe_rel_l2_per_sample(p, Y)
acc_c = (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))
rr = mr.predict(Xin, metadata={})
pr = rr["prediction"]
dm = S.rel_l2_per_sample(pr, Y, C); tk = S.tke_rel_l2_per_sample(pr, Y, C); mv = S.mvpe_rel_l2_per_sample(pr, Y)
acc_r = (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))
print("\n  local-scale accuracy   rel_l2    tke   mvpe")
print("    reference (banked)  %7.2f %6.2f %6.2f" % acc_r)
print("    candidate (soup)    %7.2f %6.2f %6.2f" % acc_c)
print("    delta               %+7.2f %+6.2f %+6.2f" % tuple(acc_c[i]-acc_r[i] for i in range(3)))

print()
print("=" * 68)
print("GATE 3 -- time budget (interleaved ratio, not absolute ms)")
print("=" * 68)
NS = 128
xs = Xin[:NS]
mc.predict(xs, metadata={}); mr.predict(xs, metadata={})       # warm both
tc, tr_ = [], []
for _ in range(5):
    t = time.time(); mc.predict(xs, metadata={}); tc.append((time.time()-t)/NS)
    t = time.time(); mr.predict(xs, metadata={}); tr_.append((time.time()-t)/NS)
mc_ms = float(np.median(tc))*1000; mr_ms = float(np.median(tr_))*1000
ratio = mc_ms / mr_ms
ANCHOR = 6.59            # bare FNO ms/sample on the A800 (validated: predicted 6.57)
est = ANCHOR * ratio
tscore = lambda t: 100/(1+np.sqrt((t/1000)/0.72896))
print("  candidate  %.2f ms/sample   (5 runs, median)" % mc_ms)
print("  reference  %.2f ms/sample" % mr_ms)
print("  RATIO      %.3f  -> A800 est %.2f ms/sample" % (ratio, est))
print("  time subscore: reference %.2f (real 91.36) | candidate %.2f  -> delta %+.2f final"
      % (tscore(ANCHOR), tscore(est), 0.100*(tscore(est)-tscore(ANCHOR))))
print("  N=5140 windows would take %.0fs on the A800 (limit 180s)" % (est/1000*5140))

print()
print("  -- fallback paths --")
sav = mc._TIME_BUDGET
mc._TIME_BUDGET = -1.0
r2 = mc.predict(Xin[:16], metadata={})
h2 = (r2["upper"]-r2["lower"])/2
print("  full trip   : finite %s  lo<=up %s  h_u const %s"
      % (bool(np.isfinite(r2["prediction"]).all()), bool((r2["lower"] <= r2["upper"]).all()),
         bool(h2[..., 0].std() < 1e-9)))
mc._TIME_BUDGET = 0.02
r3 = mc.predict(Xin[:64], metadata={})
print("  partial trip: finite %s  lo<=up %s"
      % (bool(np.isfinite(r3["prediction"]).all()), bool((r3["lower"] <= r3["upper"]).all())))
mc._TIME_BUDGET = sav
json.dump({"acc_cand": acc_c, "acc_ref": acc_r, "ratio": ratio, "a800_ms": est,
           "tscore_cand": float(tscore(est)), "tscore_ref": float(tscore(ANCHOR))},
          open(f"{B}/train_work_soup/fstest.json", "w"), indent=1)
print("\nwrote fstest.json")
