"""Run every submitted zip's OWN predict() on the same held-out windows, then ask the
corrected calibration to predict its real sps. Compare against what it actually scored.

This is an out-of-sample test for every zip except the four that were used to FIT the
original-family calibration (fno / calibrated / plain_sps / ROBUST) -- those are marked
IN-SAMPLE. FULLSTACK_v6, FULLSTACK_v7 and SOUP_v1/WIDE125 were never used in any fit.
"""
import importlib.util, json, os, shutil, subprocess, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; C = 2
from scipy.optimize import brentq
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
inv = lambda t: brentq(lambda e: S.score_error(e) - t, 1e-9, 50.0)
n_ = lambda x: x/(0.5+x)
def Wof(dm, tke, mv):
    return 0.5*(1-n_(inv(dm))) + 0.3*(1-n_(inv(tke))) + 0.2*(1-n_(inv(mv)))

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off, lens = meta["off"], meta["lens"]
ntraj = len(lens); held = sorted(set(range(0, ntraj, 5)))
starts = []
for i in held:
    starts += list(range(off[i], off[i]+lens[i]-39, 20))
rng = np.random.default_rng(0); starts = rng.choice(starts, size=min(400, len(starts)), replace=False)
Wd = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40,32,64,1), np.float32)], -1)
               for s in starts]).astype(np.float32)
Xin, Y = Wd[:, :20], Wd[:, 20:]
SCM = (Y[..., :C] != 0.0)
print("held-out probe windows:", Xin.shape[0], flush=True)

# calibrations: (a_u, a_v) per model family
CAL_ORIG = (1.550, 3.200)      # sec23.1, fitted on the ORIGINAL's four real anchors
CAL_SOUP = (1.579, 2.770)      # sec24.2, ratio 0.570 + level from the soup's anchors

SUBS = [
    # zip,                          real subscores (rel_l2,tke,mvpe), real sps, cal,  insample
    ("submission_fno.zip",            (94.17,74.03,92.84), 14.08, CAL_ORIG, True),
    ("submission_fno_calibrated.zip", (94.17,74.03,92.84), 18.30, CAL_ORIG, True),
    ("submission_fno_plain_sps.zip",  (94.17,74.03,92.84), 29.84, CAL_ORIG, True),
    ("submission_ROBUST.zip",         (94.17,74.03,92.84), 33.08, CAL_ORIG, True),
    ("submission_FULLSTACK_v6.zip",   (93.85,74.97,92.84), 28.38, CAL_ORIG, False),
    ("submission_FULLSTACK_v7.zip",   (93.85,74.97,92.84), 33.41, CAL_ORIG, False),
    ("submission_SOUP_v1.zip",        (94.05,76.00,92.87), 34.37, CAL_SOUP, False),
    ("submission_LUTFIX.zip",         (94.05,76.00,92.87), None,  CAL_SOUP, False),
]

def run(zipname, tag):
    d = f"{B}/_tmp/cmp_{tag}"
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d); subprocess.run(["unzip","-q",f"{B}/submissions/{zipname}","-d",d], check=True)
    sys.path.insert(0, d)
    for m in list(sys.modules):
        if m in ("submission","load_baseline") or m.startswith("rpde_baselines") or m.startswith("einops"):
            del sys.modules[m]
    sp = importlib.util.spec_from_file_location(f"s_{tag}", f"{d}/submission.py")
    M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    out = M.predict(Xin)
    sys.path.remove(d); shutil.rmtree(d)
    return out

print("\n%-32s %8s %8s %9s %9s %8s" % ("zip","E_local","pred sps","actual","error","sample"))
print("-"*82)
for zipname, sub, real_sps, cal, insample in SUBS:
    try:
        out = run(zipname, zipname.replace(".zip","")[-8:])
    except Exception as e:
        print("%-32s  FAILED: %s" % (zipname, str(e)[:40])); continue
    if isinstance(out, dict) and "prediction" in out:
        P = np.asarray(out["prediction"], np.float32)
        if "upper" in out and out["upper"] is not None:
            h = (np.asarray(out["upper"], np.float32) - np.asarray(out["lower"], np.float32))/2.0
        else:
            h = 0.05*np.abs(P)          # scorer default band
    else:
        P = np.asarray(out, np.float32)
        h = 0.05*np.abs(P)              # no bounds returned -> scorer default band
    ERR = np.abs(P[..., :C] - Y[..., :C])
    au, av = cal
    num = 0.0; tot = 0
    for ci, a in ((0, au), (1, av)):
        m = SCM[..., ci]
        e = ERR[..., ci][m]*a; hh = h[..., ci][m]
        num += float((np.exp(-2*hh/SIG)*(e <= hh)).sum()); tot += e.size
    E = num/tot
    Wv = Wof(*sub); pred = 100*Wv*E
    if real_sps is None:
        print("%-32s %8.4f %8.2f %9s %9s %8s" % (zipname, E, pred, "--", "--", "CANDIDATE"))
    else:
        print("%-32s %8.4f %8.2f %9.2f %+9.2f %8s"
              % (zipname, E, pred, real_sps, pred-real_sps, "in" if insample else "OUT"))
