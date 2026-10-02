"""GATE 6 check: does the freshly rebuilt train/run_all.sh artifact (train_work/submission.zip,
built AFTER fixing the missing `global_scale` import) score within noise of the shipped
submission_FULLSTACK_v7.zip on held-out trajectories?

Compares on LOCAL-scale E directly (exp(-2h/sigma)*1[|P-Y|<=h]) -- no leaderboard alpha/beta
transform involved, so this is valid for a per-element policy per GATE 4B.
"""
import json, os, sys, zipfile, shutil
import importlib.util as iu
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
SIG = S.SIGMA_GLOBAL; C = 2; IN = 20

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
vidx = sorted(set(range(0, len(lens), 5)))
wins = []; wt = []
for i in vidx:
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wt = np.array(wt)
W_ = np.stack([np.concatenate([np.asarray(X[s:s + 40]), np.zeros((40, 32, 64, 1), np.float32)], -1)
               for s in wins]).astype(np.float32)
Xin, Y = W_[:, :IN], W_[:, IN:]
SCM = (Y[..., :C] != 0.0)

def run_zip(path, tag):
    WD = f"{B}/_tmp/g6_{tag}"; shutil.rmtree(WD, ignore_errors=True); os.makedirs(WD)
    with zipfile.ZipFile(path) as z: z.extractall(WD)
    sys.path.insert(0, WD)
    s2 = iu.spec_from_file_location("z6_" + tag, os.path.join(WD, "submission.py"))
    m = iu.module_from_spec(s2); s2.loader.exec_module(m)
    P = []; L = []; U = []
    for i in range(0, len(Xin), 48):
        r = m.predict(Xin[i:i + 48], metadata={})
        P.append(r["prediction"]); L.append(r["lower"]); U.append(r["upper"])
    return (np.concatenate(P, 0).astype(np.float32), np.concatenate(L, 0).astype(np.float32),
            np.concatenate(U, 0).astype(np.float32))

def report(name, P, L, U):
    dm = S.rel_l2_per_sample(P, Y, C); tk = S.tke_rel_l2_per_sample(P, Y, C); mv = S.mvpe_rel_l2_per_sample(P, Y)
    h = ((U - L) / 2.0)[..., :C]
    err = np.abs(P[..., :C] - Y[..., :C])
    ok = (err <= h) & SCM
    E = float((np.exp(-2 * h / SIG) * ok).sum() / SCM.sum())
    cov = float(ok.sum() / SCM.sum())
    hu = h[..., 0]
    print("%-16s rel_l2 %.2f tke %.2f mvpe %.2f | local-E %.4f cov %.3f | h_u med %.5f std(log h_u) %.3f" % (
        name, S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())),
        E, cov, float(np.median(hu)), float(np.std(np.log(hu + 1e-9)))))
    return E

print("windows:", len(Xin))
P_ship, L_ship, U_ship = run_zip(f"{B}/submissions/submission_FULLSTACK_v7.zip", "shipped")
E_ship = report("SHIPPED v7", P_ship, L_ship, U_ship)

P_reb, L_reb, U_reb = run_zip(f"{B}/train_work/submission.zip", "rebuilt")
E_reb = report("REBUILT (fixed pipeline)", P_reb, L_reb, U_reb)

print("\nGATE 6: delta local-E (rebuilt - shipped) = %+.4f  (%s)" % (
    E_reb - E_ship, "within noise (<0.02)" if abs(E_reb - E_ship) < 0.02 else "OUT OF NOISE BAND"))
