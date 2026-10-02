"""tke TRANSFER FORENSICS -- experiment A.

Question: our tke is 80.80 LOCAL (900-window protocol, SHIPPED soup) but 76.00 LIVE.
Is that -4.80 a TRANSFER gap, or is part of it MEMORISATION (the shipped soup trained on
the very trajectories the local protocol validates on)?

Measures, on IDENTICAL windows:
  kit          sim_real_fno.pth         (organizer baseline, trained on all real)
  shipped      _bldfp16 fp16 soup       (our banked model; TRAINED on re_lohi)
  honest_lohi  soup_lohi_honest.pth     (6-member soup that genuinely HELD OUT re_lohi)

across splits: lohi664 / lohi_all / aoa15 / re_int / all.

Also decomposes each tke error into MAGNITUDE vs PATTERN:
  err   = ||KE_p - KE_t|| / ||KE_t||
  a*    = <KE_p,KE_t>/<KE_p,KE_p>            (per-sample optimal rescale)
  err_pat = ||a* KE_p - KE_t|| / ||KE_t||    (= sin of the angle; magnitude removed)

MANDATORY: prints split sizes; asserts honest_lohi@lohi664 == known; ABORTS on mismatch.
NEVER .float() a state dict (complex64 spectral weights) -- uses unpack_fp16 / direct load.
"""
import os, sys, json, argparse, importlib.util as iu
import numpy as np, torch

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
LH = f"{B}/local_harness"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
sys.path.insert(0, f"{B}/_bld")
from load_baseline import load_baseline
sd_spec = iu.spec_from_file_location("lb2", f"{B}/_bld/load_baseline.py")
LB2 = iu.module_from_spec(sd_spec); sd_spec.loader.exec_module(LB2)

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
DEV = f"cuda:{a.gpu}"; C = 2
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)

meta = json.load(open(f"{LH}/tr_meta.json")); off, lens, names = meta["off"], meta["lens"], meta["names"]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
# window list EXACTLY as train_es/mkcache_honest.py builds it (stride 10)
wins, wt = [], []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 10):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
RE = np.array([int(names[i].split("_")[0]) for i in wt])          # per-WINDOW via wt (sec55.3)
AOA = np.array([int(names[i].split("_", 1)[1].replace(".h5", "")) for i in wt])
print("total windows: %d  (expect 6602)" % len(wins), flush=True)
assert len(wins) == 6602, "window construction changed -- ABORT"

HOLD = [3750, 5025, 25425, 26700]
INT = [8850, 13950, 19050, 22875]
SPLITS = {
    "lohi664":  np.where(np.isin(RE, HOLD))[0][::2],
    "lohi_all": np.where(np.isin(RE, HOLD))[0],
    "aoa15":    np.where(AOA == 15)[0],
    "re_int":   np.where(np.isin(RE, INT))[0],
    "all":      np.arange(len(wins)),
}
for k, v in SPLITS.items():
    print("  split %-9s : %5d windows | Re %s | AoA %s" % (
        k, len(v), sorted(set(RE[v].tolist()))[:6], sorted(set(AOA[v].tolist()))), flush=True)
assert len(SPLITS["lohi664"]) == 664, "lohi664 must be 664 -- ABORT"

pad = lambda arr: np.concatenate([arr, np.zeros(arr.shape[:-1] + (1,), np.float32)], -1)

def predict(model, idx, bs=16):
    P = np.zeros((len(idx), 20, 32, 64, 2), np.float32)
    T = np.zeros((len(idx), 20, 32, 64, 2), np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sl = idx[i:i + bs]
            w = np.stack([np.concatenate([np.asarray(X[s:s + 40]),
                                          np.zeros((40, 32, 64, 1), np.float32)], -1)
                          for s in wins[sl]]).astype(np.float32)
            xb = torch.from_numpy(w[:, :20]).to(DEV)
            p = (model((xb - MI) / SI) * ST + MT).cpu().numpy()
            P[i:i + bs] = p[..., :2]; T[i:i + bs] = w[:, 20:, ..., :2]
    return P, T

def ke(x):
    u = x[..., 0]; v = x[..., 1]
    up = np.mean((u - u.mean(axis=1, keepdims=True)) ** 2, axis=1)
    vp = np.mean((v - v.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 0.5 * (up + vp)

def report(P, T):
    r = S.score_error(float(S.rel_l2_per_sample(pad(P), pad(T), C).mean()))
    t = S.score_error(float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean()))
    m = S.score_error(float(S.mvpe_rel_l2_per_sample(pad(P), pad(T)).mean()))
    KP = ke(P).reshape(len(P), -1); KT = ke(T).reshape(len(T), -1)
    nt = np.linalg.norm(KT, axis=1).clip(min=1e-12)
    err = np.linalg.norm(KP - KT, axis=1) / nt
    astar = (KP * KT).sum(1) / (KP * KP).sum(1).clip(min=1e-30)
    errpat = np.linalg.norm(astar[:, None] * KP - KT, axis=1) / nt
    return dict(rel_l2=r, tke=t, mvpe=m,
                tke_err=float(err.mean()), tke_err_pat=float(errpat.mean()),
                ke_ratio=float(KP.sum() / KT.sum()), astar=float(np.median(astar)))

MODELS = {}
m, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV); m.eval(); MODELS["kit"] = m
m2, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
st = LB2.unpack_fp16(f"{B}/_bldfp16/sim_real_fno_fp16.pth")
m2.load_state_dict(st if not isinstance(st, tuple) else st[0]); m2.eval(); MODELS["shipped"] = m2
m3, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
st3 = torch.load(f"{B}/train_es/soup_lohi_honest.pth", map_location=DEV)
m3.load_state_dict(st3); m3.eval(); MODELS["honest_lohi"] = m3
for k, mm in MODELS.items():
    ncplx = sum(1 for p in mm.parameters() if p.is_complex())
    print("model %-12s complex params kept: %d" % (k, ncplx), flush=True)

KNOWN = (95.3069, 78.7002, 96.0540)
res = {}
for sn, idx in SPLITS.items():
    for mn, mm in MODELS.items():
        P, T = predict(mm, idx)
        d = report(P, T); res[(sn, mn)] = d
        print("%-9s %-12s rel %8.4f  tke %8.4f  mvpe %8.4f | tke_err %.4f pat %.4f "
              "ke_ratio %.4f a* %.4f" % (sn, mn, d["rel_l2"], d["tke"], d["mvpe"],
              d["tke_err"], d["tke_err_pat"], d["ke_ratio"], d["astar"]), flush=True)
        if sn == "lohi664" and mn == "honest_lohi":
            got = (d["rel_l2"], d["tke"], d["mvpe"])
            dev = max(abs(x - y) for x, y in zip(got, KNOWN))
            print("  GATE honest@lohi664 vs KNOWN %s -> maxdev %.4f" % (KNOWN, dev), flush=True)
            assert dev < 0.02, "BASELINE MISMATCH -- HARNESS BROKEN, ABORT"
            print("  GATE PASSED", flush=True)
        del P, T

print("\n=== MEMORISATION DELTA (shipped - honest_lohi), identical windows ===", flush=True)
for sn in SPLITS:
    a1 = res[(sn, "shipped")]; a2 = res[(sn, "honest_lohi")]
    print("%-9s d_rel %+7.4f  d_tke %+7.4f  d_mvpe %+7.4f" % (
        sn, a1["rel_l2"] - a2["rel_l2"], a1["tke"] - a2["tke"], a1["mvpe"] - a2["mvpe"]), flush=True)
json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(f"{B}/_tke/tkeA.json", "w"), indent=1)
print("\nwrote _tke/tkeA.json", flush=True)
