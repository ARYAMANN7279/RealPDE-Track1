"""tke TRANSFER FORENSICS -- experiment C+D.

C) Is sec56.6 ("greedy-souping the wtke members is negative") CONFOUNDED by leakage?
   sec56.6 mixed the SHIPPED soup (which TRAINED on re_lohi) with ftaug_r2w* members
   (which HELD OUT re_lohi) and scored the mix ON re_lohi. Any such mix must lose tke
   monotonically in alpha -- that is dilution of memorisation, not model quality.
   Honest re-test: mix the HONEST soup with the honest wtke members, both re_lohi-disjoint.

D) Condition-extrapolation ladder: does out-of-condition tke error grow with the DISTANCE
   of the held-out Reynolds number from the training range? If so, how far out would the
   live set have to be to produce the observed live error 0.63160?

MANDATORY: prints split sizes; asserts the honest baseline; ABORTS on mismatch.
Complex spectral weights are preserved (never .float()).
"""
import os, sys, json, argparse, importlib.util as iu
import numpy as np, torch

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"; LH = f"{B}/local_harness"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
sys.path.insert(0, f"{B}/_bld")
from load_baseline import load_baseline

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args(); DEV = f"cuda:{a.gpu}"; C = 2
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)

meta = json.load(open(f"{LH}/tr_meta.json")); off, lens, names = meta["off"], meta["lens"], meta["names"]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
wins, wt = [], []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 10): wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
RE = np.array([int(names[i].split("_")[0]) for i in wt])
AOA = np.array([int(names[i].split("_", 1)[1].replace(".h5", "")) for i in wt])
assert len(wins) == 6602, "window construction changed -- ABORT"
HOLD = [3750, 5025, 25425, 26700]
IDX664 = np.where(np.isin(RE, HOLD))[0][::2]
IDXALL = np.where(np.isin(RE, HOLD))[0]
print("windows total %d | lohi664 %d | lohi_all %d" % (len(wins), len(IDX664), len(IDXALL)), flush=True)
assert len(IDX664) == 664, "ABORT: lohi664 != 664"

pad = lambda arr: np.concatenate([arr, np.zeros(arr.shape[:-1] + (1,), np.float32)], -1)
proto, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)

def load_sd(p):
    sd = torch.load(p, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    return sd

def mix(sds, ws):
    out = {}
    for k in sds[0]:
        v = sds[0][k]
        if torch.is_tensor(v) and (v.is_complex() or torch.is_floating_point(v)):
            acc = torch.zeros_like(v, dtype=torch.complex128 if v.is_complex() else torch.float64)
            for s, w in zip(sds, ws): acc += s[k].to(acc.dtype) * w
            out[k] = acc.to(v.dtype)
        else: out[k] = v
    return out

def predict(sd, idx, bs=16):
    proto.load_state_dict(sd); proto.eval()
    P = np.zeros((len(idx), 20, 32, 64, 2), np.float32); T = np.zeros_like(P)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sl = idx[i:i + bs]
            w = np.stack([np.concatenate([np.asarray(X[s:s + 40]),
                          np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins[sl]]).astype(np.float32)
            xb = torch.from_numpy(w[:, :20]).to(DEV)
            p = (proto((xb - MI) / SI) * ST + MT).cpu().numpy()
            P[i:i + bs] = p[..., :2]; T[i:i + bs] = w[:, 20:, ..., :2]
    return P, T

def sc(P, T):
    return (S.score_error(float(S.rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(pad(P), pad(T)).mean())))

MV = dict(rel_l2=0.669, tke=0.157, mvpe=0.170)
HON = load_sd(f"{B}/train_es/soup_lohi_honest.pth")
P0, T0 = predict(HON, IDX664); b0 = sc(P0, T0)
KNOWN = (95.3069, 78.7002, 96.0540)
print("BASELINE honest soup @lohi664: rel %.4f tke %.4f mvpe %.4f" % b0, flush=True)
dev = max(abs(x - y) for x, y in zip(b0, KNOWN))
assert dev < 0.02, "BASELINE MISMATCH vs %s (dev %.4f) -- ABORT" % (KNOWN, dev)
print("  GATE PASSED (dev %.4f)\n" % dev, flush=True)
dacc = lambda s: sum(MV[k] * (s[i] - b0[i]) for i, k in enumerate(("rel_l2", "tke", "mvpe")))

# ---------- C: honest greedy soup ----------
print("=== C: HONEST mixes (both sides re_lohi-disjoint) ===", flush=True)
MEMS = ["r2w003", "r2w005", "r2w008", "r2w015", "wtke000", "wtke005", "wtke010", "wtke015"]
best = None
for tag in MEMS:
    p = f"{B}/train_es/ftaug_{tag}.pth"
    if not os.path.exists(p): print("  %-8s MISSING" % tag); continue
    m = load_sd(p)
    s = sc(*predict(m, IDX664))
    print("  %-8s ALONE           rel %.4f tke %.4f mvpe %.4f  d_acc %+.4f" % (tag, *s, dacc(s)), flush=True)
    for al in (0.10, 0.20, 0.30, 0.50):
        sm = sc(*predict(mix([HON, m], [1 - al, al]), IDX664))
        flag = ""
        if sm[1] > b0[1] + 0.01: flag = "  <-- tke UP"
        print("     +honest a=%.2f      rel %.4f tke %.4f mvpe %.4f  d_acc %+.4f%s"
              % (al, *sm, dacc(sm), flag), flush=True)

# uniform 10-member re-soup: 6 honest members + the 4 round-2 wtke members
RUNS = f"{B}/train_mvpe/runs"
base6 = ["s_lohi", "L_lr3e5", "L_lr3e6", "L_s16k", "L_s4k", "L_w30"]
sds6 = [load_sd(f"{RUNS}/{m}_besteff.pth") for m in base6]
for extra in (["r2w008"], ["r2w005", "r2w008"], ["r2w003", "r2w005", "r2w008", "r2w015"]):
    sds = sds6 + [load_sd(f"{B}/train_es/ftaug_{t}.pth") for t in extra]
    w = [1.0 / len(sds)] * len(sds)
    s = sc(*predict(mix(sds, w), IDX664))
    print("  uniform soup 6+%-22s rel %.4f tke %.4f mvpe %.4f  d_acc %+.4f"
          % (",".join(extra), *s, dacc(s)), flush=True)

# ---------- D: extrapolation ladder ----------
print("\n=== D: condition-extrapolation ladder (honest soup, per held-out Re) ===", flush=True)
TRAIN_RE = sorted(set(RE.tolist()) - set(HOLD))
print("  train Re range: %d .. %d" % (min(TRAIN_RE), max(TRAIN_RE)), flush=True)
def ke(x):
    u = x[..., 0]; v = x[..., 1]
    return 0.5 * (np.mean((u - u.mean(1, keepdims=True)) ** 2, 1) + np.mean((v - v.mean(1, keepdims=True)) ** 2, 1))
rows = []
for r in HOLD:
    ii = np.where(RE == r)[0]
    P, T = predict(HON, ii)
    e = float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean())
    er = float(S.rel_l2_per_sample(pad(P), pad(T), C).mean())
    d = min(abs(r - min(TRAIN_RE)), abs(r - max(TRAIN_RE)))
    rows.append((r, len(ii), d, e, er, S.score_error(e)))
    print("  Re %5d  n=%4d  dist_from_train_range=%5d  tke_err %.4f  rel_err %.4f  tke %.4f"
          % (r, len(ii), d, e, er, S.score_error(e)), flush=True)
print("\n  in-sample control (honest soup on its own training Re):", flush=True)
ii = np.where(np.isin(RE, [8850, 13950, 19050, 22875]))[0]
P, T = predict(HON, ii); e = float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean())
print("  re_int (IN training) n=%d  tke_err %.4f  tke %.4f" % (len(ii), e, S.score_error(e)), flush=True)
d = np.array([r[2] for r in rows], float); y = np.array([r[3] for r in rows], float)
A = np.polyfit(d, y, 1)
LIVE_ERR = 2 * (100 / 75.998930 - 1)
print("\n  linear fit tke_err = %.6e*dist + %.6f  (in-sample intercept check)" % (A[0], A[1]), flush=True)
print("  live tke_err = %.5f  =>  implied extrapolation distance = %.0f Re units"
      % (LIVE_ERR, (LIVE_ERR - A[1]) / A[0] if A[0] != 0 else float("nan")), flush=True)

print("\n=== per-sample tke error distribution, honest soup @lohi_all ===", flush=True)
P, T = predict(HON, IDXALL)
es = S.tke_rel_l2_per_sample(pad(P), pad(T), C)
print("  mean %.4f  median %.4f  p10 %.4f p90 %.4f  max %.4f" %
      (es.mean(), np.median(es), np.percentile(es, 10), np.percentile(es, 90), es.max()), flush=True)
print("  score of mean-err = %.4f ; mean of per-sample scores = %.4f" %
      (S.score_error(float(es.mean())), float(np.mean([S.score_error(float(x)) for x in es]))), flush=True)
for lo in (0, 5, 10, 15, 20):
    jj = np.where(AOA[IDXALL] == lo)[0]
    if len(jj): print("  AoA %2d  n=%4d  tke_err %.4f  tke %.4f" % (lo, len(jj), es[jj].mean(), S.score_error(float(es[jj].mean()))), flush=True)
print("DONE", flush=True)
