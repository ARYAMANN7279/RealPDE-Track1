"""Score a checkpoint on each candidate validation split.

Purpose: find a LOCAL split whose measured delta matches the REAL leaderboard
delta. The historical every-5th split over-predicts tke gains by ~2x (sec11A:
local +4.20/+4.27 -> real +1.97). If a condition-disjoint split reproduces the
real delta, every future experiment can be decided locally instead of costing
one of the ~32 remaining submission slots.

Real anchors (from the leaderboard, sec1):
  original kit ckpt : rel_l2 94.17  tke 74.03  mvpe 92.84
  SOUP_v1 (shipped) : rel_l2 94.05  tke 76.00  mvpe 92.87
  real delta        :        -0.12       +1.97       +0.03
"""
import sys, os, json, argparse, re as _re
import numpy as np, torch

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--nval", type=int, default=1200)
a = ap.parse_args()
DEV = f"cuda:{a.gpu}"; C = 2; IN = 20

MI = torch.tensor([0.154960856, -0.000513992854, 0.0])
SI = torch.tensor([0.0968056545, 0.015960684, 1.0])
MT = torch.tensor([0.154962569, -0.000517793698, 0.0])
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0])
MI, SI, MT, ST = [t.to(DEV) for t in (MI, SI, MT, ST)]

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
re_of = lambda t: float(_re.match(r"([0-9.]+)_", t).group(1))
aoa_of = lambda t: t.split("_")[1].replace(".h5", "")

SPLITS = {
    "every5":  lambda: set(range(0, ntraj, 5)),
    "re_lohi": lambda: {i for i in range(ntraj) if re_of(names[i]) in {3750., 5025., 25425., 26700.}},
    "re_int":  lambda: {i for i in range(ntraj) if re_of(names[i]) in {8850., 13950., 19050., 22875.}},
    "aoa15":   lambda: {i for i in range(ntraj) if aoa_of(names[i]) == "15"},
    "aoa20":   lambda: {i for i in range(ntraj) if aoa_of(names[i]) == "20"},
    "aoa0":    lambda: {i for i in range(ntraj) if aoa_of(names[i]) == "0"},
    "aoa5":    lambda: {i for i in range(ntraj) if aoa_of(names[i]) == "5"},
    "aoa10":   lambda: {i for i in range(ntraj) if aoa_of(names[i]) == "10"},
    "hiRe":    lambda: {i for i in range(ntraj) if re_of(names[i]) >= 19050.},
    "loRe":    lambda: {i for i in range(ntraj) if re_of(names[i]) <= 8850.},
}

model, _ = load_baseline(a.ckpt, device=DEV); model = model.to(DEV).eval()

def fwd(x): return model((x - MI) / SI) * ST + MT

def batch(idx):
    w = np.stack([np.asarray(X[s:s + 40]) for s in idx])
    z = np.zeros(w.shape[:-1] + (1,), np.float32)
    w = np.concatenate([w, z], -1)
    return torch.from_numpy(w[:, :IN]).to(DEV), torch.from_numpy(w[:, IN:]).to(DEV)

@torch.no_grad()
def score(vidx):
    starts = []
    for i in range(ntraj):
        if i in vidx:
            starts += list(range(off[i], off[i] + lens[i] - 39))
    starts = np.array(starts)
    sub = np.random.default_rng(0).choice(starts, size=min(a.nval, len(starts)), replace=False)
    P, T = [], []
    for i in range(0, len(sub), 16):
        x, y = batch(sub[i:i + 16])
        P.append(fwd(x).cpu().numpy()); T.append(y.cpu().numpy())
    P = np.concatenate(P, 0).astype(np.float32); T = np.concatenate(T, 0).astype(np.float32)
    return (S.score_error(float(S.rel_l2_per_sample(P, T, C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(P, T, C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(P, T).mean())),
            len(sub))

out = {}
print("ckpt: %s" % os.path.basename(a.ckpt), flush=True)
print("%-9s %6s %8s %8s %8s" % ("split", "nval", "rel_l2", "tke", "mvpe"), flush=True)
for nm, fn in SPLITS.items():
    r = score(fn())
    out[nm] = r[:3]
    print("%-9s %6d %8.2f %8.2f %8.2f" % (nm, r[3], r[0], r[1], r[2]), flush=True)
json.dump(out, open(f"{B}/train_mvpe/runs/evalsplit_{os.path.basename(a.ckpt).replace('.pth','')}.json", "w"), indent=1)
