"""Evaluate official foil baselines (FNO/CNO/Transolver) individually and as
ensembles, on REAL held-out windows, scored with the official scoring.py.
CPU-only, no GPU. Caches each model's mirror-TTA prediction so ensemble
weightings are cheap to sweep.
"""
import importlib.util
import itertools
import os
import sys
import time

import numpy as np
import torch

HARNESS = "/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT = "/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
CKPT = os.path.join(HARNESS, "baseline_ckpts", "foil")
FNO_FP16 = os.path.join(HARNESS, "fno_model", "sim_real_fno_fp16.pth")
EVAL = os.path.join(HARNESS, "real_eval_heldout")

sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline  # noqa

spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
scoring = importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)

# GaussianNormalizer stats on train_real (same for every model; data stats).
MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)

DEV = "cpu"
mi = torch.tensor(MEAN_IN); si = torch.tensor(STD_IN)
mt = torch.tensor(MEAN_TGT); st = torch.tensor(STD_TGT)


@torch.no_grad()
def infer(model, X, mirror=True, batch=16):
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i+batch]).float()
        xn = (xb - mi) / si
        y = model(xn) * st + mt
        if mirror:
            x2 = torch.flip(xb, dims=[2]).clone(); x2[..., 1] = -x2[..., 1]
            y2 = model((x2 - mi) / si) * st + mt
            y2 = torch.flip(y2, dims=[2]).clone(); y2[..., 1] = -y2[..., 1]
            y = 0.5 * (y + y2)
        outs.append(y.numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def score(pred, Y, half=None):
    c = scoring.measured_channels(Y)
    rel = float(np.mean(scoring.rel_l2_per_sample(pred, Y, c)))
    tke = float(np.mean(scoring.tke_rel_l2_per_sample(pred, Y, c)))
    mvpe = scoring.mvpe_rel_l2(pred, Y)
    if half is not None:
        lo, hi = pred - half, pred + half
        sps, cov = scoring.aggregate_sps(pred, Y, c, lower=lo, upper=hi)
    else:
        sps, cov = scoring.aggregate_sps(pred, Y, c)
    return dict(L2=scoring.score_error(rel), TKE=scoring.score_error(tke),
                MVPE=scoring.score_error(mvpe), SPS=scoring.score_sps(sps),
                cov=cov*100, rel=rel, tke=tke, mvpe=mvpe)


def line(name, s):
    proxy4 = np.mean([s['L2'], s['TKE'], s['MVPE'], s['SPS']])
    print(f"{name:26s} L2={s['L2']:6.2f} TKE={s['TKE']:6.2f} MVPE={s['MVPE']:6.2f} "
          f"SPS={s['SPS']:5.2f} cov={s['cov']:5.1f}%  | acc3={np.mean([s['L2'],s['TKE'],s['MVPE']]):.2f}")


def main():
    X = np.load(os.path.join(EVAL, "inputs.npz"))["input"]
    Y = np.load(os.path.join(EVAL, "targets.npz"))["target"]
    print(f"held-out: {X.shape[0]} windows\n")

    want = [
        ("FNO", lambda: load_baseline(FNO_FP16, device=DEV)[0]),
        ("CNO", lambda: load_baseline("cno", os.path.join(CKPT, "cno", "finetune.pth"), device=DEV)[0]),
        ("Transolver", lambda: load_baseline("transolver", os.path.join(CKPT, "transolver", "finetune.pth"), device=DEV, strict=False)[0]),
    ]
    models = {}
    for name, loader in want:
        try:
            t0 = time.time(); models[name] = loader(); print(f"loaded {name} {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"SKIP {name}: {type(e).__name__}: {str(e)[:80]}")

    preds = {}
    for name, m in models.items():
        t0 = time.time(); preds[name] = infer(m, X); dt = time.time()-t0
        print(f"  inferred {name} {dt:.0f}s ({dt/X.shape[0]*1000:.0f}ms/sample)")
    print()

    names = list(preds.keys())
    for name in names:
        line(name, score(preds[name], Y))
    print()
    # all pairwise + full ensembles
    for combo in itertools.chain(itertools.combinations(names, 2), itertools.combinations(names, 3)):
        p = sum(preds[n] for n in combo) / len(combo)
        line("+".join(combo), score(p, Y))
    # weight sweep across available models by acc3
    if len(names) >= 2:
        best = None
        grid = np.arange(0, 1.01, 0.1)
        if len(names) == 2:
            for w in grid:
                p = w*preds[names[0]] + (1-w)*preds[names[1]]
                s = score(p, Y); acc = np.mean([s['L2'],s['TKE'],s['MVPE']])
                if best is None or acc > best[0]: best = (acc, {names[0]: round(w,1), names[1]: round(1-w,1)}, s)
        print(f"\nbest by acc3: {best[1]}")
        line("  best-weighted", best[2])
    np.savez(os.path.join(EVAL, "cached_preds.npz"), **preds, Y=Y)
    print("\ncached preds ->", os.path.join(EVAL, "cached_preds.npz"))


if __name__ == "__main__":
    main()
