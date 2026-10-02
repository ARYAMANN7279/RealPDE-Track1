"""Run any extracted submission's predict() on the real validation windows and
score with the OFFICIAL scoring.py. CPU/MPS — no GPU needed.

    python score_real.py <extracted_submission_dir>
"""
import importlib.util
import os
import sys
import time

import numpy as np

HARNESS = "/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT = "/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
EVAL = os.path.join(HARNESS, "real_eval_heldout")

# official scorer
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def load_predict(sub_dir):
    sys.path.insert(0, sub_dir)
    cwd = os.getcwd()
    os.chdir(sub_dir)
    spec = importlib.util.spec_from_file_location("submission", os.path.join(sub_dir, "submission.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.chdir(cwd)
    return mod.predict


def main():
    sub_dir = sys.argv[1]
    X = np.load(os.path.join(EVAL, "inputs.npz"))["input"]
    Y = np.load(os.path.join(EVAL, "targets.npz"))["target"]
    print(f"eval windows: {X.shape}, targets {Y.shape}")

    predict = load_predict(sub_dir)
    t0 = time.time()
    out = predict(X, {})
    dt = time.time() - t0
    if isinstance(out, dict):
        pred = np.asarray(out["prediction"], np.float32)
        lower = out.get("lower"); upper = out.get("upper")
    else:
        pred = np.asarray(out, np.float32); lower = upper = None
    per = dt / max(X.shape[0], 1)

    c = scoring.measured_channels(Y)
    rel = float(np.mean(scoring.rel_l2_per_sample(pred, Y, c)))
    tke = float(np.mean(scoring.tke_rel_l2_per_sample(pred, Y, c)))
    mvpe = scoring.mvpe_rel_l2(pred, Y)
    sps, cov = scoring.aggregate_sps(pred, Y, c, lower=lower, upper=upper)
    s = dict(
        rel_l2=scoring.score_error(rel), tke=scoring.score_error(tke),
        mvpe=scoring.score_error(mvpe), time=scoring.score_time(per),
        sps=scoring.score_sps(sps), cov=cov * 100,
    )
    print(f"\n{'RESULT':22s} L2={s['rel_l2']:.2f} TKE={s['tke']:.2f} "
          f"MVPE={s['mvpe']:.2f} SPS={s['sps']:.2f} Time={s['time']:.2f} "
          f"cov={s['cov']:.1f}% | {per*1000:.1f}ms/sample")
    print(f"equal-weight-mean proxy: {np.mean([s['rel_l2'],s['tke'],s['mvpe'],s['time'],s['sps']]):.2f}")
    print(f"raw: rel_l2={rel:.4f} tke={tke:.4f} mvpe={mvpe:.4f}")


if __name__ == "__main__":
    main()
