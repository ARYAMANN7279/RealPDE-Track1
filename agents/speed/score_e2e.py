"""End-to-end through the kit's own scoring.py.

In THIS fresh process: import <sub>/submission.py, call its predict() once on the 900 re_lohi
windows (same construction as r12_eval.val_starts(900) / window(), rng 0), write
<out>/in/predictions.npz (prediction, lower, upper, mean_t_neural_s measured around predict())
and <out>/in/ref/targets.npz, then run  `python scoring.py <out>/in <out>/res`  as a subprocess
and print its scores.json.  Also prints md5 of the three output arrays."""
import os, sys, json, time, argparse, importlib.util, hashlib, subprocess
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"


def val_starts(n=900):
    meta = json.load(open(f"{LH}/tr_meta.json"))
    off, lens, names = meta["off"], meta["lens"], meta["names"]
    RE = np.array([int(x.split("_")[0]) for x in names])
    VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())
    st = [t0 for i in range(len(lens)) if i in VT for t0 in range(off[i], off[i] + lens[i] - 39)]
    rng = np.random.default_rng(0)
    return rng.choice(np.array(st), size=min(n, len(st)), replace=False)


def window(idx):
    X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
    w = np.stack([np.asarray(X[s:s + 40]) for s in idx]).astype(np.float32)
    w = np.concatenate([w, np.zeros(w.shape[:-1] + (1,), np.float32)], -1)
    return np.ascontiguousarray(w[:, :20]), np.ascontiguousarray(w[:, 20:])


x, y = window(val_starts(900))
spec = importlib.util.spec_from_file_location("submission", os.path.join(args.sub, "submission.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["submission"] = mod
spec.loader.exec_module(mod)
t = time.perf_counter()
out = mod.predict(x, {})
dt = time.perf_counter() - t
ind = os.path.join(args.out, "in")
os.makedirs(os.path.join(ind, "ref"), exist_ok=True)
np.savez(os.path.join(ind, "predictions.npz"), prediction=out["prediction"], lower=out["lower"],
         upper=out["upper"], mean_t_neural_s=np.array([dt / len(x)], np.float64))
np.savez(os.path.join(ind, "ref", "targets.npz"), target=y)
md5 = {k: hashlib.md5(np.ascontiguousarray(out[k]).view(np.uint8)).hexdigest()
       for k in ("prediction", "lower", "upper")}
res = os.path.join(args.out, "res")
p = subprocess.run([sys.executable, os.path.join(KIT, "scoring.py"), ind, res],
                   capture_output=True, text=True)
print("scoring.py rc", p.returncode, p.stderr[-2000:])
sc = json.load(open(os.path.join(res, "scores.json")))
print(json.dumps(dict(sub=args.sub, n=len(x), cold_call_ms_per_sample=dt / len(x) * 1e3,
                      md5=md5, scores=sc), indent=1))
print("[done]")
