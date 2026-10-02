"""End-to-end timing of a submission's predict(), called the way an ingestion loop would.

VM usage:
  setsid nohup env CUDA_VISIBLE_DEVICES=3 $P prof_e2e.py --sub <dir> --n 5140 --reps 5 --tag screen \
      [--save_ref <dir>] [--cmp_ref <dir>] [--chunk 0] < /dev/null > log 2>&1 & disown

Reports: module import time, cold first predict() (split into _get_model / _get_net / rest),
every warm predict() call in ms/sample, repeat-call identity, and (optionally) max|delta| of
prediction/lower/upper against a saved reference produced by the banked submission.
"""
import os, sys, time, json, argparse, importlib.util, hashlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True, help="directory containing submission.py")
ap.add_argument("--n", type=int, default=5140)
ap.add_argument("--reps", type=int, default=5, help="warm calls after the cold one")
ap.add_argument("--tag", default="run")
ap.add_argument("--save_ref", default="", help="dir: save cold-call outputs as .npy")
ap.add_argument("--cmp_ref", default="", help="dir: compare outputs to saved .npy")
ap.add_argument("--chunk", type=int, default=0, help="0 = one call with all N; else calls of this size")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json", default="", help="write summary json here")
args = ap.parse_args()

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KEYS = ("prediction", "lower", "upper")


def make_windows(n, seed):
    """Real windows from the training cache, same construction as r12_eval.window()."""
    meta = json.load(open(f"{LH}/tr_meta.json"))
    off, lens = meta["off"], meta["lens"]
    st = np.array([t0 for i in range(len(lens)) for t0 in range(off[i], off[i] + lens[i] - 39)])
    idx = np.sort(np.random.default_rng(seed).choice(st, size=n, replace=False))
    X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
    x = np.zeros((n, 20) + tuple(X.shape[1:3]) + (3,), np.float32)
    for k, s in enumerate(idx):
        x[k, :, :, :, :2] = np.asarray(X[s:s + 20], dtype=np.float32)
    return x, idx


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def call(mod, x):
    if args.chunk <= 0:
        return mod.predict(x, {})
    parts = [mod.predict(x[i:i + args.chunk], {}) for i in range(0, len(x), args.chunk)]
    return {k: np.concatenate([p[k] for p in parts], 0) for k in KEYS}


x, idx = make_windows(args.n, args.seed)
print(f"[{args.tag}] windows {x.shape} idx_md5 {md5(idx)} x_md5 {md5(x)}", flush=True)

t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location("submission", os.path.join(args.sub, "submission.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["submission"] = mod
spec.loader.exec_module(mod)
t_imp = time.perf_counter() - t0

# time the lazy loaders on the cold call without changing what they do
_lt = {}
for name in ("_get_model", "_get_net"):
    if hasattr(mod, name):
        orig = getattr(mod, name)

        def wrap(*a, _o=orig, _n=name, **k):
            t = time.perf_counter()
            r = _o(*a, **k)
            _lt[_n] = _lt.get(_n, 0.0) + time.perf_counter() - t
            return r
        setattr(mod, name, wrap)

import torch  # already imported by the submission; no CUDA work happens here
print(f"[{args.tag}] torch {torch.__version__} cudnn {torch.backends.cudnn.version()} "
      f"import {t_imp * 1e3:.1f} ms", flush=True)

ref = None
if args.cmp_ref:
    ref = {k: np.load(os.path.join(args.cmp_ref, k + ".npy"), mmap_mode="r") for k in KEYS}

rows, first = [], None
for r in range(args.reps + 1):
    _lt.clear()
    t0 = time.perf_counter()
    out = call(mod, x)
    dt = time.perf_counter() - t0
    row = dict(call=r, s=dt, ms_per_sample=dt / args.n * 1e3,
               get_model_s=_lt.get("_get_model", 0.0), get_net_s=_lt.get("_get_net", 0.0))
    if r == 0:
        first = {k: np.array(out[k], copy=True) for k in KEYS}
        row["md5"] = {k: md5(first[k]) for k in KEYS}
        row["finite"] = bool(all(np.isfinite(first[k]).all() for k in KEYS))
        row["lower_le_upper"] = bool((first["lower"] <= first["upper"]).all())
        if ref is not None:
            row["vs_ref_maxabs"] = {k: float(np.max(np.abs(first[k] - ref[k]))) for k in KEYS}
            row["vs_ref_equal"] = {k: bool(np.array_equal(first[k], ref[k])) for k in KEYS}
    else:
        row["same_as_call0"] = {k: bool(np.array_equal(out[k], first[k])) for k in KEYS}
    rows.append(row)
    print(f"[{args.tag}] " + json.dumps(row), flush=True)
    del out

warm = np.array([rw["ms_per_sample"] for rw in rows[1:]]) if len(rows) > 1 else np.array([np.nan])
summ = dict(tag=args.tag, sub=args.sub, n=args.n, chunk=args.chunk, torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            import_s=t_imp, cold=rows[0], warm_ms=warm.tolist(),
            warm_median=float(np.median(warm)), warm_min=float(np.min(warm)),
            warm_max=float(np.max(warm)))
print(f"[{args.tag}] SUMMARY cold {rows[0]['ms_per_sample']:.4f} ms/sample | warm median "
      f"{summ['warm_median']:.4f} (min {summ['warm_min']:.4f} max {summ['warm_max']:.4f})", flush=True)

if args.save_ref:
    os.makedirs(args.save_ref, exist_ok=True)
    for k in KEYS:
        np.save(os.path.join(args.save_ref, k + ".npy"), first[k])
    np.save(os.path.join(args.save_ref, "idx.npy"), idx)
    print(f"[{args.tag}] saved reference to {args.save_ref}", flush=True)
if args.json:
    with open(args.json, "w") as f:
        json.dump(summ, f, indent=1)
print(f"[{args.tag}] [done]", flush=True)
