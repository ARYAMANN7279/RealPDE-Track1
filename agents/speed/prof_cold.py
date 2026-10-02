"""Cold-start cost of ONE ingestion part.

The organizers' announcement: "predict may now be called more than once, each call in a fresh
isolated subprocess".  So every call pays: module import, the lazy model/bounds load inside
predict(), CUDA context creation, and cuFFT-plan / cuDNN first-use costs.  This script is run once
per fresh process (by drive_cold.py) and prints one JSON line:

  import_s     submission import (imports torch etc.; no model work happens at import)
  cold_s       wall of the FIRST predict() on n windows  (what a timer around predict() sees)
  loaders      (only with --hooks) time inside the lazy loaders, synchronised
  warm_s       wall of --warm further predict() calls on the same windows, same process
"""
import time
_T_PROC = time.perf_counter()
import os, sys, json, argparse, importlib.util
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--hooks", action="store_true")
ap.add_argument("--ctx_first", action="store_true", help="create the CUDA context before predict()")
ap.add_argument("--evict", action="store_true", help="drop checkpoint/npz pages from the OS page cache")
ap.add_argument("--tag", default="")
args = ap.parse_args()

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"


def make_windows(n, seed):
    meta = json.load(open(f"{LH}/tr_meta.json"))
    off, lens = meta["off"], meta["lens"]
    st = np.array([t0 for i in range(len(lens)) for t0 in range(off[i], off[i] + lens[i] - 39)])
    idx = np.sort(np.random.default_rng(seed).choice(st, size=n, replace=False))
    X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
    x = np.zeros((n, 20) + tuple(X.shape[1:3]) + (3,), np.float32)
    for k, s in enumerate(idx):
        x[k, :, :, :, :2] = np.asarray(X[s:s + 20], dtype=np.float32)
    return x


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


x = make_windows(args.n, args.seed)
if args.evict:
    for f in ("sim_real_fno_fp16.pth", "bounds_assets.npz"):
        evict(os.path.join(args.sub, f))

res = dict(tag=args.tag, sub=args.sub, n=args.n, hooks=args.hooks, ctx_first=args.ctx_first,
           evict=args.evict)
t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location("submission", os.path.join(args.sub, "submission.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["submission"] = mod
spec.loader.exec_module(mod)
res["import_s"] = time.perf_counter() - t0
import torch  # noqa: E402  (already imported by the submission)

T = {}


def timed(fn, name, sync=True):
    def w(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        if sync and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        T[name] = T.get(name, 0.0) + time.perf_counter() - t
        return r
    return w


if args.hooks:
    LB = sys.modules.get("load_baseline")
    if LB is not None:
        for fname, key in (("build_model", "build_model"), ("load_checkpoint_state", "load_ckpt_state")):
            if hasattr(LB, fname):
                setattr(LB, fname, timed(getattr(LB, fname), key))
    _lsd = torch.nn.Module.load_state_dict

    def lsd(self, *a, **k):
        t = time.perf_counter()
        r = _lsd(self, *a, **k)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        key = "load_state_dict_" + type(self).__name__
        T[key] = T.get(key, 0.0) + time.perf_counter() - t
        return r
    torch.nn.Module.load_state_dict = lsd
    for fname in ("_get_model", "_get_net", "_assets"):
        if hasattr(mod, fname):
            setattr(mod, fname, timed(getattr(mod, fname), fname, sync=(fname != "_assets")))

if args.ctx_first:
    t = time.perf_counter()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    res["ctx_s"] = time.perf_counter() - t

t = time.perf_counter()
out = mod.predict(x, {})
res["cold_s"] = time.perf_counter() - t
res["loaders"] = dict(T)
res["out_ok"] = bool(all(np.isfinite(out[k]).all() for k in ("prediction", "lower", "upper"))
                     and (out["lower"] <= out["upper"]).all())
del out
res["warm_s"] = []
for _ in range(args.warm):
    t = time.perf_counter()
    out = mod.predict(x, {})
    res["warm_s"].append(time.perf_counter() - t)
    del out
res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
res["proc_total_s"] = time.perf_counter() - _T_PROC
print("JSON " + json.dumps(res), flush=True)
