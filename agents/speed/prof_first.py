"""Where does the FIRST predict() of a fresh process spend its time?

mode "predict" (default): in a fresh process, time CUDA-context creation, _get_model, _get_net
separately, then wrap the FNO forward, each SpectralConv (inner fp32 part), and every U-Net trunk
with synchronised per-call timers, and run predict() twice on n windows.  First-call vs
second-call per component isolates one-time costs (cuFFT planning, cuDNN heuristics / runtime
compilation, lazy kernel loading) from steady-state work.

mode "micro": in a fresh process, first/second/third call of the raw primitives at SCREEN's
shapes (rfftn / irfftn on the padded 26x38x70 grid at batch 48 and 4; the fp16 1x1 Conv3d; a
U-Net 3x3 Conv2d at batch 16 and 4).
"""
import time
T0 = time.perf_counter()
import os, sys, json, argparse, importlib.util
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--mode", default="predict", choices=["predict", "micro"])
args = ap.parse_args()

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"


def make_windows(n, seed=0):
    meta = json.load(open(f"{LH}/tr_meta.json"))
    off, lens = meta["off"], meta["lens"]
    st = np.array([t0 for i in range(len(lens)) for t0 in range(off[i], off[i] + lens[i] - 39)])
    idx = np.sort(np.random.default_rng(seed).choice(st, size=n, replace=False))
    X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
    x = np.zeros((n, 20) + tuple(X.shape[1:3]) + (3,), np.float32)
    for k, s in enumerate(idx):
        x[k, :, :, :, :2] = np.asarray(X[s:s + 20], dtype=np.float32)
    return x


def sync_t(fn):
    import torch
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t, r


if args.mode == "micro":
    import torch
    t = time.perf_counter()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    print(f"ctx_create {time.perf_counter() - t:.3f}s  torch {torch.__version__} "
          f"arch {torch.cuda.get_arch_list()} cufft_plan_cache_max "
          f"{torch.backends.cuda.cufft_plan_cache.max_size}", flush=True)
    for b in (48, 4):
        xa = torch.randn(b, 64, 26, 38, 70, device="cuda")
        for r in range(3):
            dt, X = sync_t(lambda: torch.fft.rfftn(xa, dim=[-3, -2, -1]))
            print(f"rfftn  b={b:2d} call{r} {dt * 1e3:9.2f} ms", flush=True)
        for r in range(3):
            dt, _ = sync_t(lambda: torch.fft.irfftn(X, s=(26, 38, 70)))
            print(f"irfftn b={b:2d} call{r} {dt * 1e3:9.2f} ms", flush=True)
    c3 = torch.nn.Conv3d(64, 64, 1).cuda()
    for b in (48, 4):
        xa = torch.randn(b, 64, 26, 38, 70, device="cuda")
        for r in range(3):
            with torch.autocast("cuda", dtype=torch.float16):
                dt, _ = sync_t(lambda: c3(xa))
            print(f"conv3d1x1 fp16 b={b:2d} call{r} {dt * 1e3:9.2f} ms", flush=True)
    c2 = torch.nn.Conv2d(80, 96, 3, padding=1).cuda()
    for b in (16, 4):
        xa = torch.randn(b, 80, 32, 64, device="cuda")
        for r in range(3):
            dt, _ = sync_t(lambda: c2(xa))
            print(f"conv2d3x3 fp32 b={b:2d} call{r} {dt * 1e3:9.2f} ms", flush=True)
    print("[done]", flush=True)
    sys.exit(0)

x = make_windows(args.n)
t = time.perf_counter()
spec = importlib.util.spec_from_file_location("submission", os.path.join(args.sub, "submission.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["submission"] = mod
spec.loader.exec_module(mod)
import torch  # noqa: E402
print(f"import {time.perf_counter() - t:.3f}s (process start->here {time.perf_counter() - T0:.3f}s)",
      flush=True)
t = time.perf_counter()
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"ctx_create {time.perf_counter() - t:.3f}s", flush=True)
dt, (model, device) = sync_t(lambda: mod._get_model())
print(f"_get_model {dt:.3f}s", flush=True)
dt, _r = sync_t(lambda: mod._get_net(device))
net = _r[0]
print(f"_get_net {dt:.3f}s  extra_nets={len(mod._state['extra'])}", flush=True)

rec = []


def wrap(obj, attr, name):
    orig = getattr(obj, attr)

    def w(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig(*a, **k)
        torch.cuda.synchronize()
        rec.append((name, int(a[0].shape[0]) if a and torch.is_tensor(a[0]) else -1,
                    time.perf_counter() - t))
        return r
    setattr(obj, attr, w)


wrap(model, "forward", "fno_forward")
for i, sc in enumerate(model.spectral_convs):
    wrap(sc, "forward", f"spectral{i}_inner")
wrap(net, "trunk", "unet0_trunk")
for j, e in enumerate(mod._state["extra"]):
    wrap(e, "trunk", f"unet{j + 1}_trunk")

for call in range(2):
    rec.clear()
    t = time.perf_counter()
    out = mod.predict(x, {})
    wall = time.perf_counter() - t
    agg = {}
    for name, b, dt in rec:
        agg.setdefault((name, b), []).append(dt)
    print(f"--- predict call {call}: wall {wall:.3f}s for n={args.n} "
          f"(sum of wrapped {sum(d for _, _, d in rec):.3f}s)", flush=True)
    for (name, b), v in sorted(agg.items()):
        print(f"   {name:16s} b={b:3d} calls={len(v):3d} first {v[0] * 1e3:9.2f} ms  "
              f"rest-mean {np.mean(v[1:]) * 1e3 if len(v) > 1 else float('nan'):9.2f} ms  "
              f"total {sum(v) * 1e3:9.2f} ms", flush=True)
print("[done]", flush=True)
