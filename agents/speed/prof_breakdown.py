"""Warm steady-state breakdown of the banked predict() (GPU branch).

1. Wall time of the unmodified predict() on n windows (reps, after warm-up).
2. A VERBATIM replica of the GPU branch with synchronised section timers.  Its outputs are
   checked bit-for-bit against predict(), so the attribution is of the real computation.
3. torch.profiler over one predict(): GPU busy time = union of device-activity intervals, vs wall;
   kernel/launch counts; top ops by device time.
4. Micro-costs: FNO3d.get_grid (CPU build + pageable H2D, runs on every forward) and the two
   device->host copy styles.
"""
import os, sys, time, json, argparse, importlib.util
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True)
ap.add_argument("--n", type=int, default=960)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--prof_n", type=int, default=480)
ap.add_argument("--json", default="")
ap.add_argument("--no_profile", action="store_true")
args = ap.parse_args()

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KEYS = ("prediction", "lower", "upper")


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


x = make_windows(args.n, 0)
spec = importlib.util.spec_from_file_location("submission", os.path.join(args.sub, "submission.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["submission"] = mod
spec.loader.exec_module(mod)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

res = dict(sub=args.sub, n=args.n, torch=torch.__version__)
mod.predict(x[:100], {})          # cold: loads, fp32 wrap, cuFFT plans (48 and 4), cuDNN
mod.predict(x[:100], {})

# ---- 1. unmodified predict() -------------------------------------------------------------
plain, ref = [], None
for r in range(args.reps):
    t = time.perf_counter()
    ref = mod.predict(x, {})
    plain.append(time.perf_counter() - t)
res["plain_ms_per_sample"] = [v / args.n * 1e3 for v in plain]
print("plain ms/sample", [round(v, 4) for v in res["plain_ms_per_sample"]], flush=True)

# ---- 2. verbatim replica with section timers ---------------------------------------------
model, device = mod._get_model()
net, lut_t, ed_t = mod._get_net(device)
z = mod._assets()
alpha = float(z["alpha"])
cw = mod._state["cw"]
_mpa = float(mod._state.get("mp_alpha", 0.0) or 0.0)
_mh = mod._state.get("mh") if _mpa != 0.0 else None
_fold = mod._fold
BATCH, CH = mod._BATCH, mod._CHUNK_GPU
mean_in = torch.from_numpy(mod._MEAN_IN).to(device)
std_in = torch.from_numpy(mod._STD_IN).to(device)
mean_tgt = torch.from_numpy(mod._MEAN_TGT).to(device)
std_tgt = torch.from_numpy(mod._STD_TGT).to(device)
res["n_extra_nets"] = len(mod._state["extra"])
res["mh_on"] = _mh is not None


class Timer:
    def __init__(self):
        self.acc, self.t = {}, None

    def start(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def lap(self, name):
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + t - self.t
        self.t = t


def replica(x, tm):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    tm.start()
    prediction = np.empty((n, 20) + x.shape[2:], dtype=np.float32)
    lower = np.empty_like(prediction)
    upper = np.empty_like(prediction)
    tm.lap("alloc_outputs")
    with torch.no_grad():
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            xb_raw = torch.from_numpy(np.ascontiguousarray(x[i:j])).to(device)
            tm.lap("h2d_input")
            with torch.autocast('cuda', dtype=torch.float16, enabled=True):
                yb = (model((xb_raw - mean_in) / std_in) * std_tgt + mean_tgt).float()
            tm.lap("fno")
            lo = torch.zeros_like(yb)
            up = torch.zeros_like(yb)
            for k in range(0, j - i, CH):
                m = min(k + CH, j - i)
                ui = torch.cat([_fold(xb_raw[k:m, :, :, :, :2]), _fold(yb[k:m, :, :, :, :2])], 1)
                _d1 = net.trunk(ui)
                o = net.out(_d1)
                _dl = [_d1]
                c = o[:, :40].reshape(-1, 20, 2, 32, 64).float() * cw[0]
                w = o[:, 40:].reshape(-1, 20, 2, 32, 64).float()
                for _e, _wgt in zip(mod._state["extra"], cw[1:]):
                    _de = _e.trunk(ui)
                    _dl.append(_de)
                    c = c + _e.out(_de)[:, :40].reshape(-1, 20, 2, 32, 64).float() * _wgt
                tm.lap("unets")
                h = torch.empty_like(w)
                for ci in (0, 1):
                    h[:, :, ci] = lut_t[torch.bucketize(w[:, :, ci].contiguous(), ed_t[ci], right=True), ci]
                ctr = yb[k:m, :, :, :, :2].permute(0, 1, 4, 2, 3) + alpha * c
                lo[k:m, :, :, :, :2] = (ctr - h).permute(0, 1, 3, 4, 2)
                up[k:m, :, :, :, :2] = (ctr + h).permute(0, 1, 3, 4, 2)
                tm.lap("lut_bounds")
                if _mh is not None:
                    mc = F.conv2d(torch.cat(_dl, 1), _mh[0], _mh[1], padding=1) * _mpa
                    yb[k:m, :, :, :, :2] += mc.permute(0, 2, 3, 1).unsqueeze(1)
                tm.lap("meta_head")
            yb[..., 2] = 0.0
            tm.lap("misc")
            a = yb.cpu()
            b = lo.cpu()
            cc = up.cpu()
            tm.lap("d2h_pageable")
            prediction[i:j] = a.numpy()
            lower[i:j] = b.numpy()
            upper[i:j] = cc.numpy()
            tm.lap("host_copy")
    np.minimum(lower, upper, out=lower)
    tm.lap("np_minimum")
    return {"prediction": prediction, "lower": lower, "upper": upper}


secs = []
for r in range(args.reps):
    tm = Timer()
    out = replica(x, tm)
    secs.append({k: v / args.n * 1e3 for k, v in tm.acc.items()})
    if r == 0:
        res["replica_identical"] = {k: bool(np.array_equal(out[k], ref[k])) for k in KEYS}
        print("replica identical to predict():", res["replica_identical"], flush=True)
    del out
res["sections_ms_per_sample_median"] = {k: float(np.median([s[k] for s in secs])) for k in secs[0]}
res["sections_total_median"] = float(np.median([sum(s.values()) for s in secs]))
print("sections (ms/sample, median of reps):", flush=True)
for k, v in res["sections_ms_per_sample_median"].items():
    print(f"   {k:14s} {v:.4f}", flush=True)
print(f"   {'TOTAL':14s} {res['sections_total_median']:.4f}", flush=True)

# ---- 4. micro-costs ------------------------------------------------------------------------
tm = Timer()
tm.start()
for _ in range(20):
    g = model.get_grid((BATCH, 20, 32, 64, 3), device)
tm.lap("get_grid")
res["get_grid_ms_per_call"] = tm.acc["get_grid"] / 20 * 1e3
res["get_grid_ms_per_sample"] = res["get_grid_ms_per_call"] / BATCH
yb = torch.randn(BATCH, 20, 32, 64, 3, device=device)
dst = np.empty((BATCH, 20, 32, 64, 3), np.float32)
tm = Timer()
tm.start()
for _ in range(20):
    torch.from_numpy(dst).copy_(yb)
tm.lap("d2h_direct")
for _ in range(20):
    a = yb.cpu()
    dst[...] = a.numpy()
tm.lap("d2h_cpu_then_copy")
res["d2h_direct_ms_per_sample_per_output"] = tm.acc["d2h_direct"] / 20 / BATCH * 1e3
res["d2h_cpu_then_copy_ms_per_sample_per_output"] = tm.acc["d2h_cpu_then_copy"] / 20 / BATCH * 1e3
print("get_grid ms/call %.3f (%.4f ms/sample) | d2h per output per sample: direct %.4f, cpu+copy %.4f"
      % (res["get_grid_ms_per_call"], res["get_grid_ms_per_sample"],
         res["d2h_direct_ms_per_sample_per_output"], res["d2h_cpu_then_copy_ms_per_sample_per_output"]),
      flush=True)

# ---- 3. profiler ---------------------------------------------------------------------------
if not args.no_profile:
    from torch.profiler import profile, ProfilerActivity
    xs = x[:args.prof_n]
    mod.predict(xs, {})
    t = time.perf_counter()
    mod.predict(xs, {})
    wall_np = time.perf_counter() - t
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        mod.predict(xs, {})
        wall = time.perf_counter() - t
    DT = torch.autograd.DeviceType
    evs = list(prof.events())
    dev = [e for e in evs if e.device_type == DT.CUDA]
    iv = sorted((e.time_range.start, e.time_range.end) for e in dev)
    busy, cs, ce = 0.0, None, None
    for a0, b0 in iv:
        if cs is None:
            cs, ce = a0, b0
        elif a0 <= ce:
            ce = max(ce, b0)
        else:
            busy += ce - cs
            cs, ce = a0, b0
    if cs is not None:
        busy += ce - cs
    span = (iv[-1][1] - iv[0][0]) if iv else 0.0
    lnames = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx")
    res["profile"] = dict(n=len(xs), wall_unprofiled_ms=wall_np * 1e3, wall_profiled_ms=wall * 1e3,
                          gpu_busy_union_ms=busy / 1e3, gpu_span_ms=span / 1e3,
                          busy_frac_of_unprofiled_wall=(busy / 1e3) / (wall_np * 1e3),
                          n_device_events=len(dev),
                          n_launch_calls=sum(1 for e in evs if e.name in lnames))
    print("PROFILE", json.dumps(res["profile"]), flush=True)
    try:
        tbl = prof.key_averages().table(sort_by="self_device_time_total", row_limit=30)
    except Exception:
        tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    print(tbl, flush=True)

if args.json:
    with open(args.json, "w") as f:
        json.dump(res, f, indent=1)
print("[done]", flush=True)
