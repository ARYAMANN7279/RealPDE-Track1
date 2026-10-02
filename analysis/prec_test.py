"""Does reduced-precision inference cost accuracy?

time_score = 100/(1+sqrt(t_neural/t_numerical)). We sit at 90.32; teams at
92.4-93.0 exist, and zhoubojian ships `v013_bf16.zip` at time 92.78. Reaching
92.4 needs ~1.7x speedup. sec7 ruled out "Time subscore optimisation" as
latency-bound, but that was about algorithmic change -- precision is not in any
ruled-out row.

This measures BOTH sides on the same windows: subscore cost, and wall-clock.
"""
import sys, os, json, time, argparse
import numpy as np, torch

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{B}/train_work_soup/sim_real_fno_fp16.pth")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--nval", type=int, default=800)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
DEV = f"cuda:{a.gpu}"; C = 2; IN = 20

MI = torch.tensor([0.154960856, -0.000513992854, 0.0])
SI = torch.tensor([0.0968056545, 0.015960684, 1.0])
MT = torch.tensor([0.154962569, -0.000517793698, 0.0])
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0])
MI, SI, MT, ST = [t.to(DEV) for t in (MI, SI, MT, ST)]

X = np.load(f"{B}/local_harness/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{B}/local_harness/tr_meta.json"))
off, lens = meta["off"], meta["lens"]
starts = []
for i in range(0, len(lens), 5):
    starts += list(range(off[i], off[i] + lens[i] - 39))
sub = np.random.default_rng(0).choice(np.array(starts), size=a.nval, replace=False)

def batch(idx):
    w = np.stack([np.asarray(X[s:s + 40]) for s in idx])
    z = np.zeros(w.shape[:-1] + (1,), np.float32)
    w = np.concatenate([w, z], -1)
    return torch.from_numpy(w[:, :IN]).to(DEV), torch.from_numpy(w[:, IN:]).to(DEV)

model, _ = load_baseline(a.ckpt, device=DEV); model = model.to(DEV).eval()

def run(mode):
    @torch.no_grad()
    def fwd(x):
        if mode == "fp32":
            return model((x - MI) / SI) * ST + MT
        dt = torch.bfloat16 if mode == "bf16" else torch.float16
        with torch.autocast("cuda", dtype=dt):
            y = model((x - MI) / SI)
        return y.float() * ST + MT
    # warm-up
    x, _ = batch(sub[:a.bs]); [fwd(x) for _ in range(3)]
    torch.cuda.synchronize()
    P, T = [], []; t0 = time.perf_counter()
    for i in range(0, len(sub), a.bs):
        x, y = batch(sub[i:i + a.bs])
        P.append(fwd(x).cpu().numpy()); T.append(y.cpu().numpy())
    torch.cuda.synchronize(); el = time.perf_counter() - t0
    P = np.concatenate(P, 0).astype(np.float32); T = np.concatenate(T, 0).astype(np.float32)
    return (S.score_error(float(S.rel_l2_per_sample(P, T, C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(P, T, C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(P, T).mean())),
            1000 * el / len(sub))

print("ckpt: %s   n=%d  bs=%d" % (os.path.basename(a.ckpt), a.nval, a.bs), flush=True)
print("%-6s %8s %8s %8s %10s %9s" % ("mode", "rel_l2", "tke", "mvpe", "ms/sample", "speedup"), flush=True)
base = None
for m in ["fp32", "bf16", "fp16"]:
    try:
        r = run(m)
    except Exception as e:
        print("%-6s FAILED: %s" % (m, str(e)[:60]), flush=True); continue
    if base is None: base = r[3]
    print("%-6s %8.2f %8.2f %8.2f %10.3f %8.2fx" % (m, r[0], r[1], r[2], r[3], base / r[3]), flush=True)
