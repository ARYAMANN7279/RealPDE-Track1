"""Decode a downloaded RealPDEBench real-foil Arrow shard into leak-free
(input20 -> target20) validation windows at the official 32x64 eval resolution.

Each Arrow row is one full real PIV trajectory: sim_id + u/v stored as raw
binary blobs of shape (shape_t, shape_h, shape_w). We decode, subsample to
32x64 the same strided way the official pipeline does, and cut NON-OVERLAPPING
windows (stride = 40) so no target leaks into any input.
"""
import glob
import json
import os
import sys

import numpy as np
from datasets import Dataset

HARNESS = "/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
SHARD_DIR = "/Users/aryamannsrivastava/Desktop/sem7/UGP/data/real_hf/foil/hf_dataset/real"
OUT = os.path.join(HARNESS, "real_eval")
os.makedirs(OUT, exist_ok=True)

IN_STEP, OUT_STEP = 20, 20
HORIZON = IN_STEP + OUT_STEP
TARGET_H, TARGET_W = 32, 64


def decode_field(blob, t, h, w):
    n = t * h * w
    for dt in (np.float32, np.float16, np.float64):
        if len(blob) == n * np.dtype(dt).itemsize:
            return np.frombuffer(blob, dtype=dt).reshape(t, h, w).astype(np.float32)
    raise ValueError(f"cannot match blob len {len(blob)} to shape ({t},{h},{w})")


def strided_to_32x64(arr):
    # arr: (T, H, W). Reduce to (T, 32, 64) by strided subsample, matching the
    # official [::s, ::s] convention (NOT bilinear). Pick strides from the ratio.
    T, H, W = arr.shape
    sh, sw = max(1, H // TARGET_H), max(1, W // TARGET_W)
    out = arr[:, ::sh, ::sw][:, :TARGET_H, :TARGET_W]
    return out


def main():
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*.arrow")))
    if not shards:
        print("no shards downloaded yet at", SHARD_DIR)
        sys.exit(1)
    print("shards present:", [os.path.basename(s) for s in shards])

    xs, ys, meta = [], [], []
    for sp in shards:
        ds = Dataset.from_file(sp)
        for row in ds:
            t, h, w = row["shape_t"], row["shape_h"], row["shape_w"]
            sim_id = row["sim_id"]
            u = decode_field(row["u"], t, h, w)
            v = decode_field(row["v"], t, h, w)
            print(f"  {sim_id}: raw (T={t}, H={h}, W={w})")
            u = strided_to_32x64(u)
            v = strided_to_32x64(v)
            T = u.shape[0]
            n_win = (T - HORIZON) // HORIZON + 1
            for k in range(max(n_win, 0)):
                t0 = k * HORIZON
                uu, vv = u[t0:t0 + HORIZON], v[t0:t0 + HORIZON]
                pp = np.zeros_like(uu)
                traj = np.stack([uu, vv, pp], axis=-1)  # (40,32,64,3)
                xs.append(traj[:IN_STEP])
                ys.append(traj[IN_STEP:])
                meta.append({"sim_id": sim_id, "t0": int(t0)})

    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    print(f"\nbuilt {X.shape[0]} windows -> X{X.shape} Y{Y.shape}")
    np.savez(os.path.join(OUT, "inputs.npz"), input=X)
    np.savez(os.path.join(OUT, "targets.npz"), target=Y)
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
