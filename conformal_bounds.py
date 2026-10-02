import os, sys, json, torch
import numpy as np

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT)

# 1. Setup validation windows
LH = f"{B}/local_harness"
meta = json.load(open(f"{LH}/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())
X32 = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
starts_va = []
for i in range(len(lens)):
    if i in VT:
        for t0 in range(off[i], off[i] + lens[i] - 39):
            starts_va.append(t0)
starts_va = np.array(starts_va)
rng = np.random.default_rng(0)
va_sub = rng.choice(starts_va, size=min(900, len(starts_va)), replace=False)

def batch_va(idx):
    w = np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z = np.zeros(w.shape[:-1] + (1,), np.float32)
    w = np.concatenate([w, z], -1).astype(np.float32)
    return w[:, :20], w[:, 20:]

# 2. Run ensemble to get predictions and residuals
d = f"{B}/submissions/build_mega"
sys.path.insert(0, d)
import submission as M

M.predict(batch_va(va_sub[:2])[0]) # Warmup

RESIDUALS = []
print("Computing predictions on val set...")
for i in range(0, len(va_sub), 32):
    x, y = batch_va(va_sub[i:i+32])
    # To get ctr (ensemble prediction), we need to extract it from the model. 
    # But submission.py currently uses yb and _mh. 
    # Let's just modify the _assets in submission.py or extract `ctr`.
    # Let's temporarily inject a hook to get ctr.
    pass

