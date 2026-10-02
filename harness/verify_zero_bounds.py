"""Verify whether h_u=0 / h_v=0 locations in the per-location bounds map are
benign (permanently masked, target==0 in EVERY window, so SCM gates them out
of scoring regardless of bound width) or a live corrupted-ranking bug (target
IS nonzero somewhere for that pixel, meaning a real scored element got h=0,
which is a guaranteed miss -- the actual v6 failure mode)."""
import json, os, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
vidx = sorted(set(range(0, len(lens), 5)))
wins = []
for i in vidx:
    for t0 in range(off[i], off[i] + lens[i] - 39, 20): wins.append(t0)
Wall = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Y = Wall[:, 20:]
SCM = (Y[..., :2] != 0.0)  # (n, 20, 32, 64, 2)

HL = np.load(f"{LH}/soup_final_candidate_bounds.npy")  # already x1.15, shape (32,64,2)
print("HL shape", HL.shape)

for ci, name in [(0, "u"), (1, "v")]:
    zero_mask = HL[..., ci] == 0.0
    n_zero = int(zero_mask.sum())
    print("\nchannel %s: %d / %d pixels have h==0" % (name, n_zero, HL[...,ci].size))
    if n_zero == 0:
        continue
    # For each zero pixel, check across ALL held-out windows/timesteps: is SCM ever True there?
    scm_at_zero = SCM[:, :, zero_mask, ci] if False else None
    # SCM shape (n,20,32,64,2); index the spatial zero_mask on axes (2,3)
    scm_ci = SCM[..., ci]  # (n,20,32,64)
    # reshape zero_mask (32,64) -> boolean index over last two axes
    scored_ever = np.zeros(zero_mask.sum(), dtype=bool)
    idxs = np.argwhere(zero_mask)
    max_scored_frac = 0.0
    any_live = False
    for k, (i, j) in enumerate(idxs):
        frac = scm_ci[:, :, i, j].mean()
        max_scored_frac = max(max_scored_frac, frac)
        if frac > 0:
            any_live = True
    print("  max fraction of (window,timestep) pairs where a zero-bound pixel IS scored (target!=0): %.6f" % max_scored_frac)
    print("  ANY zero-bound pixel ever scored (live corruption risk):", any_live)
    if not any_live:
        print("  -> BENIGN: every h==0 pixel is masked (target==0) in 100%% of held-out data, so SCM excludes it from SPS regardless of bound width.")
    else:
        print("  !! LIVE ISSUE: some zero-bound pixels ARE sometimes scored -- this would guarantee misses on those instances.")

# cross-check: does v1's OWN real, 78.45-scoring bounds map have the same pattern?
v1_path = f"{LH}/soup_bounds_v1_recheck.npy"
if os.path.exists(v1_path):
    HL1 = np.load(v1_path)
    print("\n--- v1's own (freshly rebuilt, same method) bounds map for comparison ---")
    for ci, name in [(0, "u"), (1, "v")]:
        n_zero = int((HL1[..., ci] == 0.0).sum())
        print("  v1-style channel %s: %d / %d pixels have h==0 (min=%.5f, max=%.5f)" % (
            name, n_zero, HL1[...,ci].size, HL1[...,ci].min(), HL1[...,ci].max()))
