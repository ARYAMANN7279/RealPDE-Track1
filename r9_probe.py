import os, sys, time, json
import numpy as np
from scipy.stats import spearmanr
import importlib.util as iu
import torch

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"

sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp)
sp.loader.exec_module(S)

sys.path.insert(0, f"{B}/_r9")
from submission import predict

meta = json.load(open(f"{LH}/tr_meta.json"))
off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())

X32 = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
starts_va = []
for i in range(ntraj):
    for t0 in range(off[i], off[i]+lens[i]-39):
        if i in VT:
            starts_va.append(t0)
starts_va = np.array(starts_va)
rng = np.random.default_rng(0)
va_sub = rng.choice(starts_va, size=min(900, len(starts_va)), replace=False)

def batch_va(idx):
    w = np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z = np.zeros(w.shape[:-1]+(1,), np.float32)
    w = np.concatenate([w, z], -1).astype(np.float32)
    return w[:,:20], w[:,20:]

SIGMA = 0.0563870259

chunk_size = 120
P_list, L_list, U_list, T_list = [], [], [], []

fallback_u = 0.0129
fallback_v = 0.0098

n_chunks = 0
for i in range(0, len(va_sub), chunk_size):
    x, y = batch_va(va_sub[i:i+chunk_size])
    res = predict(x)
    pred, lower, upper = res["prediction"], res["lower"], res["upper"]
    
    h_u = (upper[..., 0] - lower[..., 0]) / 2.0
    h_v = (upper[..., 1] - lower[..., 1]) / 2.0
    
    match_u = np.isclose(h_u, fallback_u, atol=1e-6)
    match_v = np.isclose(h_v, fallback_v, atol=1e-6)
    match = match_u & match_v
    if match.mean() > 0.01:
        print(f"ERROR: Fallback detected in chunk {i} - {match.mean()*100:.2f}% elements match fallback widths [0.0129, 0.0098]!")
        sys.exit(1)
        
    P_list.append(pred)
    L_list.append(lower)
    U_list.append(upper)
    T_list.append(y)
    n_chunks += 1

print(f"Processed {n_chunks} chunks, trap-B fallback assertion passed on all.", flush=True)

P = np.concatenate(P_list, 0)
L = np.concatenate(L_list, 0)
U = np.concatenate(U_list, 0)
T = np.concatenate(T_list, 0)

scored_mask = (T[..., :2] != 0.0)
res_dict = {"per_channel": {}}

for c, ch in enumerate(["u", "v"]):
    print(f"Computing metrics for channel {ch}...", flush=True)
    mask_c = scored_mask[..., c]
    tc = T[..., c][mask_c]
    pc = P[..., c][mask_c]
    lc = L[..., c][mask_c]
    uc = U[..., c][mask_c]
    
    e = np.abs(tc - pc)
    h = (uc - lc) / 2.0
    nil = (uc - lc) / SIGMA
    inside = (tc >= lc) & (tc <= uc)
    
    coverage = inside.mean()
    g_ship = (np.exp(-nil) * inside).mean()
    
    h_mean = h.mean()
    h_median = np.median(h)
    
    print(f"  spearmanr...", flush=True)
    spearman_h_e, _ = spearmanr(h, e)
    
    h_shuffled = rng.permutation(h)
    centre = (lc + uc) / 2.0
    uc_s = centre + h_shuffled
    lc_s = centre - h_shuffled
    inside_s = (tc >= lc_s) & (tc <= uc_s)
    nil_s = (uc_s - lc_s) / SIGMA
    g_shuffled = (np.exp(-nil_s) * inside_s).mean()
    
    e_sort = np.argsort(e)
    h_sorted = np.sort(h)
    h_ranked = np.empty_like(h)
    h_ranked[e_sort] = h_sorted
    uc_r = centre + h_ranked
    lc_r = centre - h_ranked
    inside_r = (tc >= lc_r) & (tc <= uc_r)
    nil_r = (uc_r - lc_r) / SIGMA
    g_ranked = (np.exp(-nil_r) * inside_r).mean()
    
    g_oracle = np.exp(-2 * e / SIGMA).mean()
    
    h_consts = np.linspace(0.0005, 0.06, 3000)
    best_g_const = -1.0
    best_h_const = -1.0
    
    print(f"  g_const sweep...", flush=True)
    d_c_gpu = torch.from_numpy(np.abs(tc - centre)).cuda()
    
    for hc in h_consts:
        cov = (d_c_gpu <= hc).float().mean().item()
        gc = np.exp(-2 * hc / SIGMA) * cov
        if gc > best_g_const:
            best_g_const = gc
            best_h_const = hc
            
    res_dict["per_channel"][ch] = {
        "g_ship": float(g_ship),
        "g_shuffled": float(g_shuffled),
        "g_ranked": float(g_ranked),
        "g_oracle": float(g_oracle),
        "g_const": float(best_g_const),
        "h_const": float(best_h_const),
        "coverage": float(coverage),
        "h_mean": float(h_mean),
        "h_median": float(h_median),
        "spearman_h_e": float(spearman_h_e)
    }

ew = {}
for k in ["g_ship", "g_shuffled", "g_ranked", "g_oracle"]:
    ew[k] = (res_dict["per_channel"]["u"][k] + res_dict["per_channel"]["v"][k]) / 2.0
    
res_dict["equal_weight"] = ew
res_dict["n_windows"] = len(va_sub)
res_dict["n_scored_u"] = int(scored_mask[..., 0].sum())
res_dict["n_scored_v"] = int(scored_mask[..., 1].sum())

if not (ew["g_shuffled"] <= ew["g_ship"] + 1e-6 and ew["g_ship"] <= ew["g_ranked"] + 1e-6 and ew["g_ranked"] <= ew["g_oracle"] + 1e-6):
    print("ERROR: Information ladder assertion failed!")
    print(f"shuffled: {ew['g_shuffled']}, ship: {ew['g_ship']}, ranked: {ew['g_ranked']}, oracle: {ew['g_oracle']}")
    sys.exit(1)

print(f"Computing kit SPS...", flush=True)
weighted, sps_cov = S.aggregate_sps(P, T, c=2, lower=L, upper=U)
res_dict["kit_sps"] = {
    "weighted": float(weighted),
    "coverage": float(sps_cov),
    "score_sps": float(S.score_sps(weighted))
}

nu = float(res_dict["n_scored_u"])
nv = float(res_dict["n_scored_v"])
cu = res_dict["per_channel"]["u"]["coverage"]
cv = res_dict["per_channel"]["v"]["coverage"]
cov_ship = (cu * nu + cv * nv) / (nu + nv)

if abs(cov_ship - sps_cov) > 1e-6:
    print(f"ERROR: Coverage mismatch! 3.1={cov_ship}, kit={sps_cov}")
    sys.exit(1)

print(json.dumps(res_dict, indent=2))
with open(f"{B}/r9_results.json", "w") as f:
    json.dump(res_dict, f, indent=2)
