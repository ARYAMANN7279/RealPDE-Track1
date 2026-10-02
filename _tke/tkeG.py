"""Confirm the memorisation effect on aoa15 -- a 2x2 difference-in-differences with
IDENTICAL model architecture/recipe on both sides, so nothing but the holdout differs.

            eval aoa15        eval lohi664
 s_lohi     IN-SAMPLE         HELD OUT
 s_aoa      HELD OUT          IN-SAMPLE
 shipped    in-sample         in-sample

Same recipe (finetune2.py lr1e-5 8k wtke0.15), only --split differs.
"""
import os, sys, json, argparse, importlib.util as iu
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"; LH = f"{B}/local_harness"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py")); S = iu.module_from_spec(sp); sp.loader.exec_module(S)
sys.path.insert(0, f"{B}/_bld")
from load_baseline import load_baseline
lb = iu.spec_from_file_location("lb2", f"{B}/_bld/load_baseline.py"); LB2 = iu.module_from_spec(lb); lb.loader.exec_module(LB2)
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
DEV = f"cuda:{a.gpu}"; C = 2
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV); SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV); ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)
meta = json.load(open(f"{LH}/tr_meta.json")); off, lens, names = meta["off"], meta["lens"], meta["names"]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
wins, wt = [], []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 10): wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
RE = np.array([int(names[i].split("_")[0]) for i in wt])
AOA = np.array([int(names[i].split("_", 1)[1].replace(".h5", "")) for i in wt])
assert len(wins) == 6602, "ABORT"
SPL = {"lohi664": np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0][::2],
       "aoa15":   np.where(AOA == 15)[0][::2]}
for k, v in SPL.items(): print("split %-8s %5d windows" % (k, len(v)), flush=True)
assert len(SPL["lohi664"]) == 664, "ABORT"
pad = lambda arr: np.concatenate([arr, np.zeros(arr.shape[:-1] + (1,), np.float32)], -1)
proto, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def load_sd(p):
    sd = torch.load(p, map_location="cpu")
    return sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
def sc(sd, idx):
    proto.load_state_dict(sd); proto.eval()
    P = np.zeros((len(idx), 20, 32, 64, 2), np.float32); T = np.zeros_like(P)
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            sl = idx[i:i + 16]
            w = np.stack([np.concatenate([np.asarray(X[s:s + 40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins[sl]]).astype(np.float32)
            p = (proto((torch.from_numpy(w[:, :20]).to(DEV) - MI) / SI) * ST + MT).cpu().numpy()
            P[i:i + 16] = p[..., :2]; T[i:i + 16] = w[:, 20:, ..., :2]
    return (S.score_error(float(S.rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(pad(P), pad(T)).mean())))
M = {"shipped": LB2.unpack_fp16(f"{B}/_bldfp16/sim_real_fno_fp16.pth"),
     "s_lohi(holds out re_lohi)": load_sd(f"{B}/train_mvpe/runs/s_lohi_besteff.pth"),
     "s_aoa (holds out aoa15)":   load_sd(f"{B}/train_mvpe/runs/s_aoa_besteff.pth"),
     "kit":                       load_sd(f"{B}/data/comp_real/sim_real_fno.pth")}
R = {}
for sn, idx in SPL.items():
    print("\n--- eval on %s ---" % sn, flush=True)
    for mn, sd in M.items():
        r = sc(sd, idx); R[(sn, mn)] = r
        print("  %-26s rel %8.4f  tke %8.4f  mvpe %8.4f" % (mn, *r), flush=True)
print("\n=== 2x2: tke of each honest fine-tune, IN-SAMPLE vs HELD-OUT condition ===", flush=True)
sl = "s_lohi(holds out re_lohi)"; sa = "s_aoa (holds out aoa15)"
print("  s_lohi : aoa15 (in-sample) %.4f  vs  lohi664 (HELD OUT) %.4f   -> %+.4f" % (
    R[("aoa15", sl)][1], R[("lohi664", sl)][1], R[("lohi664", sl)][1] - R[("aoa15", sl)][1]), flush=True)
print("  s_aoa  : lohi664 (in-sample) %.4f  vs  aoa15 (HELD OUT) %.4f   -> %+.4f" % (
    R[("lohi664", sa)][1], R[("aoa15", sa)][1], R[("aoa15", sa)][1] - R[("lohi664", sa)][1]), flush=True)
print("\n=== memorisation delta vs the SHIPPED soup, same windows ===", flush=True)
for sn in SPL:
    for mn in (sl, sa):
        held = ("HELD OUT" if ((sn == "lohi664" and mn == sl) or (sn == "aoa15" and mn == sa)) else "in-sample")
        print("  %-8s shipped - %-26s d_tke %+7.4f   (%s for that model)" % (
            sn, mn, R[(sn, "shipped")][1] - R[(sn, mn)][1], held), flush=True)
print("DONE", flush=True)
