"""Measure the honest soup on the EXACT 900-window protocol of soupcmp.py / greedysoup.py,
so the de-leaked local anchor (95.4680 / 80.8027 / 96.3807 -> ?) is measured, not inferred.
Gate: the SHIPPED soup must reproduce the known 900-window triple, else the harness is wrong.
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
RE = np.array([int(n.split("_")[0]) for n in names])
VT = set(np.where(np.isin(RE, [3750, 5025, 25425, 26700]))[0].tolist())
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
va = []
for t in sorted(VT):
    for s in range(off[t], off[t] + lens[t] - 40 + 1, 10): va.append(s)
va = va[:900]
print("900-protocol val windows: %d (must be 900)" % len(va), flush=True)
assert len(va) == 900, "ABORT"
pad = lambda arr: np.concatenate([arr, np.zeros(arr.shape[:-1] + (1,), np.float32)], -1)
proto, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def sc(sd):
    proto.load_state_dict(sd); proto.eval()
    P = np.zeros((900, 20, 32, 64, 2), np.float32); T = np.zeros_like(P)
    with torch.no_grad():
        for i in range(0, 900, 16):
            sl = va[i:i + 16]
            w = np.stack([np.concatenate([np.asarray(X[s:s + 40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in sl]).astype(np.float32)
            p = (proto((torch.from_numpy(w[:, :20]).to(DEV) - MI) / SI) * ST + MT).cpu().numpy()
            P[i:i + 16] = p[..., :2]; T[i:i + 16] = w[:, 20:, ..., :2]
    return (S.score_error(float(S.rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(pad(P), pad(T)).mean())))
def load_sd(p):
    sd = torch.load(p, map_location="cpu")
    return sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
ship = sc(LB2.unpack_fp16(f"{B}/_bldfp16/sim_real_fno_fp16.pth"))
print("SHIPPED soup  @900: rel %.4f tke %.4f mvpe %.4f" % ship, flush=True)
KNOWN = (95.4680, 80.8027, 96.3807)
dev = max(abs(x - y) for x, y in zip(ship, KNOWN))
print("  GATE vs KNOWN %s -> dev %.4f" % (KNOWN, dev), flush=True)
assert dev < 0.05, "BASELINE MISMATCH -- ABORT"
print("  GATE PASSED", flush=True)
hon = sc(load_sd(f"{B}/train_es/soup_lohi_honest.pth"))
kit = sc(load_sd(f"{B}/data/comp_real/sim_real_fno.pth"))
print("HONEST soup   @900: rel %.4f tke %.4f mvpe %.4f" % hon, flush=True)
print("kit           @900: rel %.4f tke %.4f mvpe %.4f" % kit, flush=True)
print("\nMEMORISATION @900 (shipped - honest): rel %+.4f tke %+.4f mvpe %+.4f" % tuple(s - h for s, h in zip(ship, hon)), flush=True)
LIVE = (94.074271, 75.998930, 93.124176)
e = lambda s: 2.0 * (100.0 / s - 1.0)
print("\n%-8s %9s %9s %9s | %9s %9s | %s" % ("sub", "LEAKY900", "HONEST900", "LIVE", "gap_leaky", "gap_honest", "e_live/e_honest"), flush=True)
for i, k in enumerate(("rel_l2", "tke", "mvpe")):
    print("%-8s %9.4f %9.4f %9.4f | %+9.4f %+9.4f | %.4f" % (
        k, ship[i], hon[i], LIVE[i], LIVE[i] - ship[i], LIVE[i] - hon[i], e(LIVE[i]) / e(hon[i])), flush=True)
print("DONE", flush=True)
