"""tke TRANSFER FORENSICS -- experiment E: what does SOUPING buy, honestly?

sec11A recorded "tke transfer rate 46%" from local +4.20 vs live +1.97.  The +4.20 was
the SHIPPED (leaky) soup's gain over the kit on re_lohi.  Measure the HONEST gain, and
the per-member spread, so the soup route can be priced for any architecture.

Also prints the corrected local->live arithmetic in ERROR space.
MANDATORY: gate on the honest-soup baseline; ABORT on mismatch.
"""
import os, sys, json, argparse, importlib.util as iu
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"; LH = f"{B}/local_harness"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
sys.path.insert(0, f"{B}/_bld")
from load_baseline import load_baseline
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args(); DEV = f"cuda:{a.gpu}"; C = 2
MI = torch.tensor([0.154960856, -0.000513992854, 0.0]).to(DEV)
SI = torch.tensor([0.0968056545, 0.015960684, 1.0]).to(DEV)
MT = torch.tensor([0.154962569, -0.000517793698, 0.0]).to(DEV)
ST = torch.tensor([0.0968104079, 0.0159636438, 1.0]).to(DEV)
meta = json.load(open(f"{LH}/tr_meta.json")); off, lens, names = meta["off"], meta["lens"], meta["names"]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
wins, wt = [], []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 10): wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
RE = np.array([int(names[i].split("_")[0]) for i in wt])
assert len(wins) == 6602, "ABORT"
HOLD = [3750, 5025, 25425, 26700]
IDX = np.where(np.isin(RE, HOLD))[0][::2]
print("lohi664 windows: %d" % len(IDX), flush=True); assert len(IDX) == 664, "ABORT"
pad = lambda arr: np.concatenate([arr, np.zeros(arr.shape[:-1] + (1,), np.float32)], -1)
proto, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def load_sd(p):
    sd = torch.load(p, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    return sd
def sc(sd):
    proto.load_state_dict(sd); proto.eval()
    P = np.zeros((len(IDX), 20, 32, 64, 2), np.float32); T = np.zeros_like(P)
    with torch.no_grad():
        for i in range(0, len(IDX), 16):
            sl = IDX[i:i + 16]
            w = np.stack([np.concatenate([np.asarray(X[s:s + 40]),
                          np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins[sl]]).astype(np.float32)
            p = (proto((torch.from_numpy(w[:, :20]).to(DEV) - MI) / SI) * ST + MT).cpu().numpy()
            P[i:i + 16] = p[..., :2]; T[i:i + 16] = w[:, 20:, ..., :2]
    return (S.score_error(float(S.rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(pad(P), pad(T), C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(pad(P), pad(T)).mean())))
HON = load_sd(f"{B}/train_es/soup_lohi_honest.pth")
b0 = sc(HON); KNOWN = (95.3069, 78.7002, 96.0540)
print("GATE honest soup: rel %.4f tke %.4f mvpe %.4f" % b0, flush=True)
assert max(abs(x - y) for x, y in zip(b0, KNOWN)) < 0.02, "BASELINE MISMATCH -- ABORT"
print("GATE PASSED\n", flush=True)
kit = sc(load_sd(f"{B}/data/comp_real/sim_real_fno.pth"))
print("kit sim_real_fno      : rel %.4f tke %.4f mvpe %.4f" % kit, flush=True)
RUNS = f"{B}/train_mvpe/runs"; MEM = ["s_lohi", "L_lr3e5", "L_lr3e6", "L_s16k", "L_s4k", "L_w30"]
tk = []
for m in MEM:
    r = sc(load_sd(f"{RUNS}/{m}_besteff.pth")); tk.append(r[1])
    print("member %-9s      : rel %.4f tke %.4f mvpe %.4f   (vs kit d_tke %+.4f)" % (m, *r, r[1] - kit[1]), flush=True)
print("\nSOUP vs members: soup tke %.4f | best member %.4f | mean member %.4f" % (b0[1], max(tk), float(np.mean(tk))), flush=True)
print("  souping buys over BEST member : %+.4f tke" % (b0[1] - max(tk)), flush=True)
print("  souping buys over MEAN member : %+.4f tke" % (b0[1] - float(np.mean(tk))), flush=True)
print("  fine-tune+soup over kit       : %+.4f tke  (sec11A recorded +4.20 using the LEAKY soup)" % (b0[1] - kit[1]), flush=True)
e = lambda s: 2.0 * (100.0 / s - 1.0)
LIVE = dict(rel_l2=94.074271, tke=75.998930, mvpe=93.124176)
print("\n=== corrected local->live arithmetic (honest lohi664 -> live FP16) ===", flush=True)
print("%-8s %10s %10s %10s %10s %10s" % ("sub", "local", "live", "d_pts", "e_local", "e_live/e_local"), flush=True)
for i, k in enumerate(("rel_l2", "tke", "mvpe")):
    print("%-8s %10.4f %10.4f %+10.4f %10.5f %10.4f" % (
        k, b0[i], LIVE[k], LIVE[k] - b0[i], e(b0[i]), e(LIVE[k]) / e(b0[i])), flush=True)
print("\nLIVE soup gain over kit (real anchors, sec1): rel -0.12  tke +1.97  mvpe +0.03", flush=True)
print("HONEST LOCAL soup gain over kit             : rel %+.4f  tke %+.4f  mvpe %+.4f" % (
    b0[0] - kit[0], b0[1] - kit[1], b0[2] - kit[2]), flush=True)
print("=> tke GAIN transfer rate = %.1f%%" % (100 * 1.97 / (b0[1] - kit[1])), flush=True)
print("DONE", flush=True)
