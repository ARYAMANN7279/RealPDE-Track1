"""Sweep a GLOBAL multiplier on the shipped LUT half-widths.

sps is our biggest gap (34.37 vs ~43.9 at the top = +2.07 final). The LUT's
half-widths come from best_h() on locally-scaled errors; if that scale is off,
every half-width is systematically too tight or too wide and a single global
multiplier recovers the loss. A global multiplier cannot reorder elements, so it
is safe under the GATE 4B rule that killed v6.

Evaluated on trajectory-disjoint held-out windows (stride 5), the same split
SOUP_v1/v3 used. Reports the local optimum and, importantly, WHICH SIDE of 1.0
it falls on: memory 5B says the harness grows optimistic as bounds tighten, so a
multiplier >1 (wider) is a conservative signal that should transfer, while <1
(tighter) is exactly the regime the harness is known to over-promise in.
"""
import json, os, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sys.path.insert(0, f"{B}/train_maxsoup")
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
from head_common import feats, Net
SIG = S.SIGMA_GLOBAL; DEV = "cuda:0" if torch.cuda.is_available() else "cpu"; C = 2

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(DEV) for x in (MI, SI, MT, ST)]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
ntraj = len(lens); vidx = sorted(set(range(0, ntraj, 5)))
wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
W = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = W[:, :20], W[:, 20:]; SCM = (Y[..., :C] != 0.0)
ev = np.isin(wt, vidx)
print("held-out windows: %d of %d" % (ev.sum(), len(wt)))

m, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
sd = torch.load(f"{LH}/soup_final_candidate.pth", map_location=DEV)
m.load_state_dict(sd); m = m.to(DEV).eval()
P = []
with torch.no_grad():
    for i in range(0, len(Xin), 32):
        xb = torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
        P.append((m((xb - mi) / si) * st + mt).cpu().numpy())
P = np.concatenate(P, 0).astype(np.float32)
# CRITICAL: 05_build_lut.py fits best_h() on errors multiplied by the per-channel
# global_scale, so the LUT's half-widths live in REAL-error units. Comparing them
# against raw local errors makes the bounds look far too wide (local coverage 0.96
# vs the ~0.73 actually observed) and invents a spurious "tighter is better"
# optimum. Scale the errors the same way before evaluating.
from head_common import global_scale
ERR = np.abs(P[..., :C] - Y[..., :C]).astype(np.float32)
SCALE = global_scale(ERR, SCM)
for _ci in range(C):
    ERR[..., _ci] *= SCALE[_ci]
print("global per-channel scale applied: u x%.3f  v x%.3f" % (SCALE[0], SCALE[1]))

z = np.load(f"{B}/train_work_maxsoup/head_assets.npz")
NF = int(z["nf"]); LUT = z["LUT"]; ED = z["ED"]
net = Net(NF).to(DEV)
net.load_state_dict({k[2:]: torch.from_numpy(z["w_" + k[2:]]) for k in z.files if k.startswith("w_")})
net.eval()
FT = (feats(P) - z["fmu"]) / z["fsd"]
MU = []
with torch.no_grad():
    for i in range(0, len(P), 8):
        f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV).permute(0,1,4,2,3).reshape(-1, NF, 32, 64)
        MU.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
MU = np.concatenate(MU, 0)

H = np.empty_like(ERR)
for ci in range(C):
    H[..., ci] = LUT[np.digitize(MU[..., ci], ED[ci]), ci]

def E_of(mult):
    h = H[ev] * mult; e = ERR[ev]; sc = SCM[ev]
    return float((np.exp(-2*h/SIG) * (e <= h) * sc).sum() / sc.sum())

def cov_of(mult):
    h = H[ev] * mult; e = ERR[ev]; sc = SCM[ev]
    return float(((e <= h) & sc).sum() / sc.sum())

# real anchor: SOUP_v1 shipped LUT -> real sps 34.3656, W 0.6845 -> real E 0.5020
W_REAL = 0.684538
print("\n%-8s %-10s %-10s %-12s" % ("mult", "local E", "coverage", "d_sps(if transfers)"))
base = E_of(1.0)
rows = []
for mult in [0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.3,1.5]:
    e = E_of(mult); c = cov_of(mult)
    rows.append((mult, e, c))
    print("%-8.2f %-10.4f %-10.3f %+12.2f" % (mult, e, c, (e - base) * 100 * W_REAL))
best = max(rows, key=lambda r: r[1])
print("\nlocal optimum: mult=%.2f  E=%.4f  (base mult=1.00 E=%.4f, delta %+.4f)"
      % (best[0], best[1], base, best[1] - base))
print("d_final if it transfers 1:1: %+.3f" % ((best[1] - base) * 100 * W_REAL * 0.217))
if best[0] < 1.0:
    print("\n*** optimum is TIGHTER than shipped. Memory 5B: the harness is known to")
    print("    OVER-predict E as bounds tighten (it over-promised by +0.029 at the tight")
    print("    anchor). Treat this as unreliable and do NOT ship it on local evidence.")
elif best[0] > 1.0:
    print("\n optimum is WIDER than shipped -- the conservative direction, which the")
    print(" harness's known bias works AGAINST, so a local gain here is credible.")
else:
    print("\n shipped scale is already optimal; no free gain here.")
