import os, sys, json, torch, numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline
DEV = "cuda:0"
print(); print("="*70); print("PART 2 -- independent hu/hv optimum check on soup_v2 (in-sample)"); print("="*70)
model, _ = load_baseline("/tmp/sim_real_fno_soupv2probe.pth", device=DEV)
sd = torch.load(f"{LH}/soup_v2.pth", map_location=DEV)
model.load_state_dict(sd.get("model_state_dict", sd)); model = model.to(DEV).eval()

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(DEV) for x in (MI, SI, MT, ST)]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
wins = []
for i in range(len(lens)):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0)
wins = np.array(wins)
W = np.stack([np.concatenate([np.asarray(X[s:s + 40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = W[:, :20], W[:, 20:]
P = []
with torch.no_grad():
    for i in range(0, len(Xin), 32):
        xb = torch.from_numpy(Xin[i:i+32]).to(DEV)
        yb = (model((xb - mi) / si) * st + mt).cpu().numpy()
        P.append(yb)
P = np.concatenate(P, 0)
SIG = 0.0563870259
SCM = (Y[..., :2] != 0.0)
ERR = np.abs(P[..., :2] - Y[..., :2])
print("raw abs-error median  u: %.5f  v: %.5f  (ratio u/v %.2f)" %
      (np.median(ERR[..., 0][SCM[..., 0]]), np.median(ERR[..., 1][SCM[..., 1]]),
       np.median(ERR[..., 0][SCM[..., 0]]) / np.median(ERR[..., 1][SCM[..., 1]])))

def E_const(hu, hv):
    eu = ERR[..., 0][SCM[..., 0]]; ev = ERR[..., 1][SCM[..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum() + (np.exp(-2*hv/SIG)*(ev <= hv)).sum()) / (eu.size + ev.size)

claimed = E_const(0.0095, 0.0105)
print("\nclaimed OPTIMAL [hu=0.0095, hv=0.0105] -> local E = %.4f" % claimed)

best = None
for hu in np.linspace(0.005, 0.020, 61):
    for hv in np.linspace(0.002, 0.014, 61):
        e = E_const(hu, hv)
        if best is None or e > best[0]:
            best = (e, hu, hv)
print("independently-found optimum: hu=%.4f hv=%.4f -> E=%.4f" % (best[1], best[2], best[0]))
print("(grid: hu in [0.005,0.020], hv in [0.002,0.014], step ~0.00025/0.0002, 61x61=3721 points)")

# also report E at the analogous ORIGINAL-model-shaped ratio for comparison
print("\nfor reference, ROBUST used hu=0.0129 hv=0.0098 (hv < hu, ratio %.2f); on soup_v2's OWN errors that gives E=%.4f"
      % (0.0129/0.0098, E_const(0.0129, 0.0098)))
