"""Fast, exact per-channel optimum via sort (same method as best_h() in soup.py /
05_build_lut.py, not a grid search) -- SPS is separable across elements, so the
joint (hu,hv) optimum is just the two independent per-channel optima."""
import os, sys, json, torch, numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
from load_baseline import load_baseline
DEV = "cuda:0"
SIG = 0.0563870259

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
SCM = (Y[..., :2] != 0.0)
ERR = np.abs(P[..., :2] - Y[..., :2])
eu = ERR[..., 0][SCM[..., 0]]; ev = ERR[..., 1][SCM[..., 1]]
print("n_scored u=%d v=%d" % (eu.size, ev.size))
print("median u %.5f  v %.5f  (ratio %.2f)" % (np.median(eu), np.median(ev), np.median(eu)/np.median(ev)))

def best_h(v):
    v = np.sort(v)
    k = np.arange(1, v.size + 1) / v.size
    obj = np.exp(-2*v/SIG) * k
    i = np.argmax(obj)
    return float(v[i]), float(obj[i])

hu, ou = best_h(eu); hv, ov = best_h(ev)
print("\nTRUE per-channel optimum (exact, sort-based, matches best_h() used everywhere else in this repo):")
print("  hu* = %.5f  (E-contribution %.4f)" % (hu, ou))
print("  hv* = %.5f  (E-contribution %.4f)" % (hv, ov))
print("  ratio hu*/hv* = %.2f" % (hu/hv))

def E_const(hu_, hv_):
    return float((np.exp(-2*hu_/SIG)*(eu <= hu_)).sum() + (np.exp(-2*hv_/SIG)*(ev <= hv_)).sum()) / (eu.size + ev.size)

print("\nclaimed OPTIMAL [hu=0.0095, hv=0.0105] -> local E = %.4f" % E_const(0.0095, 0.0105))
print("TRUE optimum    [hu=%.4f, hv=%.4f] -> local E = %.4f" % (hu, hv, E_const(hu, hv)))
print("ROBUST-style    [hu=0.0129, hv=0.0098] on soup_v2's own errors -> local E = %.4f" % E_const(0.0129, 0.0098))
