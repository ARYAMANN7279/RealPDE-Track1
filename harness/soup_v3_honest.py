"""SOUP_v3: honest analog of the ft_all grid (w in {0.15,0.33} x lr in {1e-5,3e-5}),
but built entirely from checkpoints selected on v1's real 65/16 held-out split --
3 of the 4 points already exist as v1's own best-on-val checkpoints (w015~=w15/lr1,
lr1e5~=w33/lr1, lr3e5~=w33/lr3); only w15lr3 is newly trained, same split, same script.
Evaluated on the SAME held-out trajectories as v1's soup, for an apples-to-apples
comparison -- not on soup_v2's in-sample everything.
"""
import json, os, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG = S.SIGMA_GLOBAL; DEV = "cuda:1"; C = 2; IN = 20

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI_ = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(DEV) for x in (MI, SI_, MT, ST)]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
ntraj = len(lens); vidx = sorted(set(range(0, ntraj, 5)))     # SAME split v1 used
wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
Wall = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = Wall[:, :IN], Wall[:, IN:]; SCM = (Y[..., :C] != 0.0)
ev = np.isin(wt, vidx)          # v1's honest held-out trajectories -- untouched by ANY of these 4 checkpoints
fit = ~ev
print("windows: %d total, %d held-out (%d traj), %d fit-eligible-for-LUT" % (len(wt), ev.sum(), len(vidx), fit.sum()))

base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def run_sd(sd):
    m = base_model; m.load_state_dict(sd); m = m.to(DEV).eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o, 0).astype(np.float32)

ORIG = torch.load(f"{B}/data/comp_real/sim_real_fno.pth", map_location=DEV)
ORIG = ORIG.get("model_state_dict", ORIG)
P0 = run_sd(ORIG)

CK = {"w15lr1": "ft_w015_best.pth", "w33lr1": "ft_lr1e5_best.pth",
      "w33lr3": "ft_lr3e5_best.pth", "w15lr3": "ft_w15lr3_best.pth"}
CK = {k: f"{LH}/{v}" for k, v in CK.items() if os.path.exists(f"{LH}/{v}")}
print("members found:", list(CK.keys()))
sds = [torch.load(p, map_location=DEV) for p in CK.values()]
avg = {}
for k in sds[0]:
    v0 = sds[0][k]
    if torch.is_tensor(v0) and (v0.is_complex() or torch.is_floating_point(v0)):
        avg[k] = (sum(sd[k] for sd in sds) / len(sds)).to(v0.dtype)
    else:
        avg[k] = v0
ncplx = sum(1 for k in avg if torch.is_tensor(avg[k]) and avg[k].is_complex())
print("soup_v3: %d members, %d complex tensors preserved" % (len(sds), ncplx))
P3 = run_sd(avg)

def acc(P, m):
    dm = S.rel_l2_per_sample(P[m], Y[m], C); tk = S.tke_rel_l2_per_sample(P[m], Y[m], C)
    mv = S.mvpe_rel_l2_per_sample(P[m], Y[m])
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))

RL, TK, MV = 94.168150, 74.025866, 92.836278
n_ = lambda x: x/(0.5+x)
W_of = lambda rl, tk, mv: 0.5*(1-n_((100/rl-1)*2)) + 0.3*(1-n_((100/tk-1)*2)) + 0.2*(1-n_((100/mv-1)*2))

ao, a3 = acc(P0, ev), acc(P3, ev)
d = [a3[k]-ao[k] for k in range(3)]
print("\n%-28s %7s %7s %7s" % ("model (held-out only)", "rel_l2", "tke", "mvpe"))
print("%-28s %7.2f %7.2f %7.2f" % ("original", *ao))
print("%-28s %7.2f %7.2f %7.2f" % ("soup_v3 (honest)", *a3))
print("%-28s %+7.2f %+7.2f %+7.2f" % ("delta", *d))
print("\nfor comparison, v1's own held-out deltas (soup.py/analyze2.py): -0.30/+4.20/-0.13, -0.29/+4.27/-0.12")

RLp, TKp, MVp = RL+d[0], TK+d[1], MV+d[2]
print("\nprojected real (honest transfer basis, same method as SOUP_v1): rel_l2 %.2f tke %.2f mvpe %.2f" % (RLp, TKp, MVp))
print("W: original %.4f -> soup_v3 %.4f" % (W_of(RL, TK, MV), W_of(RLp, TKp, MVp)))

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

ERR3 = np.abs(P3[..., :2]-Y[..., :2])
fit4 = fit[:, None, None, None]
hu_star, hv_star = best_h(ERR3[..., 0][fit4 & SCM[..., 0]]), best_h(ERR3[..., 1][fit4 & SCM[..., 1]])
def E_const(ERR, m, hu, hv):
    eu = ERR[m][..., 0][SCM[m][..., 0]]; evv = ERR[m][..., 1][SCM[m][..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum() + (np.exp(-2*hv/SIG)*(evv <= hv)).sum())/(eu.size+evv.size)
print("\nTRUE bounds optimum for soup_v3 (fit on train windows, SAME method used throughout repo):")
print("  hu*=%.4f hv*=%.4f" % (hu_star, hv_star))
print("  held-out local E at these bounds: %.4f" % E_const(ERR3, ev, hu_star, hv_star))
ERR0 = np.abs(P0[..., :2]-Y[..., :2])
print("  original model, ROBUST bounds [.0129,.0098], held-out local E: %.4f" % E_const(ERR0, ev, 0.0129, 0.0098))

json.dump({"delta": d, "hu_star": hu_star, "hv_star": hv_star, "n_members": len(sds),
           "members": list(CK.keys())}, open(f"{LH}/soup_v3_honest_result.json", "w"), indent=1)
print("\nwrote soup_v3_honest_result.json")
