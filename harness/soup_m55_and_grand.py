"""Soup the 3 M55-init (sim-only pretrained) checkpoints, evaluate on the shared
17-traj holdout, then build a GRAND soup mixing ALL 12 checkpoints found today
(6 v1 originals + 3 real-init long runs + 3 sim-init M55 runs) as the final,
maximal-diversity check before declaring the ceiling final."""
import json, os, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG = S.SIGMA_GLOBAL; DEV = "cuda:0"; C = 2; IN = 20

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI_ = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(DEV) for x in (MI, SI_, MT, ST)]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off = meta["off"]; lens = meta["lens"]
ntraj = len(lens); vidx = sorted(set(range(0, ntraj, 5)))
wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
Wall = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = Wall[:, :IN], Wall[:, IN:]; SCM = (Y[..., :C] != 0.0)
ev = np.isin(wt, vidx); fit = ~ev
print("windows: %d total, %d held-out (%d traj), %d fit" % (len(wt), ev.sum(), len(vidx), fit.sum()))

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

def acc(P, m):
    dm = S.rel_l2_per_sample(P[m], Y[m], C); tk = S.tke_rel_l2_per_sample(P[m], Y[m], C)
    mv = S.mvpe_rel_l2_per_sample(P[m], Y[m])
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))

RL, TK, MV = 94.168150, 74.025866, 92.836278
n_ = lambda x: x/(0.5+x)
W_of = lambda rl, tk, mv: 0.5*(1-n_((100/rl-1)*2)) + 0.3*(1-n_((100/tk-1)*2)) + 0.2*(1-n_((100/mv-1)*2))
ao = acc(P0, ev)

def soup_of(keys, tags):
    tags = {k: v for k, v in tags.items() if k in keys and os.path.exists(v)}
    sds = [torch.load(v, map_location=DEV) for v in tags.values()]
    avg = {k: (sum(sd[k] for sd in sds)/len(sds)).to(sds[0][k].dtype)
           if torch.is_tensor(sds[0][k]) and (sds[0][k].is_complex() or torch.is_floating_point(sds[0][k]))
           else sds[0][k] for k in sds[0]}
    return avg, list(tags.keys())

def report(name, sd):
    P = run_sd(sd)
    a = acc(P, ev)
    d = [a[k]-ao[k] for k in range(3)]
    dfin = 0.306*d[0] + 0.163*d[1] + 0.218*d[2]
    print("%-28s %7.2f %7.2f %7.2f %+8.3f" % (name, a[0], a[1], a[2], dfin))
    return a, d, P

ALL = {
    "w005": f"{LH}/ft_w005_best.pth", "w010": f"{LH}/ft_w010_best.pth", "w015": f"{LH}/ft_w015_best.pth",
    "lr1e5": f"{LH}/ft_lr1e5_best.pth", "w060": f"{LH}/ft_w060_best.pth", "lr3e5": f"{LH}/ft_lr3e5_best.pth",
    "long_w15lr3": f"{LH}/ft_long_w15lr3_best.pth", "long_w20lr2": f"{LH}/ft_long_w20lr2_best.pth", "long_w10lr1": f"{LH}/ft_long_w10lr1_best.pth",
    "m55_w15lr3": f"{LH}/ft_m55_w15lr3_best.pth", "m55_w15lr4": f"{LH}/ft_m55_w15lr4_best.pth", "m55_w20lr5": f"{LH}/ft_m55_w20lr5_best.pth",
}
present = {k: v for k, v in ALL.items() if os.path.exists(v)}
print("present:", list(present.keys()))

print("\n%-28s %7s %7s %7s %8s" % ("recipe", "rel_l2", "tke", "mvpe", "d_final~"))
sd_m55, keys_m55 = soup_of(["m55_w15lr3", "m55_w15lr4", "m55_w20lr5"], present)
report("M55-init soup (%d)" % len(keys_m55), sd_m55)

sd_grand, keys_grand = soup_of(list(present.keys()), present)
a_g, d_g, P_g = report("GRAND soup (ALL %d)" % len(keys_grand), sd_grand)

sd9, keys9 = soup_of(["w005","w010","w015","lr1e5","w060","lr3e5","long_w15lr3","long_w20lr2","long_w10lr1"], present)
a9, d9, P9 = report("real-init-only soup (%d, =yesterday's MAXIMAL)" % len(keys9), sd9)

print("\nreference plateau: +4.20/+4.27/+4.24/+4.11-4.13 tke (all prior soups)")

# pick whichever soup has the best final delta
cands = [("GRAND soup", sd_grand, d_g, P_g), ("real-init-only (9)", sd9, d9, P9)]
best_name, best_sd, best_d, best_P = max(cands, key=lambda c: 0.306*c[2][0]+0.163*c[2][1]+0.218*c[2][2])
print("\nWINNER: %s -- rel_l2 %+.2f tke %+.2f mvpe %+.2f" % (best_name, best_d[0], best_d[1], best_d[2]))
torch.save(best_sd, f"{LH}/soup_final_candidate.pth")

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
ERR = np.abs(best_P[..., :2]-Y[..., :2])
fit4 = fit[:, None, None, None]
hu_star, hv_star = best_h(ERR[..., 0][fit4 & SCM[..., 0]]), best_h(ERR[..., 1][fit4 & SCM[..., 1]])
def E_const(ERRx, m, hu, hv):
    eu = ERRx[m][..., 0][SCM[m][..., 0]]; evv = ERRx[m][..., 1][SCM[m][..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum() + (np.exp(-2*hv/SIG)*(evv <= hv)).sum())/(eu.size+evv.size)
E_here = E_const(ERR, ev, hu_star, hv_star)
print("TRUE bounds optimum: hu*=%.4f hv*=%.4f  held-out local E=%.4f" % (hu_star, hv_star, E_here))

RLp, TKp, MVp = RL+best_d[0], TK+best_d[1], MV+best_d[2]
Wp = W_of(RLp, TKp, MVp)
print("\nprojected real (delta-on-anchor method): rel_l2 %.2f tke %.2f mvpe %.2f  W=%.4f" % (RLp, TKp, MVp, Wp))
json.dump({"name": best_name, "delta": best_d, "hu_star": hu_star, "hv_star": hv_star,
           "RL_proj": RLp, "TK_proj": TKp, "MV_proj": MVp, "W": Wp, "members": keys_grand if best_name=="GRAND soup" else keys9},
          open(f"{LH}/soup_final_candidate_result.json", "w"), indent=1)
print("\nwrote soup_final_candidate.pth + result json")
