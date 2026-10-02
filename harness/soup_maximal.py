"""Maximal-diversity soup: v1's original 6 members + today's 3 longer-trained
checkpoints (long_w15lr3/w20lr2/w10lr1), all sharing the SAME stride=5/17-traj
holdout, so this is a fully honest, apples-to-apples test of whether more
diversity clears the established +4.20-4.27 tke plateau."""
import json, os, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG = S.SIGMA_GLOBAL; DEV = "cuda:3"; C = 2; IN = 20

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

ALL_TAGS = {
    "w005": "ft_w005_best.pth", "w010": "ft_w010_best.pth", "w015": "ft_w015_best.pth",
    "lr1e5": "ft_lr1e5_best.pth", "w060": "ft_w060_best.pth", "lr3e5": "ft_lr3e5_best.pth",
    "long_w15lr3": "ft_long_w15lr3_best.pth", "long_w20lr2": "ft_long_w20lr2_best.pth",
    "long_w10lr1": "ft_long_w10lr1_best.pth",
}
ALL_TAGS = {k: f"{LH}/{v}" for k, v in ALL_TAGS.items() if os.path.exists(f"{LH}/{v}")}
print("available members:", list(ALL_TAGS.keys()))

def acc(P, m):
    dm = S.rel_l2_per_sample(P[m], Y[m], C); tk = S.tke_rel_l2_per_sample(P[m], Y[m], C)
    mv = S.mvpe_rel_l2_per_sample(P[m], Y[m])
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))

RL, TK, MV = 94.168150, 74.025866, 92.836278
n_ = lambda x: x/(0.5+x)
W_of = lambda rl, tk, mv: 0.5*(1-n_((100/rl-1)*2)) + 0.3*(1-n_((100/tk-1)*2)) + 0.2*(1-n_((100/mv-1)*2))
ao = acc(P0, ev)
print("\n%-24s %7s %7s %7s %8s" % ("recipe", "rel_l2", "tke", "mvpe", "d_final~"))

def report(name, sd):
    P = run_sd(sd)
    a = acc(P, ev)
    d = [a[k]-ao[k] for k in range(3)]
    dfin = 0.306*d[0] + 0.163*d[1] + 0.218*d[2]
    print("%-24s %7.2f %7.2f %7.2f %+8.3f" % (name, a[0], a[1], a[2], dfin))
    return a, d, P

# original 6 (v1's soup) for reference
sixkeys = ["w005", "w010", "w015", "lr1e5", "w060", "lr3e5"]
sixkeys = [k for k in sixkeys if k in ALL_TAGS]
sds6 = [torch.load(ALL_TAGS[k], map_location=DEV) for k in sixkeys]
avg6 = {k: (sum(sd[k] for sd in sds6)/len(sds6)).to(sds6[0][k].dtype) if torch.is_tensor(sds6[0][k]) and (sds6[0][k].is_complex() or torch.is_floating_point(sds6[0][k])) else sds6[0][k] for k in sds6[0]}
report("soup_v1-style (%d members)" % len(sixkeys), avg6)

# all 9
allkeys = list(ALL_TAGS.keys())
sds9 = [torch.load(ALL_TAGS[k], map_location=DEV) for k in allkeys]
avg9 = {k: (sum(sd[k] for sd in sds9)/len(sds9)).to(sds9[0][k].dtype) if torch.is_tensor(sds9[0][k]) and (sds9[0][k].is_complex() or torch.is_floating_point(sds9[0][k])) else sds9[0][k] for k in sds9[0]}
a9, d9, P9 = report("MAXIMAL soup (%d members)" % len(allkeys), avg9)

# just the 3 long runs
longkeys = [k for k in ["long_w15lr3", "long_w20lr2", "long_w10lr1"] if k in ALL_TAGS]
sdsL = [torch.load(ALL_TAGS[k], map_location=DEV) for k in longkeys]
avgL = {k: (sum(sd[k] for sd in sdsL)/len(sdsL)).to(sdsL[0][k].dtype) if torch.is_tensor(sdsL[0][k]) and (sdsL[0][k].is_complex() or torch.is_floating_point(sdsL[0][k])) else sdsL[0][k] for k in sdsL[0]}
report("long-runs-only soup (%d members)" % len(longkeys), avgL)

print("\nfor reference: plateau is +4.20/+4.27/+4.24 tke (soup_v1/analyze2/soup_v3)")

best_name, best_sd, best_d, best_P = "MAXIMAL soup", avg9, d9, P9
ncplx = sum(1 for k in best_sd if torch.is_tensor(best_sd[k]) and best_sd[k].is_complex())
print("\nBEST CANDIDATE for gating: %s -- tke delta %+.2f, %d complex tensors preserved" % (best_name, best_d[1], ncplx))
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
print("TRUE bounds optimum: hu*=%.4f hv*=%.4f  held-out local E=%.4f" % (hu_star, hv_star, E_const(ERR, ev, hu_star, hv_star)))

RLp, TKp, MVp = RL+best_d[0], TK+best_d[1], MV+best_d[2]
json.dump({"name": best_name, "delta": best_d, "hu_star": hu_star, "hv_star": hv_star,
           "RL_proj": RLp, "TK_proj": TKp, "MV_proj": MVp, "W": W_of(RLp, TKp, MVp)},
          open(f"{LH}/soup_final_candidate_result.json", "w"), indent=1)
print("\nwrote soup_final_candidate.pth + result json")
