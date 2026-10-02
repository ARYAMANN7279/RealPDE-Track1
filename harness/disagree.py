"""ENSEMBLE DISAGREEMENT as the per-element uncertainty signal.

Why this is different from everything already ruled out (section 7):
  * "per-element bounds from a feature regressor" failed with corr 0.48,
    sd(log s_hat) 0.68 -- theory needs >= 1.2.
  * the shipped CNN head LEARNS a mapping features -> error on train_real; it
    scored +0.0588 E locally but only +0.0056 live (section 6A). A learned
    ranking does not transfer to unseen Reynolds numbers.
  * ensemble disagreement is MEASURED on the actual test inputs at inference
    time, not predicted from a learned mapping, so the transfer failure mode
    structurally does not apply. Only the s -> h calibration has to transfer,
    which is a 1-D monotone mapping rather than a spatial ranking.

This script measures, on the SAME 17-trajectory honest holdout used all day:
  1. corr(log s, log|err|)  vs the 0.48 bar
  2. sd(log s)              vs the >=1.2 bar
  3. actual held-out E from an s -> h LUT, vs constants and vs oracle
for several ensemble sizes (time budget scales with K).
"""
import json, os, sys, itertools
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
print("windows %d | held-out %d (%d traj) | fit %d" % (len(wt), ev.sum(), len(vidx), fit.sum()))

base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def run_sd(sd):
    m = base_model; m.load_state_dict(sd); m = m.to(DEV).eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy()[..., :C])
    return np.concatenate(o, 0).astype(np.float32)

MEMBERS = {
    "w005": "ft_w005_best.pth", "w010": "ft_w010_best.pth", "w015": "ft_w015_best.pth",
    "lr1e5": "ft_lr1e5_best.pth", "w060": "ft_w060_best.pth", "lr3e5": "ft_lr3e5_best.pth",
    "long_w15lr3": "ft_long_w15lr3_best.pth", "long_w20lr2": "ft_long_w20lr2_best.pth",
    "long_w10lr1": "ft_long_w10lr1_best.pth",
}
MEMBERS = {k: f"{LH}/{v}" for k, v in MEMBERS.items() if os.path.exists(f"{LH}/{v}")}
print("members:", list(MEMBERS.keys()))
P = {}
for k, p in MEMBERS.items():
    P[k] = run_sd(torch.load(p, map_location=DEV))
    print("  ran", k, flush=True)

def acc_of(Pm, m):
    P3 = np.concatenate([Pm, np.zeros(Pm.shape[:-1]+(1,), np.float32)], -1)
    dm = S.rel_l2_per_sample(P3[m], Y[m], C); tk = S.tke_rel_l2_per_sample(P3[m], Y[m], C)
    mv = S.mvpe_rel_l2_per_sample(P3[m], Y[m])
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())), S.score_error(float(mv.mean())))

def best_h(v):
    v = np.sort(v)
    if v.size < 50: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def E_const(ERR, m, hu, hv):
    eu = ERR[m][..., 0][SCM[m][..., 0]]; evv = ERR[m][..., 1][SCM[m][..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum() + (np.exp(-2*hv/SIG)*(evv <= hv)).sum())/(eu.size+evv.size)

def E_lut(SGN, ERR, NB=24):
    """bin by disagreement signal (fit half), look up best_h, score held-out."""
    num = 0.0; tot = 0
    fit4 = fit[:, None, None, None]; ev4 = ev[:, None, None, None]
    for ci in range(C):
        mf = SCM[..., ci] & fit4; me = SCM[..., ci] & ev4
        sf = SGN[..., ci][mf]; ef = ERR[..., ci][mf]
        q = np.quantile(sf, np.linspace(0, 1, NB+1)[1:-1])
        b = np.digitize(sf, q)
        LUT = np.array([best_h(ef[b == k][::3]) for k in range(NB)], np.float32)
        h = LUT[np.digitize(SGN[..., ci][me], q)]; ee = ERR[..., ci][me]
        num += float((np.exp(-2*h/SIG)*(ee <= h)).sum()); tot += ee.size
    return num/tot

def oracle_E(ERR):
    ev4 = ev[:, None, None, None]; num = 0.0; tot = 0
    for ci in range(C):
        me = SCM[..., ci] & ev4; ee = ERR[..., ci][me]
        num += float(np.exp(-2*ee/SIG).sum()); tot += ee.size
    return num/tot

def diagnostics(SGN, ERR, tag):
    ev4 = ev[:, None, None, None]
    m = (SCM[..., 0] & ev4)
    s = SGN[..., 0][m]; e = ERR[..., 0][m]
    ok = (s > 1e-12) & (e > 1e-12)
    ls, le = np.log(s[ok][::7]), np.log(e[ok][::7])
    corr = float(np.corrcoef(ls, le)[0, 1]); sd = float(ls.std())
    print("  %-26s corr(log s, log|err|) = %.3f   sd(log s) = %.3f" % (tag, corr, sd))
    return corr, sd

print("\n" + "="*78)
print("BAR TO BEAT (memory section 7, feature regressor that FAILED): corr 0.48, sd(log s) 0.68")
print("theory needs sd(log s) >= 1.2 for per-element to beat constants meaningfully")
print("="*78)

combos = [
    ("all 9", list(MEMBERS.keys())),
    ("6 v1-members", ["w005","w010","w015","lr1e5","w060","lr3e5"]),
    ("4 diverse", ["w005","w060","long_w20lr2","long_w10lr1"]),
    ("3 cheap", ["w005","w060","long_w20lr2"]),
]
rows = []
for tag, keys in combos:
    keys = [k for k in keys if k in P]
    if len(keys) < 2: continue
    A = np.stack([P[k] for k in keys], 0)
    Pm = A.mean(0); Sg = A.std(0)
    ERR = np.abs(Pm - Y[..., :C])
    print("\n--- %s (K=%d) ---" % (tag, len(keys)))
    corr, sd = diagnostics(Sg, ERR, "ensemble disagreement")
    a = acc_of(Pm, ev)
    hu = best_h(ERR[..., 0][fit[:, None, None, None] & SCM[..., 0]])
    hv = best_h(ERR[..., 1][fit[:, None, None, None] & SCM[..., 1]])
    Ec = E_const(ERR, ev, hu, hv)
    El = E_lut(Sg, ERR)
    Eo = oracle_E(ERR)
    print("  acc(held-out): rel_l2 %.2f tke %.2f mvpe %.2f" % a)
    print("  E constants=%.4f   E disagreement-LUT=%.4f   (gain %+.4f)   oracle=%.4f" % (Ec, El, El-Ec, Eo))
    rows.append((tag, len(keys), corr, sd, Ec, El, El-Ec, a))

print("\n" + "="*78)
print("SUMMARY")
print("="*78)
print("%-14s %3s %7s %8s %9s %9s %8s" % ("ensemble","K","corr","sd(logs)","E_const","E_disag","gain"))
for tag, K, corr, sd, Ec, El, g, a in rows:
    print("%-14s %3d %7.3f %8.3f %9.4f %9.4f %+8.4f" % (tag, K, corr, sd, Ec, El, g))
json.dump([{"tag":t,"K":K,"corr":c,"sd":s,"E_const":Ec,"E_disag":El,"gain":g,"acc":a}
           for t,K,c,s,Ec,El,g,a in rows], open(f"{LH}/disagree_result.json","w"), indent=1)
print("\nwrote disagree_result.json")
