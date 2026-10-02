"""THE DECISIVE TRANSFER TEST.

Every bounds method this project has shipped looked good locally and underdelivered
live (head: +0.0588 local -> +0.0056 live). The standing hypothesis (section 11A) is
that the real test set has genuinely unseen Reynolds numbers, while our local
"held-out" trajectories share flow conditions with training ones.

That hypothesis is directly testable. Local data spans 18 Reynolds numbers
(3750..26700). Fit bounds on the LOW half, score on the HIGH half -> a real
distribution shift, much closer to what the live evaluator does.

Measured for each bound policy:
    gain_indist = E_policy - E_const   on an in-distribution split
    gain_shift  = E_policy - E_const   under Reynolds shift
    retention   = gain_shift / gain_indist
Retention is the number that actually predicts live behaviour. The shipped CNN
head's implied live retention was ~0.10 (+0.0056 / +0.0588). Anything with high
retention is worth shipping even if its in-distribution gain is smaller.
"""
import json, os, re, sys
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
meta = json.load(open(f"{LH}/tr_meta.json")); off = meta["off"]; lens = meta["lens"]; names = meta["names"]
ntraj = len(lens)
RE = np.array([int(re.match(r"(\d+)", n).group(1)) for n in names])
uniq_re = sorted(set(RE.tolist()))
mid = uniq_re[len(uniq_re)//2]
print("Reynolds range %d..%d, split at %d" % (uniq_re[0], uniq_re[-1], mid))

wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
Wall = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = Wall[:, :IN], Wall[:, IN:]; SCM = (Y[..., :C] != 0.0)
w_re = RE[wt]

base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def run_sd(sd):
    m = base_model; m.load_state_dict(sd); m = m.to(DEV).eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy()[..., :C])
    return np.concatenate(o, 0).astype(np.float32)

MEM = ["ft_w005_best.pth","ft_w010_best.pth","ft_w015_best.pth","ft_lr1e5_best.pth",
       "ft_w060_best.pth","ft_lr3e5_best.pth","ft_long_w15lr3_best.pth",
       "ft_long_w20lr2_best.pth","ft_long_w10lr1_best.pth"]
MEM = [f"{LH}/{m}" for m in MEM if os.path.exists(f"{LH}/{m}")]
print("running %d members..." % len(MEM))
A = np.stack([run_sd(torch.load(p, map_location=DEV)) for p in MEM], 0)
Pm = A.mean(0); SG = A.std(0)
ERR = np.abs(Pm - Y[..., :C])
print("ensemble built.")

def best_h(v):
    v = np.sort(v)
    if v.size < 50: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def evaluate(fitm, evm, NB=24):
    """returns (E_const, E_disag) fitting on fitm, scoring on evm."""
    f4 = fitm[:, None, None, None]; e4 = evm[:, None, None, None]
    # constants
    hu = best_h(ERR[..., 0][f4 & SCM[..., 0]]); hv = best_h(ERR[..., 1][f4 & SCM[..., 1]])
    eu = ERR[..., 0][e4 & SCM[..., 0]]; evv = ERR[..., 1][e4 & SCM[..., 1]]
    Ec = float((np.exp(-2*hu/SIG)*(eu <= hu)).sum() + (np.exp(-2*hv/SIG)*(evv <= hv)).sum())/(eu.size+evv.size)
    # disagreement LUT
    num = 0.0; tot = 0
    for ci in range(C):
        mf = SCM[..., ci] & f4; me = SCM[..., ci] & e4
        sf = SG[..., ci][mf]; ef = ERR[..., ci][mf]
        q = np.quantile(sf, np.linspace(0, 1, NB+1)[1:-1]); b = np.digitize(sf, q)
        LUT = np.array([best_h(ef[b == k][::3]) for k in range(NB)], np.float32)
        h = LUT[np.digitize(SG[..., ci][me], q)]; ee = ERR[..., ci][me]
        num += float((np.exp(-2*h/SIG)*(ee <= h)).sum()); tot += ee.size
    return Ec, num/tot

print("\n" + "="*78)
print("SPLIT 1 -- IN-DISTRIBUTION (interleaved trajectories, what we've been measuring)")
print("="*78)
uniq_t = sorted(set(wt.tolist()))
fit1 = np.isin(wt, uniq_t[0::2]); ev1 = ~fit1
Ec1, Ed1 = evaluate(fit1, ev1)
g1 = Ed1 - Ec1
print("  E_const=%.4f  E_disag=%.4f  gain=%+.4f" % (Ec1, Ed1, g1))

print("\n" + "="*78)
print("SPLIT 2 -- REYNOLDS SHIFT (fit on LOW Re, score on HIGH Re) <-- the real test")
print("="*78)
fit2 = w_re < mid; ev2 = w_re >= mid
print("  fit windows %d (Re<%d) | eval windows %d (Re>=%d)" % (fit2.sum(), mid, ev2.sum(), mid))
Ec2, Ed2 = evaluate(fit2, ev2)
g2 = Ed2 - Ec2
print("  E_const=%.4f  E_disag=%.4f  gain=%+.4f" % (Ec2, Ed2, g2))

print("\n" + "="*78)
print("SPLIT 3 -- REVERSE SHIFT (fit on HIGH Re, score on LOW Re)")
print("="*78)
Ec3, Ed3 = evaluate(ev2, fit2)
g3 = Ed3 - Ec3
print("  E_const=%.4f  E_disag=%.4f  gain=%+.4f" % (Ec3, Ed3, g3))

print("\n" + "="*78)
print("RETENTION -- the number that predicts live behaviour")
print("="*78)
print("  in-distribution gain      %+.4f" % g1)
print("  Reynolds-shift gain (fwd) %+.4f   retention %.0f%%" % (g2, 100*g2/g1 if g1 else 0))
print("  Reynolds-shift gain (rev) %+.4f   retention %.0f%%" % (g3, 100*g3/g1 if g1 else 0))
print("  mean shifted retention    %.0f%%" % (100*((g2+g3)/2)/g1 if g1 else 0))
print("\n  for comparison, the shipped CNN head's IMPLIED live retention was ~10%%")
print("  (+0.0588 local -> +0.0056 live, section 6A)")
print("\n  NOTE: constants themselves also degrade under shift; E_const in-dist %.4f vs shifted %.4f/%.4f"
      % (Ec1, Ec2, Ec3))
json.dump({"g_indist": g1, "g_fwd": g2, "g_rev": g3,
           "Ec": [Ec1, Ec2, Ec3], "Ed": [Ed1, Ed2, Ed3]},
          open(f"{LH}/reynolds_transfer_result.json", "w"), indent=1)
print("\nwrote reynolds_transfer_result.json")
