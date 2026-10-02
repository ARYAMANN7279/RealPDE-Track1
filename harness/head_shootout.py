"""DECISIVE: error-target head vs disagreement-target head, UNDER REYNOLDS SHIFT.

The shipped (error-target) head's local gain was +0.0588 and its live gain +0.0056
-- ~10% retention. The whole case for the disagreement-distilled head is that it
should retain far more. That claim is currently supported only INDIRECTLY (the raw
ensemble retained 130%; disagreement is a more transferable ridge target). This
measures the actual trained heads, head-to-head, on a real distribution shift.

Both heads are evaluated identically: their output only RANKS elements into bins;
every half-width comes from clean errors via best_h() within each bin, fitted on
the FIT side of the split and scored on the EVAL side.
"""
import json, os, re, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"; WORK = f"{B}/train_work_maxsoup"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sys.path.insert(0, f"{B}/train_soup")
from head_common import feats, Net, global_scale, SIGMA_GLOBAL as SIG
from load_baseline import load_baseline
dev = "cuda:0"; C = 2; HOR = 40; IN_STEP = 20

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI_ = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(dev) for x in (MI, SI_, MT, ST)]
F = np.load(os.path.join(WORK, "frames.npy"), mmap_mode="r")
m = json.load(open(os.path.join(WORK, "meta.json")))
names = m["names"]
RE_t = np.array([int(re.match(r"(\d+)", n).group(1)) for n in names])
wins, wtraj = [], []
for i, (o, L) in enumerate(zip(m["off"], m["lens"])):
    for t0 in range(o, o + L - HOR + 1, HOR):
        wins.append(t0); wtraj.append(i)
W = np.stack([np.concatenate([np.asarray(F[s:s+HOR]), np.zeros((HOR,32,64,1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = W[:, :IN_STEP], W[:, IN_STEP:]
wtraj = np.array(wtraj); SCM = (Y[..., :C] != 0.0)
w_re = RE_t[wtraj]
uniq_re = sorted(set(RE_t.tolist())); mid = uniq_re[len(uniq_re)//2]

base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=dev)
mm = base_model; mm.load_state_dict(torch.load(f"{LH}/soup_final_candidate.pth", map_location=dev))
mm = mm.to(dev).eval()
o = []
with torch.no_grad():
    for i in range(0, len(Xin), 32):
        o.append((mm((torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(dev)-mi)/si)*st+mt).cpu().numpy()[..., :C])
P = np.concatenate(o, 0).astype(np.float32)
ERR = np.abs(P - Y[..., :C]).astype(np.float32)
SCALE = global_scale(ERR, SCM)
for ci in range(C): ERR[..., ci] *= SCALE[ci]

P3 = np.concatenate([P, np.zeros(P.shape[:-1]+(1,), np.float32)], -1)
FT_raw = feats(P3)

def head_mu(ckpt):
    ck = torch.load(os.path.join(WORK, ckpt), map_location=dev, weights_only=False)
    NF = ck["nf"]; net = Net(NF).to(dev); net.load_state_dict(ck["sd"]); net.eval()
    FT = (FT_raw - ck["mu"]) / ck["std"]
    out = []
    with torch.no_grad():
        for i in range(0, len(P), 8):
            f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(dev).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            out.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out, 0)

MUS = {}
for tag, ck in [("error-target (shipped style)", "head.pth"), ("disagreement-target (new)", "head_disag.pth")]:
    if os.path.exists(os.path.join(WORK, ck)):
        MUS[tag] = head_mu(ck); print("loaded", ck)

def best_h(v):
    v = np.sort(v)
    if v.size < 50: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def run_split(fitm, evm, NB=24):
    f4 = fitm[:, None, None, None]; e4 = evm[:, None, None, None]
    hu = best_h(ERR[..., 0][f4 & SCM[..., 0]]); hv = best_h(ERR[..., 1][f4 & SCM[..., 1]])
    eu = ERR[..., 0][e4 & SCM[..., 0]]; evv = ERR[..., 1][e4 & SCM[..., 1]]
    Ec = float((np.exp(-2*hu/SIG)*(eu<=hu)).sum() + (np.exp(-2*hv/SIG)*(evv<=hv)).sum())/(eu.size+evv.size)
    res = {}
    for tag, MU in MUS.items():
        num = 0.0; tot = 0
        for ci in range(C):
            mf = SCM[..., ci] & f4; me = SCM[..., ci] & e4
            mu_f = MU[..., ci][mf]; er_f = ERR[..., ci][mf]
            q = np.quantile(mu_f, np.linspace(0,1,NB+1)[1:-1]); b = np.digitize(mu_f, q)
            LUT = np.array([best_h(er_f[b==k][::3]) for k in range(NB)], np.float32)
            h = LUT[np.digitize(MU[..., ci][me], q)]; ee = ERR[..., ci][me]
            num += float((np.exp(-2*h/SIG)*(ee<=h)).sum()); tot += ee.size
        res[tag] = num/tot
    return Ec, res

uniq_t = sorted(set(wtraj.tolist()))
ind_f = np.isin(wtraj, uniq_t[0::2]); ind_e = ~ind_f
lo = w_re < mid; hi = w_re >= mid

print("\n" + "="*82)
print("%-32s %10s %10s %10s" % ("split", "E_const", "err-head", "disag-head"))
print("="*82)
rows = {}
for lab, fm, em in [("in-distribution", ind_f, ind_e),
                    ("Reynolds shift low->high", lo, hi),
                    ("Reynolds shift high->low", hi, lo)]:
    Ec, res = run_split(fm, em)
    e_err = res.get("error-target (shipped style)", float("nan"))
    e_dis = res.get("disagreement-target (new)", float("nan"))
    print("%-32s %10.4f %10.4f %10.4f" % (lab, Ec, e_err, e_dis))
    rows[lab] = (Ec, e_err, e_dis)

print("\n" + "="*82)
print("GAIN OVER CONSTANTS (this is what actually converts to score)")
print("="*82)
print("%-32s %12s %12s" % ("split", "err-head", "disag-head"))
for lab, (Ec, e_err, e_dis) in rows.items():
    print("%-32s %+12.4f %+12.4f" % (lab, e_err-Ec, e_dis-Ec))

gi = rows["in-distribution"]
gs = [(rows["Reynolds shift low->high"]), (rows["Reynolds shift high->low"])]
for name, idx in [("err-head", 1), ("disag-head", 2)]:
    g_in = gi[idx] - gi[0]
    g_sh = np.mean([r[idx] - r[0] for r in gs])
    print("\n  %-11s in-dist gain %+.4f | mean shifted gain %+.4f | RETENTION %.0f%%"
          % (name, g_in, g_sh, 100*g_sh/g_in if g_in else 0))
print("\n  (shipped error-head's IMPLIED LIVE retention was ~10%: +0.0588 -> +0.0056)")
json.dump({k: list(map(float, v)) for k, v in rows.items()},
          open(f"{LH}/head_shootout_result.json", "w"), indent=1)
