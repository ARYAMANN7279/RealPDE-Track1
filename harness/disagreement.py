"""Is ENSEMBLE DISAGREEMENT a better per-element uncertainty signal than a learned head?

Why it might be: every estimator we have tried learns error from features on
train_real and must transfer to unseen Reynolds numbers. That transfer is what
fails (+0.055 local -> +0.0056 realised). Disagreement is MEASURED on the actual
test input, so there is no learned mapping to transfer.

Design keeps accuracy fixed: the SOUP still makes the prediction (so rel_l2/tke/
mvpe are unchanged); only the bound signal changes. sps is our biggest gap
(34.37 vs ~43.9 at the top = +2.07 final).

Bar to beat, from memory 7's failed feature regressor:
    corr(log s, log|err|) = 0.48,  sd(log s) = 0.68,  theory needs sd >~ 1.2
Also compares against the CURRENT shipped head on the same held-out windows.
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
from head_common import feats, Net, global_scale
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
ev = np.isin(wt, vidx); fit = ~ev
print("windows %d | held-out %d (%d traj)" % (len(wt), ev.sum(), len(vidx)))

base, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def run(sd):
    base.load_state_dict(sd); m = base.to(DEV).eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            xb = torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
            o.append((m((xb - mi) / si) * st + mt).cpu().numpy())
    return np.concatenate(o, 0).astype(np.float32)

MEM = ["ft_w005_best.pth", "ft_w010_best.pth", "ft_w015_best.pth",
       "ft_lr1e5_best.pth", "ft_w060_best.pth", "ft_lr3e5_best.pth",
       "ft_long_w15lr3_best.pth", "ft_long_w20lr2_best.pth", "ft_long_w10lr1_best.pth"]
MEM = [m for m in MEM if os.path.exists(f"{LH}/{m}")]
print("members:", len(MEM))
PS = []
for f in MEM:
    PS.append(run(torch.load(f"{LH}/{f}", map_location=DEV))[..., :C])
PS = np.stack(PS, 0)                       # (K, N, 20, 32, 64, 2)
P_soup = run(torch.load(f"{LH}/soup_final_candidate.pth", map_location=DEV))
ERR = np.abs(P_soup[..., :C] - Y[..., :C]).astype(np.float32)
SCALE = global_scale(ERR, SCM)
for ci in range(C): ERR[..., ci] *= SCALE[ci]
print("global scale: u x%.3f v x%.3f" % (SCALE[0], SCALE[1]))

def quality(sig, tag):
    """corr(log s, log err) and sd(log s) on held-out scored elements."""
    m = SCM[ev]
    a = np.log(sig[ev][m] + 1e-8); b = np.log(ERR[ev][m] + 1e-8)
    c = float(np.corrcoef(a, b)[0, 1]); s = float(a.std())
    print("  %-26s corr %.3f   sd(log s) %.3f" % (tag, c, s))
    return c, s

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size + 1) / v.size
    return float(v[np.argmax(np.exp(-2 * v / SIG) * k)])

def E_from_signal(sig, NB=24):
    """Fit signal->half-width LUT on the FIT split, evaluate E on held-out."""
    num = 0.0; tot = 0
    for ci in range(C):
        mf = SCM[..., ci] & fit[:, None, None, None]
        me = SCM[..., ci] & ev[:, None, None, None]
        sf = sig[..., ci][mf]; ef = ERR[..., ci][mf]
        q = np.quantile(sf, np.linspace(0, 1, NB + 1)[1:-1]); bnf = np.digitize(sf, q)
        LUT = np.array([best_h(ef[bnf == k][::3]) if (bnf == k).sum() > 200 else 0.012
                        for k in range(NB)], np.float32)
        h = LUT[np.digitize(sig[..., ci][me], q)]; ee = ERR[..., ci][me]
        num += float((np.exp(-2 * h / SIG) * (ee <= h)).sum()); tot += ee.size
    return num / tot

def E_const(hu, hv):
    eu = ERR[ev][..., 0][SCM[ev][..., 0]]; evv = ERR[ev][..., 1][SCM[ev][..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum() + (np.exp(-2*hv/SIG)*(evv <= hv)).sum()) / (eu.size + evv.size)

print("\n=== signal quality (bar from memory 7: corr 0.48 / sd 0.68 FAILED; need sd >~1.2) ===")
sigs = {}
for K in (3, 5, len(PS)):
    if K > len(PS): continue
    idx = np.linspace(0, len(PS) - 1, K).astype(int)
    s = PS[idx].std(axis=0) + 1e-8
    sigs["disagree K=%d" % K] = s
    quality(s, "disagreement K=%d" % K)

# current shipped head, same split, for a like-for-like comparison
z = np.load(f"{B}/train_work_maxsoup/head_assets.npz")
NF = int(z["nf"]); net = Net(NF).to(DEV)
net.load_state_dict({k[2:]: torch.from_numpy(z["w_" + k[2:]]) for k in z.files if k.startswith("w_")})
net.eval()
FT = (feats(P_soup) - z["fmu"]) / z["fsd"]; MU = []
with torch.no_grad():
    for i in range(0, len(P_soup), 8):
        f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV).permute(0,1,4,2,3).reshape(-1, NF, 32, 64)
        MU.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
MU = np.concatenate(MU, 0)
sigs["learned head (shipped)"] = np.exp(MU)
quality(np.exp(MU), "learned head (shipped)")

print("\n=== E on held-out (fit LUT on train split) ===")
ec = E_const(0.0129, 0.0098)
print("  %-26s E %.4f" % ("constants [.0129,.0098]", ec))
res = {}
for tag, s in sigs.items():
    e = E_from_signal(s)
    res[tag] = e
    print("  %-26s E %.4f   (%+.4f vs constants)" % (tag, e, e - ec))

print("\n=== what it is worth, using the REAL anchor ===")
print("  SOUP_v1 real: sps 34.3656, W 0.684538 -> real E 0.5020")
head_e = res.get("learned head (shipped)")
for tag, e in res.items():
    if tag.startswith("disagree"):
        gain = e - head_e
        print("  %-26s local E %+.4f vs head -> if it transfers: sps %+.2f, final %+.3f"
              % (tag, gain, gain * 100 * 0.684538, gain * 100 * 0.684538 * 0.217))
print("\n  (time cost of K models must be subtracted: K=3 is roughly -0.12 final)")
