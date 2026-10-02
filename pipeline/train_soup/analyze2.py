"""HONEST held-out analysis -- supersedes analyze.py, which was contaminated.

analyze.py split trajectories even/odd. But finetune.py:32 holds out every 5th
trajectory (`vidx=set(range(0,ntraj,5))`), so the soup members TRAINED on 65 of
81 trajectories. Most of analyze.py's "held-out" set is soup training data, and
its tke delta (+5.00) is inflated relative to soup.py's honest +4.20.

Sets used here (trajectory index i, 0..80):
    soup trained on   i %% 5 != 0        (65 traj)
    head trained on   i even             (analyze/04_train_head split)
    CLEAN             i %% 5 == 0 and i odd  -> {5,15,...,75}, 8 traj
CLEAN is unseen by BOTH the soup and the head, so it is the only set on which a
shipped-artifact number means anything. It is small, so results are noisy and are
reported as such.

The head increment measured on the contaminated split was +0.0552 -- essentially
the M55 figure (+0.0537) whose real realization was +0.0056 (section 6A). Treat any
large local increment as suspect until it is measured somewhere genuinely unseen.
"""
import json, os, sys
import numpy as np, torch
from importlib.machinery import SourceFileLoader
HERE = os.path.dirname(os.path.abspath(__file__))
C = SourceFileLoader("c", os.path.join(HERE, "00_config.py")).load_module()
C.seed_all(); sys.path.insert(0, HERE)
sys.path.insert(0, C.KIT); sys.path.insert(0, os.path.join(C.KIT, "_vendor"))
from head_common import feats, Net, global_scale, SIGMA_GLOBAL as SIG
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(C.KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ORIG_WORK = os.path.join(C.ROOT, "train_work"); SOUP_WORK = C.WORK

RL, TK, MV, TM = 94.168150, 74.025866, 92.836278, 91.36
SPS_BANKED, E_CONST_REAL, E_HEAD_V7_REAL = 33.08, 0.4876, 0.4932
FIN = lambda rl, tk, mv, tm, sps: 0.306*rl + 0.163*tk + 0.218*mv + 0.100*tm + 0.217*sps
n_ = lambda x: x/(0.5+x)
W_of = lambda rl, tk, mv: (0.5*(1-n_((100/rl-1)*2)) + 0.3*(1-n_((100/tk-1)*2))
                           + 0.2*(1-n_((100/mv-1)*2)))
BANKED_F = FIN(RL, TK, MV, TM, SPS_BANKED)

do = np.load(os.path.join(ORIG_WORK, "cache.npz")); ds = np.load(os.path.join(SOUP_WORK, "cache.npz"))
Po = do["P"].astype(np.float32); Ps = ds["P"].astype(np.float32)
Y = ds["Y"].astype(np.float32); wt = ds["wtraj"]
assert np.array_equal(do["wtraj"], wt) and np.array_equal(do["Y"].astype(np.float32), Y)
uniq = sorted(set(wt.tolist()))
soup_tr = np.isin(wt, [i for i in uniq if i % 5 != 0])
head_tr = np.isin(wt, uniq[0::2])
CLEAN = np.isin(wt, [i for i in uniq if i % 5 == 0 and i % 2 == 1])
LUTFIT = head_tr & ~CLEAN
print("windows %d | soup-train %d | head-train %d | CLEAN(unseen by both) %d over %d traj"
      % (len(wt), soup_tr.sum(), head_tr.sum(), CLEAN.sum(),
         len({i for i in uniq if i % 5 == 0 and i % 2 == 1})))

def acc(P, m):
    dm = S.rel_l2_per_sample(P[m], Y[m], 2); tk = S.tke_rel_l2_per_sample(P[m], Y[m], 2)
    mv = S.mvpe_rel_l2_per_sample(P[m], Y[m])
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())),
            S.score_error(float(mv.mean())))

print("\n" + "="*70)
print("PART 1 -- soup accuracy gain, CONTAMINATED vs CLEAN")
print("="*70)
print("  %-34s %7s %7s %7s" % ("set", "rel_l2", "tke", "mvpe"))
for tag, m in [("soup-TRAIN (in-sample, inflated)", soup_tr), ("CLEAN (unseen by soup)", CLEAN)]:
    ao, as_ = acc(Po, m), acc(Ps, m)
    d = [as_[k]-ao[k] for k in range(3)]
    print("  %-34s %+7.2f %+7.2f %+7.2f" % (tag+" delta", d[0], d[1], d[2]))
    if tag.startswith("CLEAN"): D = d
print("\n  soup.py's honest figure (its own held-out): -0.30 / +4.20 / -0.13")
RL_s, TK_s, MV_s = RL+D[0], TK+D[1], MV+D[2]
W_s = W_of(RL_s, TK_s, MV_s)
print("  CLEAN deltas -> projected real: rel_l2 %.2f  tke %.2f  mvpe %.2f" % (RL_s, TK_s, MV_s))
print("  W: original %.4f -> soup %.4f" % (W_of(RL, TK, MV), W_s))

SCM = (Y[..., :2] != 0.0)
def scaled(P):
    E = np.abs(P[..., :2]-Y[..., :2]).astype(np.float32)
    sc = global_scale(E, SCM)
    for ci in range(2): E[..., ci] *= sc[ci]
    return E, sc
ERRo, sco = scaled(Po); ERRs, scs = scaled(Ps)

def E_const(ERR, m, hu, hv):
    eu = ERR[m][..., 0][SCM[m][..., 0]]; ev = ERR[m][..., 1][SCM[m][..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum()
                 + (np.exp(-2*hv/SIG)*(ev <= hv)).sum())/(eu.size+ev.size)

print("\n" + "="*70)
print("PART 2 -- SELF-CHECK on the ORIGINAL model (never trained on any of this)")
print("="*70)
a1 = E_const(ERRo, CLEAN, 0.030, 0.010); a2 = E_const(ERRo, CLEAN, 0.0129, 0.0098)
print("  E @ [0.030 ,0.010 ] = %.4f  (real 0.4399, %+0.4f)" % (a1, a1-0.4399))
print("  E @ [0.0129,0.0098] = %.4f  (real 0.4876, %+0.4f)" % (a2, a2-E_CONST_REAL))
BIAS = a2 - E_CONST_REAL
print("  --> harness bias at the TIGHT anchor: %+.4f  (memory section 5B expects ~+0.030)" % BIAS)
print("  every local E below is corrected by this bias before being turned into a score")

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def head_mu(P, ckpath):
    ck = torch.load(os.path.join(SOUP_WORK, ckpath), map_location=dev, weights_only=False)
    NF = ck["nf"]; net = Net(NF).to(dev); net.load_state_dict(ck["sd"]); net.eval()
    FT = (feats(P)-ck["mu"])/ck["std"]; out = []
    with torch.no_grad():
        for i in range(0, len(P), 8):
            f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(dev)
            out.append(net(f.permute(0,1,4,2,3).reshape(-1, NF, 32, 64))
                       .reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out, 0)

def E_head(MU, ERR, fit, ev, NB=24):
    num = 0.0; tot = 0
    for ci in range(2):
        mf = SCM[..., ci] & fit[:, None, None, None]
        me = SCM[..., ci] & ev[:, None, None, None]
        mu_f = MU[..., ci][mf]; er_f = ERR[..., ci][mf]
        q = np.quantile(mu_f, np.linspace(0, 1, NB+1)[1:-1]); b = np.digitize(mu_f, q)
        LUT = np.array([best_h(er_f[b == k][::3]) if (b == k).sum() > 200 else 0.012
                        for k in range(NB)], np.float32)
        h = LUT[np.digitize(MU[..., ci][me], q)]; er_e = ERR[..., ci][me]
        num += float((np.exp(-2*h/SIG)*(er_e <= h)).sum()); tot += er_e.size
    return num/tot

print("\n" + "="*70)
print("PART 3 -- bound policies on the SOUP, evaluated on CLEAN")
print("="*70)
print("  LUT fitted on %d windows (head-train, CLEAN excluded), evaluated on %d CLEAN windows"
      % (LUTFIT.sum(), CLEAN.sum()))
rows = [("constants [.0129,.0098]", E_const(ERRs, CLEAN, 0.0129, 0.0098))]
for tag, ckp in [("head (calibrated target)", "head.pth"), ("head (clean target)", "head_clean.pth")]:
    if os.path.exists(os.path.join(SOUP_WORK, ckp)):
        rows.append((tag, E_head(head_mu(Ps, ckp), ERRs, LUTFIT, CLEAN)))
Ec_o = E_const(ERRo, CLEAN, 0.0129, 0.0098)

print("\n  %-26s %8s %9s %8s %8s %9s" % ("policy", "E_local", "E_real*", "sps", "final", "vs bank"))
best = None
for tag, E in rows:
    Er = E - BIAS
    sps = 100*W_s*Er
    f = FIN(RL_s, TK_s, MV_s, TM, sps)
    print("  %-26s %8.4f %9.4f %8.2f %8.2f %+9.2f"
          % (tag, E, Er, sps, 78.07+(f-BANKED_F), f-BANKED_F))
    if best is None or Er > best[1]: best = (tag, Er, sps, f-BANKED_F)
print("  * E_real = E_local - harness bias (%+.4f)" % BIAS)
inc = rows[-1][1] - rows[0][1]
print("\n  head increment over constants on CLEAN: %+.4f" % inc)
print("  for comparison: M55 local increment +0.0537 -> REALIZED +0.0056 (10x over)")
print("  contaminated-split increment from analyze.py: +0.0552")
print("\n  BEST: %s  E_real %.4f  sps %.2f  %+.2f vs banked" % best)
print("\n  reference: original+constants on CLEAN E_local %.4f (soup %+.4f)"
      % (Ec_o, rows[0][1]-Ec_o))
json.dump({"delta_clean": D, "RL_s": RL_s, "TK_s": TK_s, "MV_s": MV_s, "W_s": W_s,
           "bias": BIAS, "policies": {t: e for t, e in rows}, "n_clean": int(CLEAN.sum())},
          open(os.path.join(SOUP_WORK, "analysis_clean.json"), "w"), indent=1)
print("\nwrote analysis_clean.json")
