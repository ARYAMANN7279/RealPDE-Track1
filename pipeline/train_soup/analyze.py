"""Held-out analysis: does the SOUP beat the banked original, and which bound
policy should ship with it?

Everything here is measured on trajectories the head never trained on, and every
number is reported as a DELTA against a real leaderboard anchor -- absolute local
scores are on a different scale than the competition and must never be quoted as
predictions (project_memory section 2).

Self-check: global_scale() fits the local error distribution to the two real
CONSTANT anchors (E 0.4399 at [0.030,0.010]; E 0.4876 at [0.0129,0.0098]). After
scaling, the ORIGINAL model's held-out E at those bounds must land near them, or
the whole calibration is void and the output is meaningless.
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

ORIG_WORK = os.path.join(C.ROOT, "train_work")       # original-model cache (v7 pipeline)
SOUP_WORK = C.WORK                                    # soup cache

# ---- real leaderboard anchors (ground truth, project_memory section 1) -------
RL, TK, MV, TM = 94.168150, 74.025866, 92.836278, 91.36   # banked ROBUST subscores
SPS_BANKED = 33.08                                         # banked ROBUST sps
E_CONST_REAL = 0.4876                                      # real E, constants
E_HEAD_V7_REAL = 0.4932                                    # real E, v7 clean-LUT head
FIN = lambda rl, tk, mv, tm, sps: (0.306*rl + 0.163*tk + 0.218*mv
                                   + 0.100*tm + 0.217*sps)
n_ = lambda x: x/(0.5+x)
def W_of(rl, tk, mv):
    return (0.5*(1-n_((100/rl-1)*2)) + 0.3*(1-n_((100/tk-1)*2))
            + 0.2*(1-n_((100/mv-1)*2)))

def load(work):
    d = np.load(os.path.join(work, "cache.npz"))
    return (d["P"].astype(np.float32), d["Y"].astype(np.float32), d["wtraj"])

def split(wt):
    uniq = sorted(set(wt.tolist()))
    tr = np.isin(wt, uniq[0::2])
    return tr, ~tr

def acc_of(P, Y, m):
    dm = S.rel_l2_per_sample(P[m], Y[m], 2)
    tk = S.tke_rel_l2_per_sample(P[m], Y[m], 2)
    mv = S.mvpe_rel_l2_per_sample(P[m], Y[m])
    return (S.score_error(float(dm.mean())), S.score_error(float(tk.mean())),
            S.score_error(float(mv.mean())))

def E_const(ERR, SCM, m, hu, hv):
    eu = ERR[m][..., 0][SCM[m][..., 0]]; ev = ERR[m][..., 1][SCM[m][..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum()
                 + (np.exp(-2*hv/SIG)*(ev <= hv)).sum()) / (eu.size + ev.size)

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def head_mu(P, ckpath):
    ck = torch.load(os.path.join(SOUP_WORK, ckpath), map_location=dev, weights_only=False)
    NF = ck["nf"]; net = Net(NF).to(dev); net.load_state_dict(ck["sd"]); net.eval()
    FT = (feats(P) - ck["mu"]) / ck["std"]
    out = []
    with torch.no_grad():
        for i in range(0, len(P), 8):
            f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(dev)
            f = f.permute(0,1,4,2,3).reshape(-1, NF, 32, 64)
            out.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out, 0)

def E_head(MU, ERR, SCM, tr, te, NB=24):
    """Fit the LUT on tr only, evaluate on te -- the M55 protocol."""
    Eout = 0.0; tot = 0
    for ci in range(2):
        mt = SCM[..., ci] & tr[:, None, None, None]
        me = SCM[..., ci] & te[:, None, None, None]
        mu_t = MU[..., ci][mt]; er_t = ERR[..., ci][mt]
        q = np.quantile(mu_t, np.linspace(0, 1, NB+1)[1:-1])
        b_t = np.digitize(mu_t, q)
        LUT = np.zeros(NB, np.float32)
        for k in range(NB):
            s = b_t == k
            LUT[k] = best_h(er_t[s][::3]) if s.sum() > 200 else 0.012
        mu_e = MU[..., ci][me]; er_e = ERR[..., ci][me]
        h = LUT[np.digitize(mu_e, q)]
        Eout += float((np.exp(-2*h/SIG)*(er_e <= h)).sum()); tot += er_e.size
    return Eout/tot

print("="*68)
print("PART 1 -- accuracy: soup vs original, held-out trajectories")
print("="*68)
Po, Yo, wto = load(ORIG_WORK)
Ps, Ys, wts = load(SOUP_WORK)
assert np.array_equal(wto, wts), "caches disagree on trajectory layout"
assert np.array_equal(Yo, Ys), "caches disagree on targets"
tr, te = split(wts)
print("windows: %d total, %d fit / %d held-out, over %d trajectories"
      % (len(wts), tr.sum(), te.sum(), len(set(wts.tolist()))))
ao = acc_of(Po, Yo, te); asoup = acc_of(Ps, Ys, te)
print("\n  %-12s %8s %8s %8s" % ("model", "rel_l2", "tke", "mvpe"))
print("  %-12s %8.2f %8.2f %8.2f" % ("original", ao[0], ao[1], ao[2]))
print("  %-12s %8.2f %8.2f %8.2f" % ("soup", asoup[0], asoup[1], asoup[2]))
D = [asoup[k]-ao[k] for k in range(3)]
print("  %-12s %+8.2f %+8.2f %+8.2f   <-- deltas applied to real anchors"
      % ("delta", D[0], D[1], D[2]))
RL_s, TK_s, MV_s = RL+D[0], TK+D[1], MV+D[2]
print("\n  projected real subscores: rel_l2 %.2f  tke %.2f  mvpe %.2f"
      % (RL_s, TK_s, MV_s))
W_o = W_of(RL, TK, MV); W_s = W_of(RL_s, TK_s, MV_s)
print("  W: original %.4f -> soup %.4f  (tke pays twice: better tke raises W)"
      % (W_o, W_s))

print()
print("="*68)
print("PART 2 -- SELF-CHECK: reproduce the two real constant anchors")
print("="*68)
SCM = (Ys[..., :2] != 0.0)
ERRo = np.abs(Po[..., :2]-Yo[..., :2]).astype(np.float32)
SC_o = global_scale(ERRo, SCM)
ERRo_s = ERRo.copy()
for ci in range(2): ERRo_s[..., ci] *= SC_o[ci]
print("  original global scale: u x%.3f  v x%.3f" % (SC_o[0], SC_o[1]))
a1 = E_const(ERRo_s, SCM, te, 0.030, 0.010)
a2 = E_const(ERRo_s, SCM, te, 0.0129, 0.0098)
print("  E @ [0.030 ,0.010 ] = %.4f   (real anchor 0.4399, delta %+.4f)" % (a1, a1-0.4399))
print("  E @ [0.0129,0.0098] = %.4f   (real anchor 0.4876, delta %+.4f)" % (a2, a2-E_CONST_REAL))
OK = abs(a1-0.4399) < 0.05 and abs(a2-E_CONST_REAL) < 0.05
print("  SELF-CHECK %s" % ("PASSED" if OK else "*** FAILED -- output is void ***"))
if not OK:
    sys.exit("self-check failed; refusing to report calibrated numbers")

print()
print("="*68)
print("PART 3 -- bound policies on the SOUP, held-out")
print("="*68)
ERRs = np.abs(Ps[..., :2]-Ys[..., :2]).astype(np.float32)
SC_s = global_scale(ERRs, SCM)
for ci in range(2): ERRs[..., ci] *= SC_s[ci]
print("  soup global scale: u x%.3f  v x%.3f" % (SC_s[0], SC_s[1]))
rows = []
Ec = E_const(ERRs, SCM, te, 0.0129, 0.0098)
rows.append(("constants [.0129,.0098]", Ec))
for tag, ckp in [("head (calibrated target)", "head.pth"),
                 ("head (clean target)", "head_clean.pth")]:
    if not os.path.exists(os.path.join(SOUP_WORK, ckp)):
        print("  [skip] %s -- %s not built" % (tag, ckp)); continue
    MU = head_mu(Ps, ckp)
    rows.append((tag, E_head(MU, ERRs, SCM, tr, te)))
# same three policies on the ORIGINAL model, for a like-for-like reference
Ec_o = E_const(ERRo_s, SCM, te, 0.0129, 0.0098)

print("\n  %-26s %8s %8s %9s %9s" % ("policy (on soup)", "E", "sps", "final", "vs bank"))
best = None
for tag, E in rows:
    sps = 100*W_s*E
    f = FIN(RL_s, TK_s, MV_s, TM, sps)
    d = f - FIN(RL, TK, MV, TM, SPS_BANKED)
    print("  %-26s %8.4f %8.2f %9.2f %+9.2f" % (tag, E, sps, 78.07+d, d))
    if best is None or E > best[1]: best = (tag, E, sps, d)
print("\n  reference: original + same constants, held-out E %.4f (soup %+.4f)"
      % (Ec_o, Ec - Ec_o))
print("\n  BEST POLICY: %s  (E %.4f, sps %.2f, %+.2f vs banked 78.07)"
      % (best[0], best[1], best[2], best[3]))
print("\n  NOTE: 'final' assumes the time subscore is unchanged at %.2f. The soup is"
      % TM)
print("  the same architecture as the baseline, so inference cost is unchanged;")
print("  dropping the spectral correction should RECOVER most of v7's -1.47.")
json.dump({"acc_orig": ao, "acc_soup": asoup, "delta": D,
           "RL_s": RL_s, "TK_s": TK_s, "MV_s": MV_s, "W_s": W_s,
           "anchors": [a1, a2], "policies": {t: e for t, e in rows},
           "E_const_orig_heldout": Ec_o},
          open(os.path.join(SOUP_WORK, "analysis.json"), "w"), indent=1)
print("\nwrote analysis.json")
