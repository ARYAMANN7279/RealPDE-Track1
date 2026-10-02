"""distfit, redone without a gradient optimiser.

E contains indicator functions, so its numerical Jacobian is zero almost
everywhere and least_squares stops at its initial point -- distfit.py returned
several parameters exactly equal to their inits, so its +0.0223 was void.

Exact reformulation. For a monotone per-channel map err_real = a*err**b, the
indicator 1[a*err**b <= h] is exactly 1[err <= (h/a)**(1/b)], so every anchor is
a handful of empirical-CDF lookups:

    E_const = SUM_ci n_ci * exp(-2h_ci/sig) * F_ci((h_ci/a_ci)**(1/b_ci)) / N
    E_LUT   = SUM_ci,k n_ci,k * exp(-2h_ci,k/sig) * F_ci,k((h_ci,k/a_ci)**(1/b_ci)) / N

with F the empirical CDF of the UNTRANSFORMED local error (searchsorted on a
sorted array). That is microseconds per evaluation, so the parameters can be
found by grid search, which does not care that the objective is a step function.

Held-out elements only: the shipped LUT was fitted on train trajectories and
applied to unseen data, so the held-out set is the right local analogue.
"""
import sys
import numpy as np

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
SIG = 0.0563870259
SUBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
USE_HELD = (sys.argv[2] if len(sys.argv) > 2 else "held") == "held"

z = np.load(f"{B}/train_mvpe/runs/arrays.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]
NB = LUT.shape[0]

ch = {}
for ci, nm in ((0, "u"), (1, "v")):
    err = z[f"err_{nm}"]; mut = z[f"mut_{nm}"]; held = z[f"held_{nm}"]
    if USE_HELD:
        err, mut = err[held], mut[held]
    err = err[::SUBS].astype(np.float64); mut = mut[::SUBS]
    b = np.digitize(mut, ED[ci])
    bins = []
    for k in range(NB):
        s = err[b == k]
        bins.append((np.sort(s), float(LUT[k, ci])))
    ch[nm] = dict(sorted_all=np.sort(err), n=err.size, bins=bins, ci=ci)
N = ch["u"]["n"] + ch["v"]["n"]
print("elements: u %d  v %d  (%s, subsample 1/%d)"
      % (ch["u"]["n"], ch["v"]["n"], "held-out" if USE_HELD else "all", SUBS))

def F(sorted_arr, t):
    if sorted_arr.size == 0: return 0.0
    return float(np.searchsorted(sorted_arr, t, side="right")) / sorted_arr.size

def inv(h, a, bexp):
    return (h / a) ** (1.0 / bexp)

def E_const(p, hu, hv):
    au, bu, av, bv = p
    tot = 0.0
    for nm, h, a, be in (("u", hu, au, bu), ("v", hv, av, bv)):
        c = ch[nm]
        tot += c["n"] * np.exp(-2*h/SIG) * F(c["sorted_all"], inv(h, a, be))
    return tot / N

def E_lut(p, mult=1.0):
    au, bu, av, bv = p
    tot = 0.0
    for nm, a, be in (("u", au, bu), ("v", av, bv)):
        for s, h in ch[nm]["bins"]:
            if s.size == 0: continue
            hm = h * mult
            tot += s.size * np.exp(-2*hm/SIG) * F(s, inv(hm, a, be))
    return tot / N

TARGETS = [("const wide",   0.4399), ("const banked", 0.4876),
           ("LUT x1.00",    0.5020), ("LUT x1.25",    0.4939)]

def resid(p):
    return np.array([E_const(p, 0.030, 0.010) - 0.4399,
                     E_const(p, 0.0129, 0.0098) - 0.4876,
                     E_lut(p, 1.00) - 0.5020,
                     E_lut(p, 1.25) - 0.4939])

def show(tag, p):
    r = resid(p)
    print("\n%s   a_u %.4f b_u %.4f  a_v %.4f b_v %.4f" % (tag, *p))
    for (nm, tgt), rr in zip(TARGETS, r):
        print("    %-13s target %.4f  model %.4f  resid %+.4f" % (nm, tgt, tgt + rr, rr))
    print("    max|resid| %.4f   sum|resid| %.4f" % (np.abs(r).max(), np.abs(r).sum()))
    return np.abs(r).max()

# what global_scale actually chose
show("CURRENT global_scale (scalar) ", [2.3199, 1.0, 1.1982, 1.0])

def grid(bu_list, bv_list, au_rng, av_rng, na=61):
    best = None
    for bu in bu_list:
        for bv in bv_list:
            for au in np.linspace(*au_rng, na):
                for av in np.linspace(*av_rng, na):
                    p = (au, bu, av, bv)
                    c = np.abs(resid(p)).max()
                    if best is None or c < best[0]: best = (c, p)
    return best

# scalar-only (b == 1): is global_scale merely mis-fitted, or is the SHAPE wrong?
c1, p1 = grid([1.0], [1.0], (1.0, 5.0), (0.5, 4.0), 81)
show("BEST SCALAR (b=1, refitted)   ", p1)

# allow shape
c2, p2 = grid(np.linspace(0.75, 1.35, 13), np.linspace(0.75, 1.35, 13), (1.0, 5.0), (0.5, 4.0), 31)
au, bu, av, bv = p2
c3, p3 = grid([bu], [bv], (max(0.2, au-0.3), au+0.3), (max(0.2, av-0.3), av+0.3), 61)
show("BEST POWER MAP (a,b per chan) ", p3)

BESTP = p3 if c3 <= c1 else p1
print("\nchosen: a_u %.4f b_u %.4f  a_v %.4f b_v %.4f  (max|resid| %.4f)"
      % (*BESTP, min(c1, c3)))

# ---- rebuild the LUT against the fitted real distribution ----
au, bu, av, bv = BESTP
def best_h_from(sorted_err, a, be):
    """optimal half-width on the TRANSFORMED errors, exactly (they stay sorted)."""
    if sorted_err.size < 200: return None
    v = a * sorted_err**be
    k = np.arange(1, v.size+1) / v.size
    return float(v[int(np.argmax(np.exp(-2*v/SIG) * k))])

newLUT = LUT.copy()
for nm, ci, a, be in (("u", 0, au, bu), ("v", 1, av, bv)):
    for k, (s, h) in enumerate(ch[nm]["bins"]):
        nh = best_h_from(s, a, be)
        if nh is not None: newLUT[k, ci] = nh

def E_with(lut, p):
    au, bu, av, bv = p; tot = 0.0
    for nm, ci, a, be in (("u", 0, au, bu), ("v", 1, av, bv)):
        for k, (s, _h) in enumerate(ch[nm]["bins"]):
            if s.size == 0: continue
            h = float(lut[k, ci])
            tot += s.size * np.exp(-2*h/SIG) * F(s, inv(h, a, be))
    return tot / N

E_old = E_with(LUT, BESTP); E_new = E_with(newLUT, BESTP)
W = 0.684593
dE = E_new - E_old
print("\n--- LUT rebuilt against the fitted real distribution ---")
print("  E, shipped LUT   %.4f   (real anchor 0.5020, model resid %+.4f)" % (E_old, E_old - 0.5020))
print("  E, refitted LUT  %.4f" % E_new)
print("  dE %+.4f -> dsps %+.4f -> dfinal %+.4f -> %.4f"
      % (dE, 100*W*dE, 100*W*dE*0.217, 78.4566 + 100*W*dE*0.217))
print("\n  h_u  old median %.4f  new median %.4f  (range %.4f-%.4f -> %.4f-%.4f)"
      % (np.median(LUT[:,0]), np.median(newLUT[:,0]), LUT[:,0].min(), LUT[:,0].max(),
         newLUT[:,0].min(), newLUT[:,0].max()))
print("  h_v  old median %.4f  new median %.4f  (range %.4f-%.4f -> %.4f-%.4f)"
      % (np.median(LUT[:,1]), np.median(newLUT[:,1]), LUT[:,1].min(), LUT[:,1].max(),
         newLUT[:,1].min(), newLUT[:,1].max()))
np.savez(f"{B}/train_mvpe/runs/distfit2.npz", p=np.array(BESTP), newLUT=newLUT, ED=ED,
         E_old=E_old, E_new=E_new)
