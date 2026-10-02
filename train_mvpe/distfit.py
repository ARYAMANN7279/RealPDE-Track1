"""Fit the REAL error distribution instead of rescaling the local one.

global_scale() applies ONE multiplicative constant per channel, chosen so the
local median matches a Weibull median fitted to two real anchors. Matching a
median does not match a CDF: measured on the shipped artifact, the local harness
returns E 0.5328 / 0.4806 at the two constant-bound anchors whose REAL values are
0.4876 / 0.4399 -- over-stated by ~0.045 with no head involved at all.

A scalar cannot fix that (it moves level, not shape). This fits a monotone
per-channel power map

    err_real = a * err_local**b

which changes level AND spread while preserving the within-channel ranking the
LUT encodes. Four unknowns (a,b per channel), constrained by four REAL anchors:

    E([0.030 ,0.010 ]) = 0.4399     constants, wide      (sec4)
    E([0.0129,0.0098]) = 0.4876     constants, banked    (sec5B)
    E(LUT x 1.00)      = 0.5020     shipped SOUP_v1      (sec19)
    E(LUT x 1.25)      = 0.4939     WIDE125 submission   (sec19)

The last two use the SHIPPED time-averaged head path, because that is what
produced those two numbers on the live board.
"""
import json, sys
import numpy as np
from scipy.optimize import least_squares

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
SIG = 0.0563870259
SUBS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

z = np.load(f"{B}/train_mvpe/runs/arrays.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]

d = {}
for nm in ("u", "v"):
    d[nm] = dict(err=z[f"err_{nm}"][::SUBS].astype(np.float64),
                 mut=z[f"mut_{nm}"][::SUBS].astype(np.float64))
n_u, n_v = d["u"]["err"].size, d["v"]["err"].size
print("fit points: u %d  v %d  (subsample 1/%d)" % (n_u, n_v, SUBS))

# shipped per-element half-widths from the time-averaged head path
for ci, nm in ((0, "u"), (1, "v")):
    d[nm]["h_lut"] = LUT[np.digitize(d[nm]["mut"], ED[ci]), ci].astype(np.float64)

def E_pool(hu, hv, eu, ev):
    su = float((np.exp(-2*hu/SIG) * (eu <= hu)).sum())
    sv = float((np.exp(-2*hv/SIG) * (ev <= hv)).sum())
    return (su + sv) / (eu.size + ev.size)

def transformed(p):
    au, bu, av, bv = p
    return au * d["u"]["err"]**bu, av * d["v"]["err"]**bv

ANCHORS = [("const wide",  0.030,  0.010,  0.4399),
           ("const banked", 0.0129, 0.0098, 0.4876)]

def residuals(p):
    eu, ev = transformed(p)
    r = [E_pool(hu, hv, eu, ev) - tgt for _, hu, hv, tgt in ANCHORS]
    r.append(E_pool(d["u"]["h_lut"],        d["v"]["h_lut"],        eu, ev) - 0.5020)
    r.append(E_pool(d["u"]["h_lut"]*1.25,   d["v"]["h_lut"]*1.25,   eu, ev) - 0.4939)
    return r

# ---- current behaviour: scalar scale only (b == 1), for reference ----
print("\n--- what a SCALAR scale can do (b fixed at 1, current global_scale form) ---")
def res_scalar(p):
    return residuals([p[0], 1.0, p[1], 1.0])
s0 = least_squares(res_scalar, [2.32, 1.20], bounds=([0.2, 0.2], [12, 12]))
print("  best scalar a_u %.4f a_v %.4f" % (s0.x[0], s0.x[1]))
for (nm, hu, hv, tgt), r in zip(ANCHORS + [("LUT x1.00", 0, 0, 0.5020), ("LUT x1.25", 0, 0, 0.4939)],
                                res_scalar(s0.x)):
    print("    %-13s target %.4f  fitted %.4f  resid %+.4f" % (nm, tgt, tgt + r, r))
print("  sum|resid| = %.4f" % np.abs(res_scalar(s0.x)).sum())

# ---- power map ----
print("\n--- power map  err_real = a * err_local**b ---")
best = None
for b0 in (0.8, 1.0, 1.3):
    for a0 in (1.5, 2.5, 4.0):
        try:
            s = least_squares(residuals, [a0, b0, a0*0.5, b0],
                              bounds=([0.05, 0.3, 0.05, 0.3], [40, 3.0, 40, 3.0]))
        except Exception:
            continue
        if best is None or s.cost < best.cost: best = s
p = best.x
print("  a_u %.4f  b_u %.4f   a_v %.4f  b_v %.4f" % tuple(p))
for (nm, tgt), r in zip([("const wide", 0.4399), ("const banked", 0.4876),
                         ("LUT x1.00", 0.5020), ("LUT x1.25", 0.4939)], residuals(p)):
    print("    %-13s target %.4f  fitted %.4f  resid %+.4f" % (nm, tgt, tgt + r, r))
print("  sum|resid| = %.4f" % np.abs(residuals(p)).sum())

# ---- rebuild the LUT against the FITTED REAL distribution ----
def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

eu, ev = transformed(p)
print("\n--- rebuild the LUT against the fitted real distribution ---")
newLUT = np.zeros_like(LUT)
for ci, nm, e in ((0, "u", eu), (1, "v", ev)):
    mu = d[nm]["mut"]
    b = np.digitize(mu, ED[ci])
    for k in range(LUT.shape[0]):
        s = b == k
        newLUT[k, ci] = best_h(e[s]) if s.sum() > 200 else LUT[k, ci]
h_old_u, h_old_v = d["u"]["h_lut"], d["v"]["h_lut"]
h_new_u = newLUT[np.digitize(d["u"]["mut"], ED[0]), 0]
h_new_v = newLUT[np.digitize(d["v"]["mut"], ED[1]), 1]
E_old = E_pool(h_old_u, h_old_v, eu, ev)
E_new = E_pool(h_new_u, h_new_v, eu, ev)
print("  E with SHIPPED LUT (fitted to local dist) : %.4f   <- matches real 0.5020 by construction" % E_old)
print("  E with LUT refitted to the real dist      : %.4f" % E_new)
dE = E_new - E_old
W = 0.684593
print("  dE %+.4f  ->  dsps %+.4f  ->  dfinal %+.4f  -> %.4f" % (dE, 100*W*dE, 100*W*dE*0.217, 78.4566 + 100*W*dE*0.217))
print("\n  LUT u  old %s" % np.round(LUT[:, 0], 4).tolist())
print("  LUT u  new %s" % np.round(newLUT[:, 0], 4).tolist())
print("  LUT v  old %s" % np.round(LUT[:, 1], 4).tolist())
print("  LUT v  new %s" % np.round(newLUT[:, 1], 4).tolist())
np.savez(f"{B}/train_mvpe/runs/distfit.npz", p=p, newLUT=newLUT, ED=ED, E_old=E_old, E_new=E_new)
