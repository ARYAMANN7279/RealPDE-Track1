"""GATE 4B, the gate that decides this candidate.

  "Sanity-check any per-element policy against the best constant on the SAME
   held-out data. If it does not beat constants there, it will not beat them live."
  "fs_assure.py CANNOT score a per-element policy ... estimate instead from a
   MEASURED OUT-OF-SAMPLE INCREMENT OVER CONSTANTS applied to our known real
   constant performance."

So the estimate must be built as:
    predicted real E(policy) = REAL E(best constant) + measured increment
with the increment measured out-of-sample under the corrected calibration, and
REAL E(best constant) = 0.4876, which is a live-board fact (ROBUST, sec5B).

This is a different and more conservative construction than "dE vs the shipped
LUT", because it never relies on the model reproducing the shipped artifact's own E.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
z = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
AU, AV = 2.1500, 2.4250          # refitted soup calibration (sec22.3)

srt, bins = {}, {}
for ci, nm in ((0,"u"),(1,"v")):
    h = z[f"held_{nm}"]
    err = z[f"err_{nm}"][h][::SUBS].astype(np.float64)
    srt[nm] = np.sort(err)
    for mk in ("mut","muf"):
        b = np.digitize(z[f"{mk}_{nm}"][h][::SUBS], ED[ci])
        bins[(nm,mk)] = [np.sort(err[b==k]) for k in range(NB)]
nu, nv = srt["u"].size, srt["v"].size; N = nu+nv
a_of = {"u": AU, "v": AV}

def E_const(hu, hv):
    t = 0.0
    for nm, h in (("u",hu),("v",hv)):
        t += np.exp(-2*h/SIG)*np.searchsorted(srt[nm], h/a_of[nm], side="right")
    return t/N

def E_lut(lut, mk):
    t = 0.0
    for nm, ci in (("u",0),("v",1)):
        for k, s in enumerate(bins[(nm,mk)]):
            if s.size == 0: continue
            h = float(lut[k,ci])
            t += np.exp(-2*h/SIG)*np.searchsorted(s, h/a_of[nm], side="right")
    return t/N

# best constant under the SAME calibration and SAME held-out data
g = np.linspace(0.002, 0.040, 761)
bu = max(g, key=lambda h: np.exp(-2*h/SIG)*np.searchsorted(srt["u"], h/AU, side="right")/nu)
bv = max(g, key=lambda h: np.exp(-2*h/SIG)*np.searchsorted(srt["v"], h/AV, side="right")/nv)
E_best_const = E_const(bu, bv)
E_ship_const = E_const(0.0129, 0.0098)
print("held-out elements: u %d  v %d" % (nu, nv))
print("best constant under corrected calibration: [%.4f, %.4f]  E %.4f" % (bu, bv, E_best_const))
print("shipped constant [0.0129, 0.0098]        :          E %.4f" % E_ship_const)
print("REAL E at [0.0129,0.0098] (live board)   :          0.4876")
print("  local-vs-real bias at the constant anchor: %+.4f" % (E_ship_const - 0.4876))

W = 0.684593
print("\n%-34s %8s %10s %12s %10s" % ("policy", "E_local", "incr vs", "predicted", "-> final"))
print("%-34s %8s %10s %12s %10s" % ("", "", "best const", "real E", ""))
print("-"*78)
rows = [("shipped LUT (time-avg path)", LUT, "mut"),
        ("corrected LUT (time-avg path)", np.load(f"{B}/train_mvpe/runs/LUT_corrected_mut.npy"), "mut"),
        ("corrected LUT + per-frame head", np.load(f"{B}/train_mvpe/runs/LUT_corrected_muf.npy"), "muf")]
for name, lut, mk in rows:
    e = E_lut(lut, mk); incr = e - E_best_const
    pred_real = 0.4876 + incr
    dfinal = 100*W*(pred_real - 0.5020)*0.217
    print("%-34s %8.4f %+10.4f %12.4f %10.4f" % (name, e, incr, pred_real, 78.4566+dfinal))

print("\nGATE 4B check -- does the per-element policy beat the best constant")
print("on the same held-out data under the corrected calibration?")
for name, lut, mk in rows:
    e = E_lut(lut, mk)
    print("   %-34s %s (E %.4f vs %.4f)" % (name, "PASS" if e > E_best_const else "FAIL", e, E_best_const))
