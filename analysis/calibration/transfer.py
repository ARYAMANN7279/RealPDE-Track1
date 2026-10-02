"""Is the local->real error scale a property of the EVALUATION or of the MODEL?

The original checkpoint has four well-spread real anchors and fits cleanly
(a_u 1.530 / a_v 3.210, max|resid| 0.0117, rediscovers the known optimum to 0.0007).
The soup has only TWO real anchors, both LUT-type and nearly the same configuration.

If `a` is really "the private set's errors are larger than our held-out errors", it is
a property of the evaluation and should transfer between models. Test: apply the
ORIGINAL's fitted a to the SOUP's local errors and predict the SOUP's two real anchors,
which were never used to fit it. That is a genuine out-of-sample test.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
zs = np.load(f"{B}/train_mvpe/runs/arrays2.npz")          # SOUP
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
srt, bins = {}, {}
for ci, nm in ((0,"u"),(1,"v")):
    h = zs[f"held_{nm}"]
    err = zs[f"err_{nm}"][h][::SUBS].astype(np.float64)
    srt[nm] = np.sort(err)
    b = np.digitize(zs[f"mut_{nm}"][h][::SUBS], ED[ci])
    bins[nm] = [np.sort(err[b==k]) for k in range(NB)]
nu, nv = srt["u"].size, srt["v"].size; N = nu+nv

def E_const(p, hu, hv):
    au, av = p; t = 0.0
    for nm, h, a in (("u",hu,au),("v",hv,av)):
        t += np.exp(-2*h/SIG)*np.searchsorted(srt[nm], h/a, side="right")
    return t/N
def E_lut(p, mult=1.0):
    au, av = p; t = 0.0
    for nm, ci, a in (("u",0,au),("v",1,av)):
        for k, s in enumerate(bins[nm]):
            if s.size == 0: continue
            h = float(LUT[k,ci])*mult
            t += np.exp(-2*h/SIG)*np.searchsorted(s, h/a, side="right")
    return t/N

CANDS = [("ORIGINAL ckpt fit (4 anchors)", 1.530, 3.210),
         ("soup fit to MIXED anchors    ", 2.150, 2.425),
         ("global_scale (shipped)       ", 2.3199, 1.1982)]
print("SOUP's own real anchors (never used to fit the ORIGINAL calibration):")
print("   SOUP_v1  E 0.5020    WIDE125 (LUT x1.25)  E 0.4939\n")
print("%-32s %10s %10s %9s" % ("calibration applied to SOUP", "LUTx1.00", "LUTx1.25", "max err"))
print("-"*66)
for nm, au, av in CANDS:
    e1 = E_lut((au,av), 1.00); e2 = E_lut((au,av), 1.25)
    print("%-32s %10.4f %10.4f %9.4f" % (nm, e1, e2, max(abs(e1-0.5020), abs(e2-0.4939))))
print("\n(the ORIGINAL-fit row is a pure out-of-sample prediction: different model,")
print(" different anchors, nothing about the soup used in fitting it)")

# what does the ORIGINAL calibration say the soup's shipped LUT increment over
# constants should be?  Reality: 0.5020 - 0.4876 = +0.0144, but note 0.4876 is the
# ORIGINAL's constant anchor, so this comparison is only valid if the scale transfers.
for nm, au, av in CANDS:
    g = np.linspace(0.002, 0.040, 761)
    bu = max(g, key=lambda h: np.exp(-2*h/SIG)*np.searchsorted(srt["u"], h/au, side="right")/nu)
    bv = max(g, key=lambda h: np.exp(-2*h/SIG)*np.searchsorted(srt["v"], h/av, side="right")/nv)
    print("\n%s" % nm)
    print("   best constant on soup: [%.4f, %.4f]  E %.4f" % (bu, bv, E_const((au,av), bu, bv)))
    print("   shipped LUT E %.4f -> increment vs best const %+.4f"
          % (E_lut((au,av)), E_lut((au,av)) - E_const((au,av), bu, bv)))
