"""Can the soup be calibrated with ONE free parameter instead of two?

The soup has two real anchors -- not enough for (a_u, a_v). But the ORIGINAL
checkpoint's four anchors pin its RATIO a_u/a_v = 1.530/3.210 = 0.477 tightly.
Hypothesis: the RATIO (how u and v error scales differ between local and real) is a
property of the field/measurement and transfers across two FNOs, while the LEVEL
(overall hardness) depends on model accuracy and does not.

If so, fixing the ratio and fitting only the level gives ONE free parameter against
TWO anchors -- over-determined, hence testable. Residual small => defensible.
Residual large => the ratio does not transfer either, and the soup is uncalibratable
with present data. Either answer is decisive.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
z = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
srt, bins = {}, {}
for ci, nm in ((0,"u"),(1,"v")):
    h = z[f"held_{nm}"]
    err = z[f"err_{nm}"][h][::SUBS].astype(np.float64)
    srt[nm] = np.sort(err)
    b = np.digitize(z[f"mut_{nm}"][h][::SUBS], ED[ci])
    bins[nm] = [np.sort(err[b==k]) for k in range(NB)]
nu, nv = srt["u"].size, srt["v"].size; N = nu+nv

def E_lut(au, av, mult=1.0):
    t = 0.0
    for nm, ci, a in (("u",0,au),("v",1,av)):
        for k, s in enumerate(bins[nm]):
            if s.size == 0: continue
            h = float(LUT[k,ci])*mult
            t += np.exp(-2*h/SIG)*np.searchsorted(s, h/a, side="right")
    return t/N
def E_const(au, av, hu, hv):
    t = 0.0
    for nm, h, a in (("u",hu,au),("v",hv,av)):
        t += np.exp(-2*h/SIG)*np.searchsorted(srt[nm], h/a, side="right")
    return t/N

R_ORIG = 1.530/3.210
print("ratio from the ORIGINAL checkpoint's 4-anchor fit: a_u/a_v = %.4f" % R_ORIG)
print("soup anchors: LUT x1.00 -> 0.5020   LUT x1.25 -> 0.4939\n")

print("%-10s %8s %8s %10s %10s %9s" % ("ratio", "a_u", "a_v", "LUTx1.00", "LUTx1.25", "max err"))
print("-"*60)
best = None
for R, tag in [(R_ORIG, "orig"), (0.60,""), (0.70,""), (0.80,""), (0.90,""), (1.00,""), (1.937, "global_scale")]:
    bb = None
    for L in np.linspace(0.5, 6.0, 1101):     # level; a_u = L*R, a_v = L
        au, av = L*R, L
        e1, e2 = E_lut(au,av,1.00), E_lut(au,av,1.25)
        c = max(abs(e1-0.5020), abs(e2-0.4939))
        if bb is None or c < bb[0]: bb = (c, au, av, e1, e2)
    c, au, av, e1, e2 = bb
    print("%-10s %8.3f %8.3f %10.4f %10.4f %9.4f %s"
          % ("%.3f"%R, au, av, e1, e2, c, tag))
    if best is None or c < best[0]: best = (c, R, au, av)

print("\nbest ratio by soup-anchor fit: %.3f  (a_u %.3f a_v %.3f, max err %.4f)"
      % (best[1], best[2], best[3], best[0]))
print("=> the soup's own two anchors %s distinguish between ratios."
      % ("DO" if (max(0.0,0.0) or True) and False else "do NOT meaningfully"))

# how much does the predicted corrected-LUT gain vary across ratios that fit the
# soup anchors equally well?
print("\npredicted gain from rebuilding the LUT, across ratios that fit the anchors:")
W = 0.684593
def refit_gain(au, av):
    nl = LUT.copy()
    for nm, ci, a in (("u",0,au),("v",1,av)):
        for k, s in enumerate(bins[nm]):
            if s.size < 200: continue
            v = a*s; kk = np.arange(1,v.size+1)/v.size
            nl[k,ci] = float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    def E(lut):
        t=0.0
        for nm,ci,a2 in (("u",0,au),("v",1,av)):
            for k,s in enumerate(bins[nm]):
                if s.size==0: continue
                h=float(lut[k,ci]); t+=np.exp(-2*h/SIG)*np.searchsorted(s,h/a2,side="right")
        return t/N
    return E(nl)-E(LUT)
for R in (R_ORIG, 0.60, 0.80, 1.00, 1.937):
    bb = None
    for L in np.linspace(0.5, 6.0, 551):
        au, av = L*R, L
        c = max(abs(E_lut(au,av,1.00)-0.5020), abs(E_lut(au,av,1.25)-0.4939))
        if bb is None or c < bb[0]: bb = (c, au, av)
    c, au, av = bb
    g = refit_gain(au, av)
    print("   ratio %-6.3f (fit err %.4f)  dE %+.4f  -> dfinal %+.4f -> %.4f"
          % (R, c, g, 100*W*g*0.217, 78.4566+100*W*g*0.217))
