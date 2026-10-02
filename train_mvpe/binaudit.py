"""Audit: did every LUT bin actually get refitted, or did some keep the OLD width?

build_cand.py skips bins with <200 held-out samples. Any skipped bin retains the
SHIPPED (mis-scaled) half-width, producing a LUT that is a MIX of corrected and
uncorrected widths -- exactly the kind of silent inconsistency sec6 was caused by.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
z = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
SHIP = np.load(f"{B}/train_mvpe/runs/LUT_ship.npy")
print("bins: %d   held-out subsample stride: %d\n" % (NB, SUBS))
for ci, nm in ((0,"u"),(1,"v")):
    h = z[f"held_{nm}"]
    err = z[f"err_{nm}"][h][::SUBS]
    b = np.digitize(z[f"mut_{nm}"][h][::SUBS], ED[ci])
    cnt = np.array([(b==k).sum() for k in range(NB)])
    stale = [k for k in range(NB) if cnt[k] < 200]
    unchanged = [k for k in range(NB) if abs(float(SHIP[k,ci])-float(LUT[k,ci])) < 1e-9]
    print("channel %s: min bin count %d, max %d, median %d" % (nm, cnt.min(), cnt.max(), int(np.median(cnt))))
    print("   bins with <200 samples (skipped by refit): %s" % (stale if stale else "NONE"))
    print("   bins whose width is byte-identical to shipped: %s" % (unchanged if unchanged else "NONE"))
    if stale:
        print("   ⛔ those bins carry the OLD mis-scaled width")
    print("   width change per bin (shipped -> new):")
    print("     " + "  ".join("%d:%.4f→%.4f" % (k, LUT[k,ci], SHIP[k,ci]) for k in range(0, NB, 4)))
    # how much of the scored mass sits in each bin?
    frac = cnt/cnt.sum()
    print("   scored-mass fraction in the 3 least-populated bins: %.4f"
          % frac[np.argsort(cnt)[:3]].sum())
    print()
