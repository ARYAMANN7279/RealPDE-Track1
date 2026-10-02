"""A free validation: does the refitted model REDISCOVER the empirically best constants?

sec5B searched constant bounds on the live board and found [0.0129, 0.0098] best
(E 0.4876), better than [0.030, 0.010] (0.4398) and [0.107537, 0.010307] (0.2697).
That optimum is REAL and was found without any distribution model.

So: ask each calibration what constant pair it thinks is optimal. A calibration
that is right about the real error distribution should land near [0.0129, 0.0098].
One that is wrong should point somewhere else -- and that mistake is exactly how a
bad calibration corrupts the LUT.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259
z = np.load(f"{B}/train_mvpe/runs/orig_arrays.npz")
d = {}
for nm in ("u", "v"):
    d[nm] = np.sort(z[f"err_{nm}"].astype(np.float64))
nu, nv = d["u"].size, d["v"].size; N = nu + nv

def Echan(nm, h, a):
    k = np.searchsorted(d[nm], h/a, side="right")
    return np.exp(-2*h/SIG) * k / d[nm].size

def opt_h(nm, a):
    """separable: maximise exp(-2h/sig)*F(h/a) over h"""
    grid = np.linspace(0.0005, 0.06, 2400)
    vals = [Echan(nm, h, a) for h in grid]
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])

CALS = [("global_scale (shipped)", 2.3199, 1.1982),
        ("refitted on 4 anchors ", 1.530,  3.210)]
print("empirical best constants found on the LIVE BOARD (sec5B): [0.0129, 0.0098]  E 0.4876\n")
print("%-24s %10s %10s %12s" % ("calibration", "opt h_u", "opt h_v", "predicted E"))
print("-"*60)
for nm, au, av in CALS:
    hu, eu = opt_h("u", au); hv, ev = opt_h("v", av)
    E = (nu*eu + nv*ev)/N
    print("%-24s %10.4f %10.4f %12.4f" % (nm, hu, hv, E))

print("\nfor reference, each calibration's E at the KNOWN-BEST pair [0.0129, 0.0098]:")
for nm, au, av in CALS:
    E = (nu*Echan("u",0.0129,au) + nv*Echan("v",0.0098,av))/N
    print("   %-24s %.4f   (reality: 0.4876)" % (nm, E))

print("\ndistance of each calibration's predicted optimum from the real one:")
for nm, au, av in CALS:
    hu, _ = opt_h("u", au); hv, _ = opt_h("v", av)
    print("   %-24s h_u %+.4f   h_v %+.4f" % (nm, hu-0.0129, hv-0.0098))
