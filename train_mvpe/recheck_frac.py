"""CORRECTION CHECK: competitors' captured fraction must use THEIR OWN references.

sec27 computed every competitor's fraction against OUR best-constant (0.4876) and OUR
oracle (0.7775). Both depend on the model's own error distribution. A team with better
rel_l2 has smaller errors, so BOTH references rise, and their true fraction may be far
lower than sec27 claims -- which would mean the gap is ACCURACY, not bounds technique.

Method: take our LOCAL error shape, scale it so the best-constant E matches our KNOWN
real 0.4876. That pins our real error scale. Then scale by each competitor's rel_l2
error ratio and recompute their own references.
"""
import numpy as np
from scipy.optimize import brentq
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 32
se=lambda e:100.0/(1.0+0.5*e); inv=lambda t:brentq(lambda e:se(e)-t,1e-9,50.0); n_=lambda x:x/(0.5+x)
def W(dm,tk,mv): return 0.5*(1-n_(inv(dm)))+0.3*(1-n_(inv(tk)))+0.2*(1-n_(inv(mv)))
z = np.load(f"{B}/train_mvpe/runs/arrays3.npz")
e = {nm: np.sort(z[f"err_{nm}"][z[f"held_{nm}"]][::SUBS].astype(np.float64)) for nm in ("u","v")}
N = sum(v.size for v in e.values())
def refs(a):
    """best-constant E and oracle E when local errors are scaled by a"""
    Eor = sum(float(np.exp(-2*a*v/SIG).sum()) for v in e.values())/N
    Ebc = 0.0; g = np.linspace(0.0005,0.06,1200)
    for v in e.values():
        k = np.searchsorted(v, g/a, side="right"); Ebc += float(np.max(np.exp(-2*g/SIG)*k))
    return Ebc/N, Eor
a = brentq(lambda a: refs(a)[0]-0.4876, 0.5, 6.0)
bc0, or0 = refs(a)
print("scale a=%.3f reproduces our real best-const E=0.4876"%a)
print("  => our real oracle E = %.4f   (sec27 assumed 0.7775)\n"%or0)
err_us = 2*(100-94.05)/94.05
T=[("np-user",94.67,79.52,94.12,43.91),("doomduke2",94.90,78.17,94.13,43.31),
   ("iapetos1918",94.82,79.06,93.91,43.74),("hituc",94.70,79.60,93.39,42.91),
   ("benslash2",95.01,74.70,94.03,40.96),("redouanelg",94.69,78.91,94.02,40.37),
   ("bobajeshjohnson",94.70,76.81,93.62,40.89),("simon-zhou",94.63,77.63,93.51,41.99),
   ("agent33 (#67)",94.26,77.83,93.67,38.22),("US SOUP_v1",94.05,76.00,92.87,34.3656)]
print("%-18s %7s %8s %8s %8s %9s"%("team","rel_l2","E","own bc","own or","captured"))
print("-"*62)
for nm,dm,tk,mv,sps in T:
    E = sps/(100*W(dm,tk,mv))
    r = (2*(100-dm)/dm)/err_us
    bc, orc = refs(a*r)
    fr = 100*(E-bc)/(orc-bc)
    print("%-18s %7.2f %8.4f %8.4f %8.4f %8.1f%%"%(nm,dm,E,bc,orc,fr))
print("\nsec27 (WRONG, used OUR refs for everyone): benslash2 33.3%, np-user 43.2%")
