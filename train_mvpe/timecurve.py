"""The lead-time axis is completely unexploited.

The head's features are TIME-AVERAGED, so its predicted error -- and therefore the
half-width -- is identical for output frame 0 and frame 19. But autoregressive error
grows strongly with lead time, so the SPS-optimal width must grow with it too.

Proposal: h(bin, ci, t) = LUT[bin, ci] * f[t], with f a 20-vector shipped in
head_assets.npz. Inference cost is one broadcast multiply -- effectively zero, unlike
running the head per frame (20x head cost, which the time subscore would not survive).

Measures: (a) baseline corrected LUT, (b) separable LUT x f(t), (c) full per-(bin,t)
table as an upper bound on what any time-dependence could buy.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
W = 0.684593; BANKED = 78.4566
z = np.load(f"{B}/train_mvpe/runs/arrays3.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); ED = A["ED"]; NB = A["LUT"].shape[0]
SHIP = np.load(f"{B}/train_mvpe/runs/LUT_ship.npy")
AU, AV = 1.579, 2.770
NT = 20
cell = {}
for ci, nm, a in ((0,"u",AU), (1,"v",AV)):
    h = z[f"held_{nm}"]
    err = (z[f"err_{nm}"][h][::SUBS].astype(np.float64))*a
    b = np.digitize(z[f"mut_{nm}"][h][::SUBS], ED[ci])
    t = z[f"t_{nm}"][h][::SUBS].astype(np.int64)
    for k in range(NB):
        for tt in range(NT):
            cell[(ci,k,tt)] = np.sort(err[(b==k)&(t==tt)])
NTOT = sum(v.size for v in cell.values())
def E_of(hfun):
    s = 0.0
    for (ci,k,tt), v in cell.items():
        if v.size == 0: continue
        h = hfun(ci,k,tt)
        s += np.exp(-2*h/SIG)*np.searchsorted(v, h, side="right")
    return s/NTOT
def best_h(v):
    if v.size == 0: return None
    kk = np.arange(1, v.size+1)/v.size
    return float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])

E0 = E_of(lambda ci,k,tt: float(SHIP[k,ci]))
print("(a) corrected LUT, no time dependence      E %.4f" % E0)

# error growth with lead time, for context
print("\nmedian scaled |error| by lead time (u / v):")
for tt in range(0, NT, 4):
    mu = np.concatenate([cell[(0,k,tt)] for k in range(NB)])
    mv = np.concatenate([cell[(1,k,tt)] for k in range(NB)])
    print("   t=%2d  %.5f  %.5f" % (tt, np.median(mu), np.median(mv)))

# (b) separable: coordinate ascent on f[t]
f = np.ones(NT)
for it in range(6):
    for tt in range(NT):
        grid = np.linspace(0.55, 2.2, 111)
        num = np.array([sum(np.exp(-2*(SHIP[k,ci]*g)/SIG)
                        * np.searchsorted(cell[(ci,k,tt)], SHIP[k,ci]*g, side="right")
                        for ci in (0,1) for k in range(NB)) for g in grid])
        f[tt] = grid[int(np.argmax(num))]
Eb = E_of(lambda ci,k,tt: float(SHIP[k,ci])*f[tt])
print("\n(b) separable LUT x f(t)                  E %.4f  (%+.4f)" % (Eb, Eb-E0))
print("    f(t) =", " ".join("%.2f" % x for x in f))

# (c) upper bound: independent optimum in every (bin, t) cell
def hc(ci,k,tt):
    v = cell[(ci,k,tt)]
    return best_h(v) if v.size >= 200 else float(SHIP[k,ci])
Ec = E_of(hc)
print("\n(c) full per-(bin,t) table (upper bound)  E %.4f  (%+.4f)" % (Ec, Ec-E0))

for nm, E in (("separable f(t)", Eb), ("full table", Ec)):
    d = 100*W*(E-E0)*0.217
    print("   %-16s -> dfinal %+.4f  -> %.4f" % (nm, d, BANKED+0.786+d))
np.save(f"{B}/train_mvpe/runs/timecurve.npy", f)
print("\nsaved f(t) -> runs/timecurve.npy")
