"""Is the LUT-rebuild gain robust to which distribution model and which head path?

Four combinations: {refitted scalar, power map} x {shipped time-avg MU, per-frame MU}.
If dE stays positive and similar across all four, the gain is a property of the
data, not of one fitted model. Held-out elements only.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259
z = np.load(f"{B}/train_mvpe/runs/arrays.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
SUBS = 4
MODELS = {"refit scalar": (2.1500, 1.00, 2.4250, 1.00),
          "power map":    (3.6667, 1.10, 3.1833, 1.05)}

def build(mukey):
    ch = {}
    for ci, nm in ((0, "u"), (1, "v")):
        h = z[f"held_{nm}"]
        err = z[f"err_{nm}"][h][::SUBS].astype(np.float64)
        mu  = z[f"{mukey}_{nm}"][h][::SUBS]
        b = np.digitize(mu, ED[ci])
        ch[nm] = [np.sort(err[b == k]) for k in range(NB)]
    return ch

F = lambda s, t: (np.searchsorted(s, t, side="right")/s.size if s.size else 0.0)

def E_with(ch, lut, p):
    au, bu, av, bv = p; tot = 0.0; N = 0
    for nm, ci, a, be in (("u", 0, au, bu), ("v", 1, av, bv)):
        for k, s in enumerate(ch[nm]):
            if s.size == 0: continue
            h = float(lut[k, ci]); N += s.size
            tot += s.size * np.exp(-2*h/SIG) * F(s, (h/a)**(1.0/be))
    return tot / N

def refit(ch, p):
    au, bu, av, bv = p; nl = LUT.copy()
    for nm, ci, a, be in (("u", 0, au, bu), ("v", 1, av, bv)):
        for k, s in enumerate(ch[nm]):
            if s.size < 200: continue
            v = a * s**be
            kk = np.arange(1, v.size+1)/v.size
            nl[k, ci] = float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    return nl

W = 0.684593
print("%-14s %-10s %9s %9s %8s %9s %9s" % ("dist model", "head path", "E_old", "E_new", "dE", "dfinal", "-> final"))
print("-"*72)
res = {}
for mk, mlabel in (("mut", "time-avg"), ("muf", "per-frame")):
    ch = build(mk)
    for name, p in MODELS.items():
        eo = E_with(ch, LUT, p); nl = refit(ch, p); en = E_with(ch, nl, p)
        dE = en - eo; df = 100*W*dE*0.217
        res[(name, mlabel)] = (eo, en, dE, nl)
        print("%-14s %-10s %9.4f %9.4f %+8.4f %+9.4f %9.4f" % (name, mlabel, eo, en, dE, df, 78.4566+df))

# combined effect: per-frame path AND refitted LUT, vs shipped path with shipped LUT
print("\n--- stacking the two fixes (relative to the SHIPPED artifact) ---")
for name, p in MODELS.items():
    eo_ship = res[(name, "time-avg")][0]
    en_pf   = res[(name, "per-frame")][1]
    dE = en_pf - eo_ship; df = 100*W*dE*0.217
    print("  %-14s shipped E %.4f -> per-frame + refit LUT %.4f   dE %+.4f  dfinal %+.4f -> %.4f"
          % (name, eo_ship, en_pf, dE, df, 78.4566+df))

np.savez(f"{B}/train_mvpe/runs/robust.npz",
         lut_scalar_pf=res[("refit scalar", "per-frame")][3],
         lut_scalar_ta=res[("refit scalar", "time-avg")][3],
         lut_power_pf=res[("power map", "per-frame")][3], ED=ED)
print("\nsaved candidate LUTs -> runs/robust.npz")
