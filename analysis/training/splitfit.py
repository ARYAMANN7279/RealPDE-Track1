"""Does a regime-extrapolated local subset need LESS scale correction?

If the private eval is a regime-extrapolation set (rules: "unseen parameter
regimes"), then local errors measured on re_lohi should already resemble the real
ones, and the fitted a_u/a_v should sit closer to 1.0 with smaller anchor residuals
than the every5 subset needs. If instead every split needs the same large scale,
the correction is not about regime shift at all.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
z = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
RE_LO, RE_HI = {3750, 5025}, {25425, 26700}

def masks(nm):
    re = z[f"re_{nm}"]; aoa = z[f"aoa_{nm}"]; held = z[f"held_{nm}"]
    return {"every5": held,
            "re_lohi": np.isin(re, list(RE_LO | RE_HI)),
            "aoa_edge": np.isin(aoa, [0, 20]),
            "all": np.ones(re.size, bool)}

def build(split):
    ch = {}
    for ci, nm in ((0, "u"), (1, "v")):
        m = masks(nm)[split]
        err = z[f"err_{nm}"][m][::SUBS].astype(np.float64)
        mu = z[f"mut_{nm}"][m][::SUBS]
        b = np.digitize(mu, ED[ci])
        ch[nm] = dict(all=np.sort(err), bins=[np.sort(err[b == k]) for k in range(NB)])
    return ch

F = lambda s, t: (np.searchsorted(s, t, side="right")/s.size if s.size else 0.0)
def mk(ch):
    N = ch["u"]["all"].size + ch["v"]["all"].size
    def E_const(p, hu, hv):
        au, av = p; t = 0.0
        for nm, h, a in (("u", hu, au), ("v", hv, av)):
            c = ch[nm]; t += c["all"].size*np.exp(-2*h/SIG)*F(c["all"], h/a)
        return t/N
    def E_lut(p, mult=1.0):
        au, av = p; t = 0.0
        for nm, ci, a in (("u", 0, au), ("v", 1, av)):
            for k, s in enumerate(ch[nm]["bins"]):
                if s.size == 0: continue
                h = float(LUT[k, ci])*mult
                t += s.size*np.exp(-2*h/SIG)*F(s, h/a)
        return t/N
    return E_const, E_lut, N

AU = np.linspace(0.8, 4.0, 129); AV = np.linspace(0.8, 4.5, 149)
print("%-9s %10s %8s %8s %10s   %s" % ("split", "elements", "a_u", "a_v", "max|resid|", "anchor residuals"))
print("-"*96)
for split in ("every5", "re_lohi", "aoa_edge", "all"):
    ch = build(split); E_const, E_lut, N = mk(ch)
    ANCH = [lambda p: E_const(p, 0.030, 0.010), lambda p: E_const(p, 0.0129, 0.0098),
            lambda p: E_lut(p, 1.00), lambda p: E_lut(p, 1.25)]
    TG = [0.4399, 0.4876, 0.5020, 0.4939]
    best = None
    for au in AU:
        for av in AV:
            r = [ANCH[i]((au, av)) - TG[i] for i in range(4)]
            c = max(abs(x) for x in r)
            if best is None or c < best[0]: best = (c, (au, av), r)
    c, p, r = best
    print("%-9s %10d %8.3f %8.3f %10.4f   %s" % (split, N, p[0], p[1], c,
          " ".join("%+.4f" % x for x in r)))
