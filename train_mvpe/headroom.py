"""How much of the per-element headroom does our head actually capture?

    frac = (E_policy - E_bestconst) / (E_oracle - E_bestconst)

All three terms use the SAME raw local errors, so this is CALIBRATION-FREE -- it is
immune to the local->real scale that sec26 showed we cannot fit. If frac is high
locally but ~5% in reality, we have a TRANSFER problem. If it is low locally too, the
head itself is inadequate and must be replaced.

Board reference (computed from real subscores): competitors capture 27-43%.
Ours in reality: (0.5020-0.4876)/(0.7775-0.4876) = 5.0%.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
z = np.load(f"{B}/train_mvpe/runs/arrays3.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
LOHI = {3750, 5025, 25425, 26700}

def evaluate(mask_name, sel):
    dat = {}
    for ci, nm in ((0,"u"),(1,"v")):
        m = sel(z, nm)
        err = z[f"err_{nm}"][m][::SUBS].astype(np.float64)
        mu  = z[f"mut_{nm}"][m][::SUBS]
        dat[nm] = (err, np.digitize(mu, ED[ci]))
    N = sum(v[0].size for v in dat.values())
    # oracle: h = |err| exactly, per element
    E_or = sum(float(np.exp(-2*dat[nm][0]/SIG).sum()) for nm in dat)/N
    # best constant per channel
    E_bc = 0.0
    bc = {}
    for nm in dat:
        e = np.sort(dat[nm][0]); g = np.linspace(0.0005, 0.05, 900)
        k = np.searchsorted(e, g, side="right")
        j = int(np.argmax(np.exp(-2*g/SIG)*k)); bc[nm] = g[j]
        E_bc += float(np.exp(-2*g[j]/SIG)*k[j])
    E_bc /= N
    # shipped LUT
    E_lut = 0.0
    for ci, nm in ((0,"u"),(1,"v")):
        err, b = dat[nm]
        for kk in range(NB):
            s = err[b==kk]
            if s.size == 0: continue
            h = float(LUT[kk,ci]); E_lut += float(np.exp(-2*h/SIG)*(s<=h).sum())
    E_lut /= N
    # ORACLE per-bin h: the ceiling of a 24-bin lookup on THIS feature
    E_bin = 0.0
    for ci, nm in ((0,"u"),(1,"v")):
        err, b = dat[nm]
        for kk in range(NB):
            s = np.sort(err[b==kk])
            if s.size == 0: continue
            g = np.linspace(0.0005, 0.05, 600); c = np.searchsorted(s, g, side="right")
            E_bin += float(np.max(np.exp(-2*g/SIG)*c))
    E_bin /= N
    f = lambda E: 100*(E-E_bc)/(E_or-E_bc)
    print("%-22s  N=%d" % (mask_name, N))
    print("   best const [u %.4f v %.4f]  E %.4f" % (bc["u"], bc["v"], E_bc))
    print("   shipped LUT                 E %.4f   captures %5.1f%% of headroom" % (E_lut, f(E_lut)))
    print("   BEST POSSIBLE 24-bin LUT    E %.4f   captures %5.1f%%  <- ceiling of this feature" % (E_bin, f(E_bin)))
    print("   oracle (per-element)        E %.4f   captures 100.0%%" % E_or)
    print()

evaluate("RANDOM held-out (every5)", lambda z,nm: z[f"held_{nm}"])
evaluate("re_lohi (condition-shift)", lambda z,nm: np.isin(z[f"re_{nm}"], list(LOHI)))
evaluate("ALL data (in-sample)",      lambda z,nm: np.ones(z[f"err_{nm}"].shape, bool))
print("REFERENCE -- real leaderboard, same formula:")
print("   ours       (0.5020-0.4876)/(0.7775-0.4876) =  5.0%")
print("   benslash2  ~(0.5851-0.489)/(0.7775-0.489)  = 33.3%   (tke 74.70, WORSE than ours)")
print("   np-user    ~(0.6134-0.489)/(0.7775-0.489)  = 43.2%")
