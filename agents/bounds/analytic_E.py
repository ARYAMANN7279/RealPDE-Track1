"""Step 2 -- analytic E ceiling for the RealPDE SPS bounds term.

Per scored element the reward is  r = exp(-(upper-lower)/sig) * 1[lower <= t <= upper],
sig = 0.0563870259.  E = mean r.  For an element whose signed error about the interval
midpoint is eps ~ s*Z (Z a standardised density, unit s.d.), a symmetric interval of
half-width h has expected reward R(h) = exp(-2h/sig) * P(|eps| <= h).

Parts
  A  optimal symmetric half-width h*(k), k = s/sig, and the reward R*(k)      (Gauss/Laplace/t3)
  B  "give-up" analysis and cost of calibrating to the wrong scale (live inflation)
  C  asymmetric intervals: free midpoint vs midpoint fixed at mean/median (skew-normal)
  D  heterogeneous population: log s ~ N(log s_med, sl^2); a scale predictor with
     corr(log s, log s_hat) = rho; E(rho) under the Bayes-optimal h(s_hat).
Prints markdown tables.  Pure numpy/scipy; no data.
"""
import numpy as np
from scipy import stats, optimize

SIG = 0.0563870259

FAMS = {
    "gauss": stats.norm(0, 1),
    "laplace": stats.laplace(0, 1 / np.sqrt(2)),
    "t3": stats.t(3, 0, 1 / np.sqrt(3)),   # unit s.d.
}


def cover_sym(fam, x):
    """P(|Z| <= x) for a symmetric unit-sd family."""
    return 2 * fam.cdf(x) - 1


def opt_sym(fam, k):
    """max_x exp(-2 k x) (2F(x)-1); returns (x*, R*). k = s/sig, h = x s."""
    f = lambda x: -np.exp(-2 * k * x) * cover_sym(fam, x)
    xs = np.geomspace(1e-4, 60, 4000)
    v = f(xs)
    i = int(np.argmin(v))
    lo, hi = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    r = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded",
                                 options=dict(xatol=1e-10))
    return float(r.x), float(-r.fun)


def logabs_var(fam, n=2_000_000, seed=0):
    z = fam.rvs(size=n, random_state=np.random.default_rng(seed))
    return float(np.var(np.log(np.abs(z))))


def partA():
    print("\n## A. Optimal symmetric interval for a KNOWN scale s (k = s/sigma)\n")
    ks = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
    for name, fam in FAMS.items():
        print(f"\n### {name}  (s = standard deviation of the error)\n")
        print("| k=s/sig | s | h*/sig | h* | h*/s | coverage | R* = E-contribution |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for k in ks:
            x, R = opt_sym(fam, k)
            print(f"| {k:g} | {k*SIG:.5f} | {k*x:.4f} | {k*x*SIG:.5f} | {x:.3f} | "
                  f"{cover_sym(fam, x):.4f} | {R:.4f} |")
    # closed forms for Laplace
    print("\nLaplace closed form (b = s/sqrt2, q = 2b/sig): h* = b ln(1+sig/(2b)),"
          " R* = (q/(1+q))^q / (1+q).")
    for k in [0.05, 0.1, 0.2, 1.0]:
        b = k * SIG / np.sqrt(2)
        q = 2 * b / SIG
        h = b * np.log(1 + SIG / (2 * b))
        R = (q / (1 + q)) ** q / (1 + q)
        x, Rn = opt_sym(FAMS["laplace"], k)
        print(f"  k={k}: closed h*={h:.6f} R*={R:.6f} | numeric h*={k*x*SIG:.6f} R*={Rn:.6f}")


def partB():
    print("\n## B. Give-up regime and cost of a mis-scaled interval\n")
    print("dR/dh at h=0+ equals 2 f_eps(0) > 0 for every density with f(0)>0, so a zero-width")
    print("interval is NEVER optimal; as s/sig -> inf the optimum tends to h* = sig/2 and")
    print("R* -> f_eps(0)*sig/e (flat-density limit).  Check (gauss):")
    for k in [2, 5, 20, 100]:
        x, R = opt_sym(FAMS["gauss"], k)
        f0 = stats.norm.pdf(0) / (k * SIG)
        print(f"  k={k:>4}: h*/sig={k*x:.4f}  R*={R:.5f}  f(0)*sig/e={f0*SIG/np.e:.5f}")
    print("\nReward ratio R(h fitted for scale m*s) / R*(s): using a half-width calibrated to"
          " the wrong scale (m<1: calibrated on errors that are smaller than the truth,"
          " i.e. local residuals when live is inflated by 1/m).\n")
    ms = [0.33, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]
    print("| family | k=s/sig | " + " | ".join(f"m={m:g}" for m in ms) + " |")
    print("|---|---:|" + "---:|" * len(ms))
    for name, fam in FAMS.items():
        for k in [0.05, 0.1, 0.2, 0.5]:
            x_true, R_true = opt_sym(fam, k)
            row = []
            for m in ms:
                xm, _ = opt_sym(fam, k * m)
                h = xm * k * m                      # h/sig chosen for scale k*m
                R = np.exp(-2 * h) * cover_sym(fam, h / k)
                row.append(R / R_true)
            print(f"| {name} | {k:g} | " + " | ".join(f"{v:.3f}" for v in row) + " |")


def skewnorm_std(a):
    d = a / np.sqrt(1 + a * a)
    mean = d * np.sqrt(2 / np.pi)
    sd = np.sqrt(1 - 2 * d * d / np.pi)
    return stats.skewnorm(a, loc=-mean / sd, scale=1 / sd)   # zero mean, unit sd


def partC():
    print("\n## C. Asymmetric intervals (skew-normal errors, zero mean, unit sd)\n")
    print("An interval [p-a, p+b] is the symmetric interval about its midpoint p+(b-a)/2, so")
    print("'asymmetric about the prediction' == 'symmetric about a shifted centre'.  The")
    print("question is only how much a FREE midpoint beats a midpoint fixed at mean/median.\n")
    print("| shape a | skewness | k | R* midpoint=mean | R* midpoint=median | R* free (HDI) | gain vs median |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for a in [0.5, 1.0, 2.0, 4.0]:
        fam = skewnorm_std(a)
        skw = float(fam.stats(moments="s"))
        med = float(fam.median())
        for k in [0.1, 0.3]:
            s = k * SIG

            def R(c, h):
                return np.exp(-2 * h / SIG) * (fam.cdf((c + h) / s) - fam.cdf((c - h) / s))

            def best_h(c):
                r = optimize.minimize_scalar(lambda h: -R(c, h), bounds=(1e-7, SIG),
                                             method="bounded", options=dict(xatol=1e-12))
                return -r.fun
            Rmean, Rmed = best_h(0.0), best_h(med * s)
            r2 = optimize.minimize(lambda v: -R(v[0], abs(v[1])), x0=[med * s, 0.3 * SIG],
                                   method="Nelder-Mead",
                                   options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
            Rfree = -r2.fun
            print(f"| {a:g} | {skw:.3f} | {k:g} | {Rmean:.5f} | {Rmed:.5f} | {Rfree:.5f} | "
                  f"{Rfree/Rmed-1:+.3%} |")


def mixture_E(fam, k_med, sl, rho, n_gh=48, n_h=1600):
    """E under Bayes-optimal h(s_hat) when log s ~ N(log(k_med*sig), sl^2) and the
    predictor index zeta has corr rho with xi = standardised log s."""
    gx, gw = np.polynomial.hermite_e.hermegauss(n_gh)
    gw = gw / gw.sum()
    hs = np.geomspace(1e-6, 0.2, n_h)[:, None]                  # candidate half-widths
    Etot = 0.0
    hopt = []
    for z, wz in zip(gx, gw):                                    # predictor value
        if rho >= 1.0:
            xis, wxi = np.array([z]), np.array([1.0])
        else:
            xis = rho * z + np.sqrt(1 - rho * rho) * gx
            wxi = gw
        s = k_med * SIG * np.exp(sl * xis)[None, :]
        P = (cover_sym(fam, hs / s) * wxi[None, :]).sum(1)
        Rh = np.exp(-2 * hs[:, 0] / SIG) * P
        j = int(np.argmax(Rh))
        Etot += wz * Rh[j]
        hopt.append(hs[j, 0])
    return Etot, np.array(hopt)


def realized_oracle(fam, k_med, sl, n=400_000, seed=1):
    rng = np.random.default_rng(seed)
    s = k_med * SIG * np.exp(sl * rng.standard_normal(n))
    z = fam.rvs(size=n, random_state=rng)
    return float(np.mean(np.exp(-2 * np.abs(s * z) / SIG)))


def partD():
    print("\n## D. Population of scales + imperfect scale predictor (rho = corr(log s, log s_hat))\n")
    vz = {n: logabs_var(f) for n, f in FAMS.items()}
    print("Var(log|Z|): " + ", ".join(f"{n} {v:.4f}" for n, v in vz.items())
          + "   (gauss exact pi^2/8 = 1.2337; laplace pi^2/6 = 1.6449)")
    rhos = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    print("\n| family | k_med | sl | " + " | ".join(f"rho={r:g}" for r in rhos)
          + " | realized-oracle | corr(log s,log|e|) |")
    print("|---|---:|---:|" + "---:|" * (len(rhos) + 2))
    for name in ["gauss", "laplace", "t3"]:
        fam = FAMS[name]
        for k_med in [0.05, 0.1, 0.2]:
            for sl in [0.5, 0.75, 1.0]:
                Es = [mixture_E(fam, k_med, sl, r)[0] for r in rhos]
                ro = realized_oracle(fam, k_med, sl)
                cs = sl / np.sqrt(sl * sl + vz[name])
                print(f"| {name} | {k_med:g} | {sl:g} | " + " | ".join(f"{e:.4f}" for e in Es)
                      + f" | {ro:.4f} | {cs:.3f} |")


if __name__ == "__main__":
    partA()
    partB()
    partC()
    partD()
