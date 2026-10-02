"""What predictor quality is needed to reach competitor-level headroom capture?

Our current feature's 24-bin ceiling is 18-22%. Competitors reach 33-43%. So the
question is not "fix the LUT" but "what signal do we need". Simulate predictors of
known Spearman rho against the true |error| and measure the fraction each can capture.
That converts an abstract target into a concrete spec for the uncertainty head.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 16
z = np.load(f"{B}/train_mvpe/runs/arrays3.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); ED = A["ED"]
rng = np.random.default_rng(0)
dat = {}
for ci, nm in ((0,"u"),(1,"v")):
    m = z[f"held_{nm}"]
    dat[nm] = (z[f"err_{nm}"][m][::SUBS].astype(np.float64),
               z[f"mut_{nm}"][m][::SUBS].astype(np.float64),
               z[f"muf_{nm}"][m][::SUBS].astype(np.float64))
N = sum(v[0].size for v in dat.values())

def frac_of(scorer, NB=24):
    """captured fraction for a per-element score, using an optimal NB-bin lookup"""
    E_or = E_bc = E_p = 0.0
    for nm in dat:
        err = dat[nm][0]
        E_or += float(np.exp(-2*err/SIG).sum())
        e = np.sort(err); g = np.linspace(0.0005, 0.05, 700)
        k = np.searchsorted(e, g, side="right")
        E_bc += float(np.max(np.exp(-2*g/SIG)*k))
        s = scorer(nm)
        q = np.quantile(s, np.linspace(0,1,NB+1)[1:-1]); b = np.digitize(s, q)
        for kk in range(NB):
            ss = np.sort(err[b==kk])
            if ss.size == 0: continue
            gg = np.linspace(0.0005, 0.05, 500); c = np.searchsorted(ss, gg, side="right")
            E_p += float(np.max(np.exp(-2*gg/SIG)*c))
    return 100*(E_p-E_bc)/(E_or-E_bc), E_p/N, E_bc/N, E_or/N

print("feature                                    24-bin ceiling")
print("-"*58)
f0,_,bc,orc = frac_of(lambda nm: dat[nm][1]); print("%-42s %6.1f%%" % ("current head, time-averaged (mut)", f0))
f1,_,_,_    = frac_of(lambda nm: dat[nm][2]); print("%-42s %6.1f%%" % ("current head, per-frame (muf)", f1))
f2,_,_,_    = frac_of(lambda nm: rng.random(dat[nm][0].size)); print("%-42s %6.1f%%" % ("random (sanity, should be ~0)", f2))

print("\n--- required predictor quality (log-space corr with true |err|) ---")
print("%-10s %10s %14s" % ("corr", "24-bin frac", "-> real final"))
print("-"*38)
W = 0.684593
for rho_t in (0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,1.00):
    def sc(nm, rho=rho_t):
        L = np.log(dat[nm][0]+1e-9); L = (L-L.mean())/(L.std()+1e-12)
        return rho*L + np.sqrt(max(0.0,1-rho*rho))*rng.standard_normal(L.size)
    fr,_,_,_ = frac_of(sc)
    E_real = 0.4876 + (fr/100)*(0.7775-0.4876)
    print("%-10.2f %9.1f%% %14.3f" % (rho_t, fr, 78.4566+0.217*(100*W*E_real-34.3656)))
print("\nour head's actual log-space corr with |err|:")
for nm in dat:
    L = np.log(dat[nm][0]+1e-9)
    print("   %s: mut %.3f   muf %.3f" % (nm, np.corrcoef(L, dat[nm][1])[0,1], np.corrcoef(L, dat[nm][2])[0,1]))
print("\ntargets: benslash2 33%% (final 79.66) | np-user 43%% (final 80.09)")
