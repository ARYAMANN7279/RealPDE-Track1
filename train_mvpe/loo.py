"""Leave-one-anchor-out: can the fitted model predict E at a bound setting it
has never seen? That is precisely what a REFITTED LUT is, so if LOO fails the
+0.037 projection is not trustworthy and nothing should be submitted.

Scalar model only (2 free params), so 3 anchors still over-determine it.
"""
import numpy as np, itertools
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
z = np.load(f"{B}/train_mvpe/runs/arrays.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]

ch = {}
for ci, nm in ((0, "u"), (1, "v")):
    h = z[f"held_{nm}"]
    err = z[f"err_{nm}"][h][::SUBS].astype(np.float64)
    mu = z[f"mut_{nm}"][h][::SUBS]
    b = np.digitize(mu, ED[ci])
    ch[nm] = dict(all=np.sort(err), bins=[np.sort(err[b == k]) for k in range(NB)], ci=ci)
NTOT = ch["u"]["all"].size + ch["v"]["all"].size
F = lambda s, t: (np.searchsorted(s, t, side="right")/s.size if s.size else 0.0)

def E_const(p, hu, hv):
    au, av = p; t = 0.0
    for nm, h, a in (("u", hu, au), ("v", hv, av)):
        c = ch[nm]; t += c["all"].size*np.exp(-2*h/SIG)*F(c["all"], h/a)
    return t/NTOT

def E_lut(p, mult=1.0):
    au, av = p; t = 0.0
    for nm, ci, a in (("u", 0, au), ("v", 1, av)):
        for k, s in enumerate(ch[nm]["bins"]):
            if s.size == 0: continue
            h = float(LUT[k, ci])*mult
            t += s.size*np.exp(-2*h/SIG)*F(s, h/a)
    return t/NTOT

ANCH = [("const wide",   lambda p: E_const(p, 0.030, 0.010),  0.4399),
        ("const banked", lambda p: E_const(p, 0.0129, 0.0098), 0.4876),
        ("LUT x1.00",    lambda p: E_lut(p, 1.00),             0.5020),
        ("LUT x1.25",    lambda p: E_lut(p, 1.25),             0.4939)]

AU = np.linspace(1.2, 4.0, 113); AV = np.linspace(1.0, 4.5, 141)
def fit(idxs):
    best = None
    for au in AU:
        for av in AV:
            p = (au, av)
            c = max(abs(ANCH[i][1](p) - ANCH[i][2]) for i in idxs)
            if best is None or c < best[0]: best = (c, p)
    return best[1], best[0]

print("leave-one-anchor-out (scalar model, 2 params fitted on 3 anchors)")
print("%-14s %10s %10s %9s   %s" % ("held-out", "predicted", "actual", "error", "fitted a_u/a_v"))
print("-"*70)
errs = []
for i in range(4):
    idxs = [j for j in range(4) if j != i]
    p, c = fit(idxs)
    pred = ANCH[i][1](p); act = ANCH[i][2]
    errs.append(abs(pred-act))
    print("%-14s %10.4f %10.4f %+9.4f   %.3f / %.3f" % (ANCH[i][0], pred, act, pred-act, *p))
print("\nmax LOO error %.4f | mean %.4f" % (max(errs), np.mean(errs)))
print("predicted gain from refitting the LUT: +0.0371 E")
print("=> gain is %.1fx the worst LOO error" % (0.0371/max(errs)))

# how much does the predicted gain move across params consistent with the anchors?
pfull, cfull = fit([0,1,2,3])
print("\nfull fit: a_u %.3f a_v %.3f (max|resid| %.4f)" % (*pfull, cfull))
tol = cfull*1.5
ok = [(au,av) for au in AU for av in AV
      if max(abs(ANCH[i][1]((au,av))-ANCH[i][2]) for i in range(4)) <= tol]
print("params within 1.5x that residual: %d grid points" % len(ok))
def gain(p):
    au, av = p; nl = LUT.copy()
    for nm, ci, a in (("u",0,au), ("v",1,av)):
        for k, s in enumerate(ch[nm]["bins"]):
            if s.size < 200: continue
            v = a*s
            kk = np.arange(1, v.size+1)/v.size
            nl[k,ci] = float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    def E(lut):
        t=0.0
        for nm,ci,a2 in (("u",0,au),("v",1,av)):
            for k,s in enumerate(ch[nm]["bins"]):
                if s.size==0: continue
                h=float(lut[k,ci]); t+=s.size*np.exp(-2*h/SIG)*F(s,h/a2)
        return t/NTOT
    return E(nl)-E(LUT)
gs = [gain(p) for p in ok[::max(1,len(ok)//40)]]
print("predicted dE across those: min %+.4f  median %+.4f  max %+.4f" % (min(gs), float(np.median(gs)), max(gs)))
W=0.684593
print("  -> dfinal range %+.4f .. %+.4f  (final %.3f .. %.3f)"
      % (100*W*min(gs)*0.217, 100*W*max(gs)*0.217, 78.4566+100*W*min(gs)*0.217, 78.4566+100*W*max(gs)*0.217))
