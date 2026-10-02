"""origfit, with every anchor reduced to an O(log n) CDF lookup.

For a scalar map err_real = a*err_local:
  constant band h:   1[a*err <= h]      == 1[err <= h/a]
  default band 0.05*|pred|: 1[a*err <= 0.05*ap] == 1[err/ap <= 0.05/a]
The exp(-2h/sigma) weight does not depend on a in either case (h is fixed, or a
function of ap only), so pre-sorting by err (resp. err/ap) and pre-summing the
weights makes each anchor a single searchsorted. Grid search then costs nothing.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259
z = np.load(f"{B}/train_mvpe/runs/orig_arrays.npz")
d = {}
for nm in ("u", "v"):
    err = z[f"err_{nm}"].astype(np.float64); ap = z[f"ap_{nm}"].astype(np.float64)
    o = np.argsort(err)
    d[nm] = dict(errs=err[o], n=err.size)
    r = err/np.maximum(ap, 1e-12); w = np.exp(-2*0.05*ap/SIG)
    o2 = np.argsort(r)
    d[nm]["rs"] = r[o2]; d[nm]["wcum"] = np.concatenate([[0.0], np.cumsum(w[o2])])
N = d["u"]["n"] + d["v"]["n"]
W = 0.678438
ANCH = [("default 0.05*|pred|", "prop",  0.05,                  14.08),
        ("calibrated",          "const", (0.107537, 0.010307),  18.30),
        ("plain_sps",           "const", (0.030, 0.010),        29.84),
        ("ROBUST",              "const", (0.0129, 0.0098),      33.08)]
TG = [s/(100*W) for *_, s in ANCH]

def E_model(p, kind, cfg):
    au, av = p; tot = 0.0
    for nm, a in (("u", au), ("v", av)):
        c = d[nm]
        if kind == "const":
            h = cfg[0] if nm == "u" else cfg[1]
            k = np.searchsorted(c["errs"], h/a, side="right")
            tot += np.exp(-2*h/SIG) * k
        else:
            k = np.searchsorted(c["rs"], cfg/a, side="right")
            tot += c["wcum"][k]
    return tot / N

AU = np.linspace(0.6, 5.0, 441); AV = np.linspace(0.6, 5.0, 441)
def fit(idxs):
    best = None
    for au in AU:
        for av in AV:
            p = (au, av)
            c = max(abs(E_model(p, ANCH[i][1], ANCH[i][2]) - TG[i]) for i in idxs)
            if best is None or c < best[0]: best = (c, p)
    return best[1], best[0]

print("real E targets:", " ".join("%s=%.4f" % (a[0].split()[0], t) for a, t in zip(ANCH, TG)))
p_all, c_all = fit(range(4))
print("\nFULL FIT (4 anchors)   a_u %.3f  a_v %.3f   ratio %.3f   max|resid| %.4f"
      % (*p_all, p_all[0]/p_all[1], c_all))
for i, (nm, k, cfg, sps) in enumerate(ANCH):
    print("   %-20s target %.4f  model %.4f  resid %+.4f" % (nm, TG[i], E_model(p_all, k, cfg), E_model(p_all, k, cfg)-TG[i]))

print("\nwhat global_scale chose for this checkpoint: a_u 2.320  a_v 1.198  ratio 1.937")
pg = (2.3199, 1.1982)
for i, (nm, k, cfg, sps) in enumerate(ANCH):
    print("   %-20s target %.4f  model %.4f  resid %+.4f" % (nm, TG[i], E_model(pg, k, cfg), E_model(pg, k, cfg)-TG[i]))
print("   max|resid| %.4f" % max(abs(E_model(pg, k, cfg)-TG[i]) for i,(nm,k,cfg,s) in enumerate(ANCH)))

print("\nLEAVE-ONE-ANCHOR-OUT (fit on 3, predict the 4th)")
errs = []
for i in range(4):
    p, c = fit([j for j in range(4) if j != i])
    pred = E_model(p, ANCH[i][1], ANCH[i][2])
    errs.append(abs(pred - TG[i]))
    print("   %-20s predicted %.4f  actual %.4f  error %+.4f   (fit a_u %.3f a_v %.3f)"
          % (ANCH[i][0], pred, TG[i], pred - TG[i], *p))
print("\n   max LOO error %.4f   mean %.4f" % (max(errs), float(np.mean(errs))))
