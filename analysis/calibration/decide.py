"""THE decision computation: what is the WORST-CASE gain over every calibration
consistent with the evidence?

  constraint 1: the ORIGINAL checkpoint's 4 anchors pin the RATIO a_u/a_v.
  constraint 2: the SOUP's 2 anchors pin the LEVEL, given a ratio.

A candidate is "demonstrably >= 79" only if the MINIMUM predicted final over the
whole consistent set clears 79.0 -- not the central estimate.

Also runs GATE 4B (must beat the best constant on held-out data) under every
calibration in the set, since a policy that fails it anywhere is fragile.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
W = 0.684593; BANKED = 78.4566

# ---------- ORIGINAL checkpoint: constrain the ratio ----------
zo = np.load(f"{B}/train_mvpe/runs/orig_arrays.npz")
od = {}
for nm in ("u","v"):
    err = zo[f"err_{nm}"].astype(np.float64); ap = zo[f"ap_{nm}"].astype(np.float64)
    od[nm] = dict(errs=np.sort(err), n=err.size)
    r = err/np.maximum(ap,1e-12); w = np.exp(-2*0.05*ap/SIG); o = np.argsort(r)
    od[nm]["rs"] = r[o]; od[nm]["wcum"] = np.concatenate([[0.0], np.cumsum(w[o])])
NO = od["u"]["n"] + od["v"]["n"]; WO = 0.678438
OANCH = [("prop", 0.05, 14.08), ("const", (0.107537,0.010307), 18.30),
         ("const", (0.030,0.010), 29.84), ("const", (0.0129,0.0098), 33.08)]
def E_orig(au, av, kind, cfg):
    t = 0.0
    for nm, a in (("u",au),("v",av)):
        c = od[nm]
        if kind=="const":
            h = cfg[0] if nm=="u" else cfg[1]
            t += np.exp(-2*h/SIG)*np.searchsorted(c["errs"], h/a, side="right")
        else:
            t += c["wcum"][np.searchsorted(c["rs"], cfg/a, side="right")]
    return t/NO
OTG = [s/(100*WO) for _,_,s in OANCH]
def orig_resid(au, av):
    return max(abs(E_orig(au,av,k,c)-t) for (k,c,_),t in zip(OANCH,OTG))

AU = np.linspace(0.8, 3.5, 109); AV = np.linspace(1.5, 5.5, 161)
best_o = min(((orig_resid(au,av), au, av) for au in AU for av in AV), key=lambda x: x[0])
print("ORIGINAL best fit: a_u %.3f a_v %.3f ratio %.3f  max|resid| %.4f"
      % (best_o[1], best_o[2], best_o[1]/best_o[2], best_o[0]))
TOL_O = best_o[0]*1.5
ratios = sorted({round((au/av),3) for au in AU for av in AV if orig_resid(au,av) <= TOL_O})
print("ratios consistent with the ORIGINAL's 4 anchors (resid <= %.4f): %.3f .. %.3f (%d values)"
      % (TOL_O, min(ratios), max(ratios), len(ratios)))

# ---------- SOUP: fit level per ratio, then evaluate ----------
zs = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
srt, bins = {}, {}
for ci, nm in ((0,"u"),(1,"v")):
    h = zs[f"held_{nm}"]
    err = zs[f"err_{nm}"][h][::SUBS].astype(np.float64)
    srt[nm] = np.sort(err)
    b = np.digitize(zs[f"mut_{nm}"][h][::SUBS], ED[ci])
    bins[nm] = [np.sort(err[b==k]) for k in range(NB)]
nu, nv = srt["u"].size, srt["v"].size; NS = nu+nv
def E_lut(au,av,lut=LUT,mult=1.0):
    t=0.0
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[nm]):
            if s.size==0: continue
            h=float(lut[k,ci])*mult
            t += np.exp(-2*h/SIG)*np.searchsorted(s,h/a,side="right")
    return t/NS
def best_const_E(au,av):
    g=np.linspace(0.002,0.040,381)
    bu=max(g,key=lambda h: np.exp(-2*h/SIG)*np.searchsorted(srt["u"],h/au,side="right")/nu)
    bv=max(g,key=lambda h: np.exp(-2*h/SIG)*np.searchsorted(srt["v"],h/av,side="right")/nv)
    return (np.exp(-2*bu/SIG)*np.searchsorted(srt["u"],bu/au,side="right")
           +np.exp(-2*bv/SIG)*np.searchsorted(srt["v"],bv/av,side="right"))/NS
def refit(au,av):
    nl=LUT.copy()
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[nm]):
            if s.size<200: continue
            v=a*s; kk=np.arange(1,v.size+1)/v.size
            nl[k,ci]=float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    return nl

rows=[]
for R in ratios:
    bb=None
    for L in np.linspace(0.8,6.0,521):
        au,av=L*R,L
        c=max(abs(E_lut(au,av)-0.5020), abs(E_lut(au,av,mult=1.25)-0.4939))
        if bb is None or c<bb[0]: bb=(c,au,av)
    c,au,av=bb
    if c>0.010: continue                      # soup anchors not matched
    nl=refit(au,av); g=E_lut(au,av,nl)-E_lut(au,av)
    rows.append((R,au,av,c,g,BANKED+100*W*g*0.217, E_lut(au,av,nl)-best_const_E(au,av)))
print("\nratio  a_u    a_v    soupfit    dE      final    GATE4B(incr vs best const)")
print("-"*74)
for r in rows:
    print("%.3f  %5.3f  %5.3f  %7.4f  %+.4f  %7.4f   %+.4f %s"
          % (r[0],r[1],r[2],r[3],r[4],r[5],r[6],"PASS" if r[6]>0 else "FAIL"))
if rows:
    fins=[r[5] for r in rows]
    print("\nCONSISTENT SET: %d calibrations" % len(rows))
    print("  predicted final  min %.4f   median %.4f   max %.4f" % (min(fins), float(np.median(fins)), max(fins)))
    print("  GATE 4B passes in %d/%d" % (sum(1 for r in rows if r[6]>0), len(rows)))
    print("\n  DEMONSTRABLY >= 79 ?  %s  (worst case %.4f)"
          % ("YES" if min(fins)>=79.0 else "NO", min(fins)))
