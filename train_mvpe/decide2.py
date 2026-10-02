"""Drop the ratio-transfer assumption entirely.

decide.py restricted ratios to those the ORIGINAL checkpoint's anchors allow. If the
ratio does NOT transfer between models, that restriction is unjustified. So: sweep the
FULL ratio range, constrain ONLY with the SOUP's own two real anchors, and ask again
what the worst case is. If it still clears 79, demonstrability needs no transfer
assumption at all.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
W = 0.684593; BANKED = 78.4566
zs = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
srt, bins = {}, {}
for ci, nm in ((0,"u"),(1,"v")):
    h = zs[f"held_{nm}"]
    err = zs[f"err_{nm}"][h][::SUBS].astype(np.float64)
    srt[nm] = np.sort(err)
    for mk in ("mut","muf"):
        b = np.digitize(zs[f"{mk}_{nm}"][h][::SUBS], ED[ci])
        bins[(nm,mk)] = [np.sort(err[b==k]) for k in range(NB)]
nu, nv = srt["u"].size, srt["v"].size; NS = nu+nv

def E_lut(au,av,lut=LUT,mult=1.0,mk="mut"):
    t=0.0
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[(nm,mk)]):
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
def refit(au,av,mk):
    nl=LUT.copy()
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[(nm,mk)]):
            if s.size<200: continue
            v=a*s; kk=np.arange(1,v.size+1)/v.size
            nl[k,ci]=float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    return nl

RATIOS = np.round(np.arange(0.30, 2.55, 0.05), 3)
recs=[]
for R in RATIOS:
    bb=None
    for L in np.linspace(0.8, 6.5, 571):
        au,av=L*R,L
        c=max(abs(E_lut(au,av)-0.5020), abs(E_lut(au,av,mult=1.25)-0.4939))
        if bb is None or c<bb[0]: bb=(c,au,av)
    recs.append((R,)+bb[1:]+(bb[0],))
bestfit = min(r[3] for r in recs)
print("best achievable soup-anchor fit over all ratios: %.4f" % bestfit)
print("\n%-7s %7s %7s %9s %9s %9s %9s" % ("ratio","a_u","a_v","soupfit","dE(t-avg)","final","GATE4B"))
print("-"*66)
keep=[]
for R,au,av,c in recs:
    nl=refit(au,av,"mut"); g=E_lut(au,av,nl)-E_lut(au,av)
    incr=E_lut(au,av,nl)-best_const_E(au,av)
    fin=BANKED+100*W*g*0.217
    mark = "   <-- excluded" if c > 0.010 else ""
    print("%-7.2f %7.3f %7.3f %9.4f %+9.4f %9.4f %+9.4f%s" % (R,au,av,c,g,fin,incr,mark))
    if c <= 0.010: keep.append((R,au,av,c,g,fin,incr))
fins=[k[5] for k in keep]
print("\nCONSISTENT with the SOUP's OWN anchors alone (fit err <= 0.010): %d ratios, %.2f..%.2f"
      % (len(keep), min(k[0] for k in keep), max(k[0] for k in keep)))
print("  predicted final  min %.4f  median %.4f  max %.4f" % (min(fins), float(np.median(fins)), max(fins)))
print("  GATE 4B passes %d/%d" % (sum(1 for k in keep if k[6]>0), len(keep)))
print("\n  DEMONSTRABLY >= 79 without any transfer assumption?  %s  (worst case %.4f)"
      % ("YES" if min(fins)>=79.0 else "NO", min(fins)))
for tol in (0.008, 0.012, 0.015):
    kk=[r for r in recs if r[3]<=tol]
    if not kk: continue
    ff=[]
    for R,au,av,c in kk:
        nl=refit(au,av,"mut"); ff.append(BANKED+100*W*(E_lut(au,av,nl)-E_lut(au,av))*0.217)
    print("   tolerance %.3f -> %2d ratios (%.2f..%.2f), worst-case final %.4f"
          % (tol, len(kk), min(r[0] for r in kk), max(r[0] for r in kk), min(ff)))
