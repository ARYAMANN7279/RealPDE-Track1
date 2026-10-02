"""Is correcting BOTH channels better than correcting only v, risk-adjusted?

The v-scale error is established three ways. The u-scale correction (which NARROWS
h_u by 13-25%) rests on the same fit but has no independent confirmation, and
narrowing is the more dangerous direction: it forfeits coverage that cannot be
recovered. Compare shipped / v-only / both across the full true-ratio range.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
W = 0.684593; BANKED = 78.4566
z = np.load(f"{B}/train_mvpe/runs/arrays2.npz")
A = np.load(f"{B}/_tmp/soupv1/head_assets.npz"); LUT = A["LUT"]; ED = A["ED"]; NB = LUT.shape[0]
srt, bins = {}, {}
for ci, nm in ((0,"u"),(1,"v")):
    h = z[f"held_{nm}"]
    err = z[f"err_{nm}"][h][::SUBS].astype(np.float64)
    srt[nm] = np.sort(err)
    b = np.digitize(z[f"mut_{nm}"][h][::SUBS], ED[ci])
    bins[nm] = [np.sort(err[b==k]) for k in range(NB)]
nu, nv = srt["u"].size, srt["v"].size; NS = nu+nv
def E_lut(au,av,lut,mult=1.0):
    t=0.0
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[nm]):
            if s.size==0: continue
            h=float(lut[k,ci])*mult
            t += np.exp(-2*h/SIG)*np.searchsorted(s,h/a,side="right")
    return t/NS
def fit_level(R):
    bb=None
    for L in np.linspace(0.8,6.5,571):
        au,av=L*R,L
        c=max(abs(E_lut(au,av,LUT)-0.5020), abs(E_lut(au,av,LUT,1.25)-0.4939))
        if bb is None or c<bb[0]: bb=(c,au,av)
    return bb[1],bb[2]
def refit(au,av,which):
    nl=LUT.copy()
    for nm,ci,a in (("u",0,au),("v",1,av)):
        if which=="v" and nm=="u": continue
        for k,s in enumerate(bins[nm]):
            if s.size<200: continue
            v=a*s; kk=np.arange(1,v.size+1)/v.size
            nl[k,ci]=float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    return nl

BUILD_R = 0.570
au_b, av_b = fit_level(BUILD_R)
CANDS = [("shipped", LUT),
         ("v-only corrected", refit(au_b, av_b, "v")),
         ("both corrected (current build)", refit(au_b, av_b, "both"))]
TRUE_R = [0.371, 0.45, 0.484, 0.57, 0.70, 0.90, 1.20, 1.937, 2.50]
cals = [(R,)+fit_level(R) for R in TRUE_R]
print("h_u med: shipped %.4f | v-only %.4f | both %.4f"
      % (np.median(LUT[:,0]), np.median(CANDS[1][1][:,0]), np.median(CANDS[2][1][:,0])))
print("h_v med: shipped %.4f | v-only %.4f | both %.4f\n"
      % (np.median(LUT[:,1]), np.median(CANDS[1][1][:,1]), np.median(CANDS[2][1][:,1])))
hdr = "%-32s" % "candidate \\ true ratio" + "".join("%8.2f" % R for R,_,_ in cals)
print(hdr); print("-"*len(hdr))
base = {R: E_lut(au,av,LUT) for R,au,av in cals}
res={}
for nm, lut in CANDS:
    cells=[BANKED + 100*W*(E_lut(au,av,lut)-base[R])*0.217 for R,au,av in cals]
    res[nm]=cells
    print("%-32s" % nm + "".join("%8.3f" % x for x in cells))
print("\n%-32s %10s %10s %10s" % ("", "worst(all)", "worst(<=0.57)", "median(<=0.57)"))
for nm in res:
    sub=[c for (R,_,_),c in zip(cals,res[nm]) if R<=0.570]
    print("%-32s %10.3f %10.3f %10.3f" % (nm, min(res[nm]), min(sub), float(np.median(sub))))
