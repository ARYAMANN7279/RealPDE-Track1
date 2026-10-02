"""Which LUT should actually ship?

A LUT built under calibration X but evaluated under the true calibration Y is not the
same as one built and evaluated under Y. So build under each candidate calibration,
evaluate every build under every calibration, and choose the build with the best
WORST-CASE outcome. That is the right criterion when the calibration is uncertain.
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
    b = np.digitize(zs[f"mut_{nm}"][h][::SUBS], ED[ci])
    bins[nm] = [np.sort(err[b==k]) for k in range(NB)]
nu, nv = srt["u"].size, srt["v"].size; NS = nu+nv

def E_lut(au,av,lut):
    t=0.0
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[nm]):
            if s.size==0: continue
            h=float(lut[k,ci])
            t += np.exp(-2*h/SIG)*np.searchsorted(s,h/a,side="right")
    return t/NS
def refit(au,av):
    nl=LUT.copy()
    for nm,ci,a in (("u",0,au),("v",1,av)):
        for k,s in enumerate(bins[nm]):
            if s.size<200: continue
            v=a*s; kk=np.arange(1,v.size+1)/v.size
            nl[k,ci]=float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])
    return nl

# calibrations: ratio from the ORIGINAL's consistent range, level fitted to soup anchors
def fit_level(R):
    bb=None
    for L in np.linspace(0.8,6.5,571):
        au,av=L*R,L
        c=max(abs(E_lut(au,av,LUT)-0.5020),
              abs(E_lut(au,av,LUT*1.25)-0.4939))
        if bb is None or c<bb[0]: bb=(c,au,av)
    return bb[1],bb[2],bb[0]

CAL_R = [0.371, 0.45, 0.484, 0.55, 0.570, 0.70, 0.90, 1.20, 1.937, 2.50]
CALS = []
for R in CAL_R:
    au,av,c = fit_level(R); CALS.append((R,au,av,c))
BUILD_R = [0.371, 0.484, 0.570, 0.70, 0.90]
builds = {R: refit(*fit_level(R)[:2]) for R in BUILD_R}
builds["shipped"] = LUT

print("rows = LUT built under this ratio | cols = TRUE calibration ratio")
print("cells = predicted final\n")
hdr = "%-12s" % "build\\true" + "".join("%9.2f" % c[0] for c in CALS)
print(hdr); print("-"*len(hdr))
base = {}
for R,au,av,c in CALS: base[R] = E_lut(au,av,LUT)
rowmin = {}
for bR, lut in builds.items():
    cells=[]
    for R,au,av,c in CALS:
        g = E_lut(au,av,lut) - base[R]
        cells.append(BANKED + 100*W*g*0.217)
    rowmin[bR]=min(cells)
    print("%-12s" % str(bR) + "".join("%9.3f" % x for x in cells))
print("\nworst case over the FULL true-ratio range (0.371 .. 2.50):")
for bR in builds: print("   build %-8s -> %.4f" % (str(bR), rowmin[bR]))
bestb = max((k for k in builds if k!="shipped"), key=lambda k: rowmin[k])
print("\nminimax build: ratio %s  (worst case %.4f)" % (bestb, rowmin[bestb]))

# restrict truth to what the ORIGINAL's anchors allow (0.371..0.570)
sub=[c for c in CALS if 0.371<=c[0]<=0.570]
print("\nif the ratio transfers (true ratio in 0.371..0.570):")
for bR,lut in builds.items():
    cells=[BANKED+100*W*(E_lut(au,av,lut)-base[R])*0.217 for R,au,av,c in sub]
    print("   build %-8s -> worst %.4f  median %.4f" % (str(bR), min(cells), float(np.median(cells))))
np.save(f"{B}/train_mvpe/runs/LUT_ship.npy", builds[bestb])
print("\nsaved minimax LUT -> runs/LUT_ship.npy")
print("  h_u median %.4f (shipped %.4f) | h_v median %.4f (shipped %.4f)"
      % (np.median(builds[bestb][:,0]), np.median(LUT[:,0]),
         np.median(builds[bestb][:,1]), np.median(LUT[:,1])))
