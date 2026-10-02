"""Build the candidate: SOUP_v1 with ONLY the LUT array replaced.

Minimal change by construction -- same checkpoint, same head weights, same feature
stats, same bin edges, same submission.py. Only head_assets.npz['LUT'] differs, so the
zip entry list is guaranteed identical to the artifact that really scored 78.4566
(sec14: always diff the entry list against a known-good scored zip).
"""
import os, shutil, subprocess, hashlib
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259; SUBS = 4
SRC = f"{B}/submissions/submission_SOUP_v1.zip"
OUT = f"{B}/submissions/submission_LUTFIX.zip"
WORK = f"{B}/_tmp/build_lutfix"

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
            h=float(lut[k,ci]); t += np.exp(-2*h/SIG)*np.searchsorted(s,h/a,side="right")
    return t/NS
R = 0.570                                     # minimax ratio conditional on transfer
bb=None
for L in np.linspace(0.8,6.5,571):
    au,av=L*R,L
    c=max(abs(E_lut(au,av,LUT)-0.5020), abs(E_lut(au,av,LUT*1.25)-0.4939))
    if bb is None or c<bb[0]: bb=(c,au,av)
c,au,av = bb
print("calibration: ratio %.3f -> a_u %.3f a_v %.3f (soup-anchor fit err %.4f)" % (R,au,av,c))
newLUT = LUT.copy()
for nm,ci,a in (("u",0,au),("v",1,av)):
    for k,s in enumerate(bins[nm]):
        if s.size<200: continue
        v=a*s; kk=np.arange(1,v.size+1)/v.size
        newLUT[k,ci]=float(v[int(np.argmax(np.exp(-2*v/SIG)*kk))])

# sanity assertions mirroring 05_build_lut.py
assert newLUT[:,0].min() > 0.005, "LUT u near-zero half-widths: %.5f" % newLUT[:,0].min()
assert newLUT[:,1].min() > 0.0015, "LUT v near-zero half-widths: %.5f" % newLUT[:,1].min()
assert np.all(np.isfinite(newLUT))
assert newLUT.shape == LUT.shape and newLUT.dtype == LUT.dtype
print("h_u %.4f..%.4f med %.4f   (was %.4f..%.4f med %.4f)"
      % (newLUT[:,0].min(), newLUT[:,0].max(), np.median(newLUT[:,0]),
         LUT[:,0].min(), LUT[:,0].max(), np.median(LUT[:,0])))
print("h_v %.4f..%.4f med %.4f   (was %.4f..%.4f med %.4f)"
      % (newLUT[:,1].min(), newLUT[:,1].max(), np.median(newLUT[:,1]),
         LUT[:,1].min(), LUT[:,1].max(), np.median(LUT[:,1])))
# monotone in bin index? the LUT encodes "predicted error -> width"
for ci,nm in ((0,"u"),(1,"v")):
    d = np.diff(newLUT[:,ci]); print("  %s monotone non-decreasing: %s (%d violations)"
        % (nm, bool((d>=-1e-6).all()), int((d<-1e-6).sum())))

if os.path.exists(WORK): shutil.rmtree(WORK)
os.makedirs(WORK)
subprocess.run(["unzip","-q",SRC,"-d",WORK], check=True)
z = dict(np.load(f"{WORK}/head_assets.npz"))
z["LUT"] = newLUT.astype(LUT.dtype)
np.savez_compressed(f"{WORK}/head_assets.npz", **z)
if os.path.exists(OUT): os.remove(OUT)
subprocess.run(["zip","-q","-r","-X",OUT,"."], cwd=WORK, check=True)
print("\nwrote", OUT)
print("md5", hashlib.md5(open(OUT,"rb").read()).hexdigest())
a = subprocess.run(["unzip","-l",SRC], capture_output=True, text=True).stdout.split("\n")
b = subprocess.run(["unzip","-l",OUT], capture_output=True, text=True).stdout.split("\n")
na = sorted(l.split()[-1] for l in a[3:-3] if l.strip())
nb = sorted(l.split()[-1] for l in b[3:-3] if l.strip())
print("entry list identical to SOUP_v1:", na == nb, "(%d entries)" % len(nb))
if na != nb:
    print("  only in SOUP_v1:", set(na)-set(nb)); print("  only in new:", set(nb)-set(na))
tot = subprocess.run(["unzip","-l",OUT], capture_output=True, text=True).stdout.strip().split("\n")[-1]
print("extracted size line:", tot.strip(), "(limit 256 MB = 268435456)")
np.save(f"{B}/train_mvpe/runs/LUT_ship.npy", newLUT)
