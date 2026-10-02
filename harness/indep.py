"""Contamination-free test: does |pred| carry information about the error?

Fit Weibull(k, lam) per channel to the two CONSTANT real anchors ONLY, ASSUMING
error is independent of |pred|. Then predict the PROPORTIONAL anchor (h=0.05|pred|),
which we also know for real (E 0.2076).
  - prediction ~= 0.2076  -> error is independent of |pred|; conditioning on it is useless
  - prediction != 0.2076  -> there IS real signal in |pred|, usable with NO contamination
|pred| is used only as an input; it never touches the ground truth, so this test is clean."""
import numpy as np
SIG=0.0563870259
d=np.load("/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
SCM=(Y[...,:2]!=0.0); AP=np.abs(P[...,:2])
au=AP[...,0][SCM[...,0]][::37]; av=AP[...,1][SCM[...,1]][::37]
print("|pred| sample: u n=%d med %.4f | v n=%d med %.4f"%(au.size,np.median(au),av.size,np.median(av)))
def Ec(k,lu,lv,hu,hv):
    Fu=1-np.exp(-(hu/lu)**k); Fv=1-np.exp(-(hv/lv)**k)
    return 0.5*(np.exp(-2*hu/SIG)*Fu+np.exp(-2*hv/SIG)*Fv)
best=None
for k in np.linspace(0.6,2.2,65):
    for lu in np.linspace(0.006,0.030,97):
        for lv in np.linspace(0.003,0.016,53):
            r=abs(Ec(k,lu,lv,0.030,0.010)-0.4399)+abs(Ec(k,lu,lv,0.0129,0.0098)-0.4876)
            if best is None or r<best[0]: best=(r,k,lu,lv)
r,k,lu,lv=best
print("fit on CONSTANT anchors only: k=%.2f lam_u=%.4f lam_v=%.4f (resid %.4f)"%(k,lu,lv,r))
hu=0.05*au; hv=0.05*av
Fu=1-np.exp(-(hu/lu)**k); Fv=1-np.exp(-(hv/lv)**k)
Ep=(np.sum(np.exp(-2*hu/SIG)*Fu)+np.sum(np.exp(-2*hv/SIG)*Fv))/(au.size+av.size)
print()
print("PREDICTED E at h=0.05|pred| assuming independence : %.4f"%Ep)
print("REAL      E at h=0.05|pred| (leaderboard anchor)   : %.4f"%0.2076)
print("difference %+.4f"%(Ep-0.2076))
if abs(Ep-0.2076)<0.02:
    print("=> consistent with INDEPENDENCE: |pred| carries little/no usable signal")
else:
    print("=> INCONSISTENT: |pred| and error ARE related -> exploitable with zero contamination")
