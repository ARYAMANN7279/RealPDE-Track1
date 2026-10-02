"""Calibrate the (now correct) local error distribution to the REAL anchors.

Two real anchors exist for this exact checkpoint:
    E = 0.2076  at proportional half-width 0.05*|pred|   (real sps 14.08)
    E = 0.4399  at constant half-widths [0.030, 0.010]   (real sps 29.84)
E = mean over SCORED elements of exp(-width/sigma) * 1[|err| <= h].

One free parameter (error scale lambda), two constraints => FALSIFIABLE.
On the wrong dataset the best single lambda left a residual of 3.44 and the two
anchors pulled in opposite directions.
"""
import numpy as np, os, sys
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
print("cache:",P.shape,flush=True)
err=np.abs(P[...,:C]-Y[...,:C]); sc=(Y[...,:C]!=0.0)
absP=np.abs(P[...,:C])
eu=err[...,0][sc[...,0]]; ev=err[...,1][sc[...,1]]
pu=absP[...,0][sc[...,0]]; pv=absP[...,1][sc[...,1]]
nu,nv=eu.size,ev.size; ntot=nu+nv
print("scored elements: u=%d v=%d (zero-masked fraction %.4f)"%(nu,nv,1-sc.mean()),flush=True)

def E_const(hu,hv,lam):
    return (np.exp(-2*hu/SIG)*np.mean(eu<=hu*lam)*nu + np.exp(-2*hv/SIG)*np.mean(ev<=hv*lam)*nv)/ntot
def E_prop(frac,lam):
    hu=frac*pu; hv=frac*pv
    return (np.sum(np.exp(-2*hu/SIG)*(eu<=hu*lam)) + np.sum(np.exp(-2*hv/SIG)*(ev<=hv*lam)))/ntot

A_PROP,A_CONST=0.2076,0.4399
print("\n=== single-lambda fit to BOTH real anchors ===")
print(" %-8s %-24s %-26s %s"%("lambda","E_prop (target 0.2076)","E_const (target 0.4399)","residual"))
best=None
for lam in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2]:
    a,b=E_prop(0.05,lam),E_const(0.030,0.010,lam)
    r=abs(a-A_PROP)+abs(b-A_CONST)
    print(" %-8.2f %-24.4f %-26.4f %.4f"%(lam,a,b,r))
    if best is None or r<best[1]: best=(lam,r)
lo,hi=0.05,3.0
for _ in range(60):
    m=(lo+hi)/2
    if E_const(0.030,0.010,m)<A_CONST: lo=m
    else: hi=m
lam=(lo+hi)/2
print("\n lambda matching the CONST anchor exactly: %.3f"%lam)
print("   -> E_prop at that lambda = %.4f  (target %.4f, error %+.4f)"%(
    E_prop(0.05,lam),A_PROP,E_prop(0.05,lam)-A_PROP))
print("   [wrong dataset gave a %+.4f error here -> calibration impossible]"%(0.1289-0.2076))
cov=(np.mean(eu<=0.030*lam)*nu+np.mean(ev<=0.010*lam)*nv)/ntot
print("   coverage at [0.030,0.010] after calibration = %.4f (real anchor 0.841)"%cov)
np.save(f"{B}/local_harness/calib_lambda.npy",np.array([lam]))
print("\nsaved lambda -> calib_lambda.npy",flush=True)
