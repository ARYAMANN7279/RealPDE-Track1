"""What is the MAXIMUM E our current error distribution can support?

Oracle: knowing |err| exactly, set h = |err| -> contribution exp(-2|err|/sigma),
coverage 1. E_oracle = mean(exp(-2|err_real|/sigma)) is a hard ceiling on ANY
bounds policy, per-element or not.

Errors are put on the REAL scale with the calibration anchored to the two WIDE
anchors (which the model fits: +0.0001 and -0.0030), then sanity-checked against
the new tight anchor.
"""
import json,os,sys
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; W=0.6784
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
SCM=(Y[...,:C]!=0.0); EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
m0,m1=SCM[...,0],SCM[...,1]; SUB=40
eu=EL[...,0][m0][::SUB]; ev=EL[...,1][m1][::SUB]
au=AP[...,0][m0][::SUB]; av=AP[...,1][m1][::SUB]; NT=eu.size+ev.size
best=None
for al in np.linspace(1.0,6.0,51):
    for be in np.linspace(-12,2,57):
        su,sv=eu*(al+be*au),ev*(al+be*av)
        Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
        hu,hv=0.05*au,0.05*av
        Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
        r=abs(Ec-0.4399)+abs(Ep-0.2076)
        if best is None or r<best[0]: best=(r,al,be)
_,AL,BE=best
su,sv=eu*(AL+BE*au),ev*(AL+BE*av)
chk=(np.exp(-2*0.0129/SIG)*(su<=0.0129).sum()+np.exp(-2*0.0098/SIG)*(sv<=0.0098).sum())/NT
print("calibration alpha %.2f beta %.2f ; tight-anchor check E %.4f (real 0.4876, bias %+.4f)"%(AL,BE,chk,chk-0.4876))
allerr=np.concatenate([su,sv])
Eor=float(np.mean(np.exp(-2*allerr/SIG)))
print("\n=== CEILING for OUR current error distribution ===")
print("  ORACLE E (perfect per-element bounds) = %.4f  -> sps %.2f  -> final %.2f"%(
    Eor,100*W*Eor,0.306*94.168150+0.163*74.025866+0.218*92.836278+0.100*91.362375+0.217*100*W*Eor))
print("  our achieved E                        = 0.4876 (%.0f%% of oracle)"%(100*0.4876/Eor))
print("  top-20 mean E                         = 0.5896 (%.0f%% of OUR oracle)"%(100*0.5896/Eor))
print("  doomduke2 E                           = 0.6073 (%.0f%% of OUR oracle)"%(100*0.6073/Eor))
print("\n  the tight-anchor bias above means the true oracle is LOWER than printed.")
print("\n=== error percentiles on the real scale ===")
for nm,a in (("u",su),("v",sv)):
    q=np.percentile(a,[25,50,75,90,95])
    print("  %s: p25 %.5f  p50 %.5f  p75 %.5f  p90 %.5f  p95 %.5f"%(nm,*q))
print("\n=== how much would a BETTER MODEL buy? (scale all errors by k) ===")
print("  %-6s %8s %8s %10s"%("k","E_best","sps","final"))
fin=lambda sps,rl,tk,mv:0.306*rl+0.163*tk+0.218*mv+0.100*91.362375+0.217*sps
grid=np.linspace(0.002,0.05,60)
for k in [1.0,0.9,0.8,0.7,0.6,0.5]:
    a2u,a2v=su*k,sv*k
    bu=max(grid,key=lambda h:float(np.mean(np.exp(-2*h/SIG)*(a2u<=h))))
    bv=max(grid,key=lambda h:float(np.mean(np.exp(-2*h/SIG)*(a2v<=h))))
    E=float((np.exp(-2*bu/SIG)*(a2u<=bu)).sum()+(np.exp(-2*bv/SIG)*(a2v<=bv)).sum())/NT
    print("  %-6.2f %8.4f %8.2f %10s"%(k,E,100*W*E,"(needs new rel_l2 too)"))
