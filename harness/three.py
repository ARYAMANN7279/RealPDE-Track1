"""Recalibrate on THREE real anchors and find the true optimal constant bounds.

Real anchors (original checkpoint, W = 0.6784):
    h = 0.05|pred|          -> E 0.2076   (sps 14.08)
    h = [0.030 , 0.010]     -> E 0.4399   (sps 29.84)
    h = [0.0129, 0.0098]    -> E 0.4876   (sps 33.08)   <-- NEW, today
Fit err_real = err_local*(alpha+beta*|pred|) to all three, then optimise h.
Three constraints, two parameters => the fit is falsifiable.
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
m0,m1=SCM[...,0],SCM[...,1]
SUB=60
eu=EL[...,0][m0][::SUB]; ev=EL[...,1][m1][::SUB]
au=AP[...,0][m0][::SUB]; av=AP[...,1][m1][::SUB]
NT=eu.size+ev.size
print("calibration sample: %d elements"%NT,flush=True)
def E_of(su,sv,hu,hv):
    if np.isscalar(hu):
        return float((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
    return float((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
TARG=[("prop 0.05|p|",None,0.2076),("const [.030,.010]",(0.030,0.010),0.4399),
      ("const [.0129,.0098]",(0.0129,0.0098),0.4876)]
best=None
for al in np.linspace(1.0,6.0,51):
    for be in np.linspace(-12,2,57):
        su,sv=eu*(al+be*au),ev*(al+be*av)
        r=0.0; vals=[]
        for tag,h,t in TARG:
            if h is None: E=E_of(su,sv,0.05*au,0.05*av)
            else:         E=E_of(su,sv,h[0],h[1])
            vals.append(E); r+=abs(E-t)
        if best is None or r<best[0]: best=(r,al,be,vals)
r,AL,BE,vals=best
print("\n=== 3-anchor fit: alpha %.3f  beta %.3f  total resid %.4f ==="%(AL,BE,r))
for (tag,_,t),v in zip(TARG,vals):
    print("   %-22s predicted %.4f   real %.4f   err %+.4f"%(tag,v,t,v-t))
su,sv=eu*(AL+BE*au),ev*(AL+BE*av)
print("\n=== optimal CONSTANT bounds under the 3-anchor model ===")
gu=np.linspace(0.003,0.030,55); gv=np.linspace(0.003,0.020,35)
bu=max(gu,key=lambda h:float((np.exp(-2*h/SIG)*(su<=h)).mean()))
bv=max(gv,key=lambda h:float((np.exp(-2*h/SIG)*(sv<=h)).mean()))
Eb=E_of(su,sv,bu,bv)
fin=lambda sps:0.306*94.168150+0.163*74.025866+0.218*92.836278+0.100*91.362375+0.217*sps
CUR=fin(33.078742)
print("   optimum h = [%.4f, %.4f]  ->  E %.4f  sps %.2f"%(bu,bv,Eb,100*W*Eb))
print("   current   h = [0.0129, 0.0098] ->  E 0.4876  sps 33.08  (REAL, banked 78.07)")
print("   projected gain: %+.2f final  -> %.2f"%(fin(100*W*Eb)-CUR,78.068714+fin(100*W*Eb)-CUR))
print("\n=== sensitivity around the optimum ===")
print("   %-22s %8s %8s %10s"%("h_u , h_v","E","sps","d final"))
for hu in [0.006,0.008,0.0100,0.0115,0.0129,0.015]:
    for hv in [bv]:
        E=E_of(su,sv,hu,hv); sps=100*W*E
        print("   [%.4f, %.4f]     %8.4f %8.2f %+10.2f"%(hu,hv,E,sps,fin(sps)-CUR))
print("\n   agent33 REAL: sps 35.33 (E %.4f) -> 78.64"%(35.33/(100*W)))
