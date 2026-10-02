"""How much headroom is left in bounds at all?

ORACLE: if we knew |err| per element, set h = |err| exactly -> contribution
exp(-2|err|/sigma), coverage 1. E_oracle = mean(exp(-2|err|/sigma)) is a hard
ceiling on ANY per-element policy. Compare with what we achieve.
Also tests a hybrid: per-location map scaled by a local-gradient decile.
"""
import numpy as np, json, sys, os
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; WREAL=0.6784
LAM=float(np.load(f"{B}/local_harness/calib_lambda.npy")[0])
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows])
ERR=(np.abs(P[...,:C]-Y[...,:C])/LAM).astype(np.float32); SCM=(Y[...,:C]!=0.0)
sims=sorted(set(sim.tolist())); tr=np.isin(sim,sims[0::2]); te=np.isin(sim,sims[1::2])
M=np.load(f"{B}/local_harness/struct_bounds.npz")
def score(H,m):
    e=ERR[m]; s=SCM[m]; h=np.broadcast_to(H,e.shape)
    ok=(e<=h)&s; ns=s.sum()
    return float((np.exp(-2*h/SIG)*ok).sum()/ns), float(ok.sum()/ns)

e_te=ERR[te][SCM[te]]
E_or=float(np.mean(np.exp(-2*e_te/SIG)))
print("=== CEILING ===")
print("  ORACLE per-element  E=%.4f -> real sps %.2f  (%+.2f final)"%(E_or,100*WREAL*E_or,(100*WREAL*E_or-29.84)*0.217))
for nm,key in [("best constant","Hk"),("per-location","Hl")]:
    E,cv=score(M[key],te)
    print("  %-18s E=%.4f -> real sps %.2f   (%.0f%% of oracle)"%(nm,E,100*WREAL*E,100*E/E_or))

# hybrid: per-location map modulated by local gradient decile (inference-available)
u,v=P[...,0],P[...,1]
g=np.sqrt(np.gradient(u,axis=3)**2+np.gradient(u,axis=2)**2
         +np.gradient(v,axis=3)**2+np.gradient(v,axis=2)**2)
q=np.quantile(g[tr][::11],np.linspace(0,1,11)[1:-1])
bin_id=np.digitize(g,q)                      # 0..9 per element
print("\n=== hybrid: per-location map x gradient-decile multiplier ===")
Hl=np.broadcast_to(M["Hl"],ERR.shape).copy()
mult=np.ones((10,C),np.float32)
for ci in range(C):
    for b in range(10):
        sel=(bin_id[tr]==b)&SCM[tr][...,ci]
        if sel.sum()<1000: continue
        base=Hl[tr][...,ci][sel]; e=ERR[tr][...,ci][sel]
        best=(1.0,-1)
        for m_ in np.linspace(0.3,3.0,28):
            h=base*m_
            val=float((np.exp(-2*h/SIG)*(e<=h)).mean())
            if val>best[1]: best=(m_,val)
        mult[b,ci]=best[0]
print("  gradient-decile multipliers (u):",np.round(mult[:,0],2))
Hh=Hl.copy()
for ci in range(C):
    Hh[...,ci]=Hl[...,ci]*mult[bin_id,ci]
E,cv=score(Hh[te],te)
print("  hybrid              E=%.4f cov=%.3f -> real sps %.2f  (%+.2f final)  (%.0f%% of oracle)"%(
    E,cv,100*WREAL*E,(100*WREAL*E-29.84)*0.217,100*E/E_or))
print("\n  real anchors: current 29.84 | agent33 (identical model) 35.33")
np.save(f"{B}/local_harness/grad_mult.npy",mult)
