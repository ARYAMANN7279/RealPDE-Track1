"""Structural per-element bounds: exploit that error grows with rollout step and
concentrates in the wake. These need no regressor -- the bound depends only on
(timestep, pixel, channel), all known at inference.

For each cell, the optimal h is found exactly: sorting that cell's TRAIN errors,
h = e_(k) gives coverage k/n and objective exp(-2 e_(k)/sigma) * k/n; maximise
over k. Fitted on TRAIN trajectories, scored on TEST.
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
ERR=(np.abs(P[...,:C]-Y[...,:C])/LAM).astype(np.float32)
SCM=(Y[...,:C]!=0.0)
sims=sorted(set(sim.tolist())); tr=np.isin(sim,sims[0::2]); te=np.isin(sim,sims[1::2])
del P,Y,d
print("train %d / test %d windows ; lambda %.3f"%(tr.sum(),te.sum(),LAM),flush=True)

def best_h(vals):
    """exact optimiser of exp(-2h/sig)*F(h) over an empirical sample."""
    v=np.sort(vals)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    obj=np.exp(-2*v/SIG)*k
    return float(v[np.argmax(obj)])

def score(H,m):
    """H: array broadcastable to (T,Hh,W,C) of half-widths."""
    e=ERR[m]; s=SCM[m]
    h=np.broadcast_to(H,e.shape)
    ok=(e<=h)&s
    val=np.exp(-2*h/SIG)*ok
    ns=s.sum()
    return float(val.sum()/ns), float(ok.sum()/ns)

T,Hh,W=ERR.shape[1],ERR.shape[2],ERR.shape[3]
res={}
# 0. baseline + best constant
Hc=np.zeros((1,1,1,C),np.float32); Hc[...,0]=0.030; Hc[...,1]=0.010
res["baseline [0.030,0.010]"]=score(Hc,te)
Hk=np.zeros((1,1,1,C),np.float32)
for ci in range(C):
    Hk[...,ci]=best_h(ERR[tr][...,ci][SCM[tr][...,ci]][::37])
res["best constant"]=score(Hk,te); print("  best constant = [%.4f,%.4f]"%(Hk[0,0,0,0],Hk[0,0,0,1]),flush=True)

# 1. per-timestep
Ht=np.zeros((T,1,1,C),np.float32)
for t in range(T):
    for ci in range(C):
        v=ERR[tr][:,t,:,:,ci][SCM[tr][:,t,:,:,ci]]
        Ht[t,0,0,ci]=best_h(v[::7])
res["per-timestep"]=score(Ht,te)
print("  per-timestep h_u: t0=%.4f t9=%.4f t19=%.4f (error growth)"%(Ht[0,0,0,0],Ht[9,0,0,0],Ht[19,0,0,0]),flush=True)

# 2. per-location (static map)
Hl=np.zeros((1,Hh,W,C),np.float32)
for ci in range(C):
    E=ERR[tr][...,ci]; M=SCM[tr][...,ci]
    for i in range(Hh):
        for j in range(W):
            v=E[:,:,i,j][M[:,:,i,j]]
            Hl[0,i,j,ci]=best_h(v[::3])
res["per-location"]=score(Hl,te)

# 3. per-timestep x per-location
Htl=np.zeros((T,Hh,W,C),np.float32)
for ci in range(C):
    E=ERR[tr][...,ci]; M=SCM[tr][...,ci]
    for t in range(T):
        Et=E[:,t]; Mt=M[:,t]
        for i in range(Hh):
            for j in range(W):
                v=Et[:,i,j][Mt[:,i,j]]
                Htl[t,i,j,ci]=best_h(v)
res["per-timestep x location"]=score(Htl,te)

print("\n=== TEST trajectories (calibrated to real scale) ===")
print("  %-28s %8s %8s %10s %10s"%("policy","E","coverage","real sps","d final"))
for k,(E,cv) in res.items():
    sps=100*WREAL*E
    print("  %-28s %8.4f %8.3f %10.2f %+10.2f"%(k,E,cv,sps,(sps-29.84)*0.217))
print("\n  real anchors: current bounds 29.84 | agent33 (identical model) 35.33")
np.savez(f"{B}/local_harness/struct_bounds.npz",Hk=Hk,Ht=Ht,Hl=Hl,Htl=Htl)
print("  saved bound maps -> struct_bounds.npz")
