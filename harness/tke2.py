"""TKE on the CORRECT data. Leaders gain +4 to +5.8 tke at equal-or-better rel_l2.

tke = rel-L2 between the predicted and target TEMPORAL-VARIANCE field.
Diagnose whether the FNO's error there is magnitude (fixable by scaling the
temporal fluctuation) or pattern (not fixable that way), then sweep fixes.
Honest split: fit on TRAIN trajectories, report on TEST.
"""
import numpy as np, json, sys, os
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
C=2
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows])
sims=sorted(set(sim.tolist())); tr=np.isin(sim,sims[0::2]); te=np.isin(sim,sims[1::2])
def ke(a):
    u,v=a[...,0],a[...,1]
    return 0.5*(np.mean((u-u.mean(1,keepdims=True))**2,1)+np.mean((v-v.mean(1,keepdims=True))**2,1))
kp,kt=ke(P),ke(Y)
print("=== TKE diagnosis on CORRECT data ===")
print("  pred KE mean=%.4e   target KE mean=%.4e   ratio=%.3f"%(kp.mean(),kt.mean(),kp.mean()/kt.mean()))
print("  => FNO %s temporal fluctuation"%("OVER-predicts" if kp.mean()>kt.mean() else "UNDER-predicts (too smooth)"))
# magnitude vs pattern: best achievable by a single global KE rescale
num=np.sum(kp*kt); den=np.sum(kp*kp)
a_opt=num/den
def tke_score(Kp):
    p=Kp.reshape(len(Kp),-1); t=kt.reshape(len(kt),-1)
    r=np.linalg.norm(p-t,axis=1)/np.linalg.norm(t,axis=1).clip(1e-8)
    return S.score_error(float(r.mean()))
print("\n  tke now                      : %.2f"%tke_score(kp))
print("  tke after OPTIMAL global KE scale (a=%.3f): %.2f"%(a_opt,tke_score(a_opt*kp)))
print("  tke with PERFECT magnitude per window     : %.2f"%tke_score(
    kp*(np.sum(kp.reshape(len(kp),-1)*kt.reshape(len(kt),-1),1)/
        np.sum(kp.reshape(len(kp),-1)**2,1).clip(1e-20))[:,None,None]))
print("  => if these are all close, the error is PATTERN, not magnitude")

print("\n=== alpha sweep on TEST (pred = mean_t + alpha*(pred-mean_t)) ===")
print("  %-7s %8s %8s %8s %10s"%("alpha","rel_l2","tke","mvpe","net final"))
base=None
for al in [0.9,1.0,1.1,1.2,1.35,1.5,1.75,2.0]:
    Q=P.copy(); mu=Q[...,:C].mean(axis=1,keepdims=True)
    Q[...,:C]=mu+al*(Q[...,:C]-mu)
    dm=S.rel_l2_per_sample(Q[te],Y[te],C); tk=S.tke_rel_l2_per_sample(Q[te],Y[te],C); mv=S.mvpe_rel_l2_per_sample(Q[te],Y[te])
    r=(S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
    if al==1.0: base=r
    net="" if base is None else "%+.3f"%(0.306*(r[0]-base[0])+0.163*(r[1]-base[1])+0.218*(r[2]-base[2]))
    print("  %-7.2f %8.2f %8.2f %8.2f %10s"%(al,r[0],r[1],r[2],net))

print("\n=== per-pixel temporal-variance recalibration (fit on TRAIN, apply to TEST) ===")
# scale each pixel's temporal fluctuation so predicted KE matches target KE on TRAIN
num=(kp[tr]*kt[tr]).sum(0); den=(kp[tr]*kp[tr]).sum(0).clip(1e-20)
gain=np.sqrt(np.clip(num/den,0.25,4.0))            # amplitude gain per pixel
print("  per-pixel gain: median=%.3f  p10=%.3f p90=%.3f"%(np.median(gain),*np.percentile(gain,[10,90])))
Q=P.copy(); mu=Q[...,:C].mean(axis=1,keepdims=True)
Q[...,:C]=mu+gain[None,None,:,:,None]*(Q[...,:C]-mu)
dm=S.rel_l2_per_sample(Q[te],Y[te],C); tk=S.tke_rel_l2_per_sample(Q[te],Y[te],C); mv=S.mvpe_rel_l2_per_sample(Q[te],Y[te])
r=(S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
print("  rel_l2 %.2f  tke %.2f  mvpe %.2f   net %+.3f final"%(
    r[0],r[1],r[2],0.306*(r[0]-base[0])+0.163*(r[1]-base[1])+0.218*(r[2]-base[2])))
np.save(f"{B}/local_harness/tke_gain.npy",gain)
