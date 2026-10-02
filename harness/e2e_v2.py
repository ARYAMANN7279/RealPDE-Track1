"""End-to-end test of the built zip: extract it, import its submission.py,
call predict(), and score the returned prediction AND bounds exactly as the
leaderboard scorer does. Verifies fp16 packing did not cost anything and that
the dict interface (prediction/lower/upper) round-trips.
"""
import json, os, shutil, sys, zipfile
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
ZIP=f"{B}/submissions/submission_soup_v2.zip"
EX=f"{B}/local_harness/_e2e"; shutil.rmtree(EX,ignore_errors=True); os.makedirs(EX)
zipfile.ZipFile(ZIP).extractall(EX)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
C=2; IN=20
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
wins=np.array(wins)
W=[]
for s in wins:
    w=np.asarray(X[s:s+40]); z=np.zeros(w.shape[:-1]+(1,),np.float32)
    W.append(np.concatenate([w,z],-1))
W=np.stack(W).astype(np.float32)
Xin,Y=W[:,:IN],W[:,IN:]
print("test windows: %s"%(Xin.shape,),flush=True)
sys.path.insert(0,EX)
sub=importlib.util.spec_from_file_location("submission",os.path.join(EX,"submission.py"))
mod=importlib.util.module_from_spec(sub); sub.loader.exec_module(mod)
out=mod.predict(Xin, metadata={})
assert isinstance(out,dict) and {"prediction","lower","upper"}<=set(out), "bad return type"
P,L,U=out["prediction"],out["lower"],out["upper"]
print("shapes: pred %s lower %s upper %s"%(P.shape,L.shape,U.shape))
assert P.shape==Y.shape and L.shape==P.shape and U.shape==P.shape
assert np.all(np.isfinite(P)) and np.all(np.isfinite(L)) and np.all(np.isfinite(U)), "non-finite"
assert np.all(L<=U), "lower > upper somewhere"
dm=S.rel_l2_per_sample(P,Y,C); tk=S.tke_rel_l2_per_sample(P,Y,C); mv=S.mvpe_rel_l2_per_sample(P,Y)
sps,cov=S.aggregate_sps(P,Y,C,lower=L,upper=U)
print("\n=== end-to-end scores on held-out trajectories (LOCAL scale) ===")
print("  rel_l2 %.2f | tke %.2f | mvpe %.2f | sps %.2f | coverage %.3f"%(
    S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),
    S.score_error(float(mv.mean())),S.score_sps(sps),cov*100))
# compare against the unpacked fp32 model to confirm packing cost ~0
import torch
sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline
m,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device="cuda:1")
m.load_state_dict(torch.load(f"{B}/local_harness/soup_best.pth",map_location="cuda:1")); m.eval()
MI=torch.tensor([0.154960856,-0.000513992854,0.0]).cuda(1);SI=torch.tensor([0.0968056545,0.015960684,1.0]).cuda(1)
MT=torch.tensor([0.154962569,-0.000517793698,0.0]).cuda(1);ST=torch.tensor([0.0968104079,0.0159636438,1.0]).cuda(1)
o=[]
with torch.no_grad():
    for i in range(0,len(Xin),16):
        xb=torch.from_numpy(Xin[i:i+16]).cuda(1)
        o.append((m((xb-MI)/SI)*ST+MT).cpu().numpy())
P32=np.concatenate(o,0).astype(np.float32); P32[...,2]=0.0
print("\n  fp16-vs-fp32 max abs diff: %.3e   rel_l2(fp32) %.2f"%(
    np.abs(P32-P).max(),S.score_error(float(S.rel_l2_per_sample(P32,Y,C).mean()))))
print("\nALL STRUCTURAL CHECKS PASSED")
