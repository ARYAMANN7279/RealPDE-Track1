import os,sys,zipfile,shutil,importlib.util,time,json
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S); C=2
WD=f"{B}/_tmp/fs"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/submission_fno_plain_sps.zip") as z: z.extractall(WD)
sys.path.insert(0,WD)
s2=iu.spec_from_file_location("fs",os.path.join(WD,"submission.py"))
m=iu.module_from_spec(s2); s2.loader.exec_module(m)
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
wins=wins[:200]
W_=np.stack([np.concatenate([np.asarray(X[s:s+40]),np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=W_[:,:20],W_[:,20:]
t=time.time(); r=m.predict(Xin,metadata={}); dt=(time.time()-t)/len(Xin)
p,lo,up=r["prediction"],r["lower"],r["upper"]
print("throughput %.2f ms/sample -> N=5140 would take %.0fs (budget 180s incl load)"%(dt*1000,dt*5140+15))
print("finite %s | lo<=up %s | shapes %s"%(bool(np.isfinite(p).all()),bool((lo<=up).all()),
      p.shape==Xin.shape and lo.shape==p.shape))
hw=(up-lo)/2
print("bounds vary per element: %s | h_u med %.5f range %.5f-%.5f"%(
   bool(hw[...,0].std()>1e-5),float(np.median(hw[...,0])),float(hw[...,0].min()),float(hw[...,0].max())))
dm=S.rel_l2_per_sample(p,Y,C); tk=S.tke_rel_l2_per_sample(p,Y,C); mv=S.mvpe_rel_l2_per_sample(p,Y)
print("local-scale: rel_l2 %.2f tke %.2f mvpe %.2f"%(S.score_error(float(dm.mean())),
      S.score_error(float(tk.mean())),S.score_error(float(mv.mean()))))
m._TIME_BUDGET=-1.0; r2=m.predict(Xin[:16],metadata={})
print("forced fallback: finite %s lo<=up %s"%(bool(np.isfinite(r2["prediction"]).all()),
      bool((r2["lower"]<=r2["upper"]).all())))
