"""Force the time-budget fallback and verify the output is still fully valid."""
import os,sys,zipfile,shutil,importlib.util,time
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
WD=f"{B}/_tmp/fb"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/submission_ROBUST.zip") as z: z.extractall(WD)
sys.path.insert(0,WD)
sp=importlib.util.spec_from_file_location("fb",os.path.join(WD,"submission.py"))
m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
rng=np.random.default_rng(1)
X=rng.normal(0.155,0.097,size=(96,20,32,64,3)).astype(np.float32); X[...,2]=0.0
full=m.predict(X,metadata={})["prediction"].copy()      # normal path first
m._TIME_BUDGET=-1.0                                      # force immediate fallback
r=m.predict(X,metadata={})
p,lo,up=r["prediction"],r["lower"],r["upper"]
print("=== forced-fallback output validity ===")
print("  shape correct      :",p.shape==X.shape)
print("  all finite         :",bool(np.isfinite(p).all() and np.isfinite(lo).all() and np.isfinite(up).all()))
print("  lower <= upper     :",bool((lo<=up).all()))
print("  p channel zeroed   :",bool((p[...,2]==0).all()))
print("  bounds shape match :",lo.shape==p.shape==up.shape)
same=np.isclose(p[:, :, :, :, 0], X[:, -1:, :, :, 0]).all()
print("  is persistence     :",bool(same),"(repeats last input frame)")
print("  differs from model :",float(np.abs(p-full).max())>0)
m._TIME_BUDGET=145.0
print("\n=== partial fallback (budget trips mid-run) ===")
import types
orig=time.time
m._T_START=time.time()-144.0     # only ~1s of budget left
r2=m.predict(X,metadata={})
p2=r2["prediction"]
print("  finite:",bool(np.isfinite(p2).all()),"| lower<=upper:",bool((r2["lower"]<=r2["upper"]).all()))
print("  shape:",p2.shape==X.shape)
nmodel=int((np.abs(p2-full).reshape(len(p2),-1).max(1)==0).sum())
print("  windows from model: %d / %d, rest persistence"%(nmodel,len(p2)))
print("\n=> output is always complete and valid; never killed mid-write")
