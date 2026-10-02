"""Time the FULL container path: import + checkpoint load + inference.
Limit is 3 MINUTES INCLUDING LOADING (spec p.3 / FAQ line 308)."""
import time, os, sys, zipfile, shutil, importlib.util
T0=time.time()
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
WD="/SML_DISK_24TB/rajeshr/Aryamann/UGP/_tmp/bench"
shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/submission_MINIMAL_const.zip") as z: z.extractall(WD)
sys.path.insert(0,WD)
print("import+extract: %.1fs"%(time.time()-T0),flush=True)
t=time.time()
sp=importlib.util.spec_from_file_location("subm",os.path.join(WD,"submission.py"))
subm=importlib.util.module_from_spec(sp); sp.loader.exec_module(subm)
print("module import: %.1fs"%(time.time()-t),flush=True)
rng=np.random.default_rng(0)
X=rng.normal(0.155,0.097,size=(64,20,32,64,3)).astype(np.float32); X[...,2]=0.0
t=time.time(); _=subm.predict(X[:8],metadata={}); tl=time.time()-t
print("first call (includes model load to GPU): %.1fs"%tl,flush=True)
t=time.time(); _=subm.predict(X,metadata={}); tw=time.time()-t
per=tw/64
print("warm throughput: %.2f ms/sample  (batch=%d)"%(per*1000,subm._BATCH),flush=True)
print()
LOAD=tl  # model load dominates the first call
for N in (500,1000,2000,3000,5000):
    tot=LOAD+per*N
    print("  N=%5d -> %6.1fs total  %s"%(N,tot,"OK" if tot<180 else "*** OVER 3 MIN ***"))
print("\nmax N within 180s: %d"%int((180-LOAD)/per))
