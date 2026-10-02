"""If the eval container gives no GPU, submission.py silently runs on CPU.
Measure that path against the 3-minute budget."""
import time, os, sys, zipfile, shutil, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"]=""          # force the CPU fallback
import numpy as np, torch
print("torch sees cuda:",torch.cuda.is_available())
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
WD=f"{B}/_tmp/benchcpu"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
with zipfile.ZipFile(f"{B}/submissions/submission_MINIMAL_const.zip") as z: z.extractall(WD)
sys.path.insert(0,WD)
sp=importlib.util.spec_from_file_location("subm",os.path.join(WD,"submission.py"))
subm=importlib.util.module_from_spec(sp); sp.loader.exec_module(subm)
rng=np.random.default_rng(0)
X=rng.normal(0.155,0.097,size=(32,20,32,64,3)).astype(np.float32); X[...,2]=0.0
t=time.time(); _=subm.predict(X[:4],metadata={}); LOAD=time.time()-t
print("model load + first call: %.1fs"%LOAD,flush=True)
t=time.time(); _=subm.predict(X,metadata={}); per=(time.time()-t)/32
print("CPU throughput: %.1f ms/sample"%(per*1000),flush=True)
print()
for N in (500,1000,2000,3000,5140):
    tot=LOAD+per*N
    print("  N=%5d -> %7.1fs  %s"%(N,tot,"OK" if tot<180 else "*** OVER 3 MIN -> Failed, no score ***"))
print("\nmax N on CPU within 180s: %d"%max(0,int((180-LOAD)/per)))
