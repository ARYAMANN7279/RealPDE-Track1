"""Is the head path kernel-launch-bound? Vary chunk size; if bigger chunks
collapse the overhead, the time_score penalty largely disappears.
Also: is the numpy FFT spectral correction (CPU) a hidden cost?"""
import os,sys,time,zipfile,shutil
import importlib.util as iu
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
def load(z,t):
    WD=f"{B}/_tmp/{t}"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
    with zipfile.ZipFile(f"{B}/submissions/{z}") as f: f.extractall(WD)
    sys.path.insert(0,WD)
    sp=iu.spec_from_file_location("m"+t,os.path.join(WD,"submission.py"))
    m=iu.module_from_spec(sp); sp.loader.exec_module(m); return m
rng=np.random.default_rng(0)
X=rng.normal(0.155,0.097,size=(128,20,32,64,3)).astype(np.float32); X[...,2]=0
base=load("submission_ROBUST.zip","ck0")
base.predict(X[:16],metadata={})
ts=[]
for _ in range(3):
    t=time.time(); base.predict(X,metadata={}); ts.append((time.time()-t)/128)
b=min(ts); print("FNO+const baseline: %.2f ms/sample (best of 3)"%(b*1000))
full=load("submission_FULLSTACK_v7.zip","ck1")
full.predict(X[:16],metadata={})
import re
for CH in (16,32,64,128):
    src=open(f"{B}/_tmp/ck1/submission.py").read()
    src2=re.sub(r"            CH = \d+","            CH = %d"%CH,src)
    open(f"{B}/_tmp/ck1/submission.py","w").write(src2)
    sp=iu.spec_from_file_location("mm%d"%CH,f"{B}/_tmp/ck1/submission.py")
    m=iu.module_from_spec(sp); sp.loader.exec_module(m)
    m.predict(X[:16],metadata={})
    ts=[]
    for _ in range(3):
        t=time.time(); m.predict(X,metadata={}); ts.append((time.time()-t)/128)
    f=min(ts)
    print("  CH=%3d: %.2f ms/sample  ratio vs FNO %.2fx"%(CH,f*1000,f/b))
