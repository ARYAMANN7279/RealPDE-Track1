import os,sys,zipfile,shutil,importlib.util,time
import numpy as np
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
def load(z,t):
    WD=f"{B}/_tmp/{t}"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
    with zipfile.ZipFile(f"{B}/submissions/{z}") as f: f.extractall(WD)
    sys.path.insert(0,WD)
    sp=importlib.util.spec_from_file_location("m"+t,os.path.join(WD,"submission.py"))
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
rng=np.random.default_rng(0)
X=rng.normal(0.155,0.097,size=(48,20,32,64,3)).astype(np.float32); X[...,2]=0.0
rob=load("submission_ROBUST.zip","r0")
base=rob.predict(X,metadata={})["prediction"]
for z,t in (("PERLOC_V3.zip","p3"),("SOUP_V3.zip","s3")):
    m=load(z,t); t0=time.time(); r=m.predict(X,metadata={}); dt=(time.time()-t0)/48
    p,lo,up=r["prediction"],r["lower"],r["upper"]
    hw=(up-lo)/2
    print("%-14s %.2f ms/sample | finite %s | lo<=up %s | shape %s"%(
        z,dt*1000,bool(np.isfinite(p).all()),bool((lo<=up).all()),p.shape==X.shape))
    print("   half-width u med %.5f  v med %.5f   varies-by-location: %s"%(
        float(np.median(hw[...,0])),float(np.median(hw[...,1])),
        bool(hw[0,0,:,:,0].std()>0)))
    print("   model same as ROBUST: %s   (soup should differ)"%bool(np.array_equal(p,base)))
    print("   has time budget: %s"%hasattr(m,"_TIME_BUDGET"))
    m._TIME_BUDGET=-1.0; r2=m.predict(X,metadata={})
    print("   forced fallback valid: finite %s, lo<=up %s"%(
        bool(np.isfinite(r2["prediction"]).all()),bool((r2["lower"]<=r2["upper"]).all())))
