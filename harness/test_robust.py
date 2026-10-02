"""Full test of the ROBUST zip: GPU timing, forced-CPU timing, and numerical
equivalence of its predictions against the proven submission."""
import os,sys,time,zipfile,shutil,importlib.util
FORCE_CPU = os.environ.get("FORCE_CPU")=="1"
if FORCE_CPU: os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
def load(zipname,tag):
    WD=f"{B}/_tmp/{tag}"; shutil.rmtree(WD,ignore_errors=True); os.makedirs(WD)
    with zipfile.ZipFile(f"{B}/submissions/{zipname}") as z: z.extractall(WD)
    sys.path.insert(0,WD)
    sp=importlib.util.spec_from_file_location("m_"+tag,os.path.join(WD,"submission.py"))
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
print("device will be:", "cpu (forced)" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"),flush=True)
rob=load("submission_ROBUST.zip","rob")
rng=np.random.default_rng(0)
N=64 if not FORCE_CPU else 24
X=rng.normal(0.155,0.097,size=(N,20,32,64,3)).astype(np.float32); X[...,2]=0.0
t=time.time(); r=rob.predict(X,metadata={}); dt=time.time()-t
per=dt/N
print("throughput %.2f ms/sample   batch=%d"%(per*1000,rob._BATCH),flush=True)
print("shapes ok:",r["prediction"].shape==X.shape and r["lower"].shape==X.shape)
print("finite:",np.isfinite(r["prediction"]).all(),np.isfinite(r["lower"]).all(),np.isfinite(r["upper"]).all())
print("lower<=upper:",bool((r["lower"]<=r["upper"]).all()))
print("half-widths u/v: %.4f %.4f"%((r["upper"]-r["lower"])[0,0,0,0,0]/2,(r["upper"]-r["lower"])[0,0,0,0,1]/2))
print("p channel zero:",bool((r["prediction"][...,2]==0).all()))
if not FORCE_CPU:
    prov=load("submission_fno_plain_sps.zip","prov")
    r2=prov.predict(X,metadata={})
    d=np.abs(r["prediction"]-r2["prediction"]).max()
    print("\nmax |ROBUST - proven| prediction: %.3e  (identical model, expect 0)"%d)
print()
for Nw in (1000,2000,5140,10000):
    tot=3.0+per*Nw
    print("  N=%5d -> %7.1fs  %s"%(Nw,tot,"OK" if tot<180 else "would hit budget -> persistence fallback engages"))
