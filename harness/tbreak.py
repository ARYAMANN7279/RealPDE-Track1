"""Where does the 13 ms/sample go, and does fp16 on the head buy headroom?
Eval hardware is 1x A800; our FNO measured 2.7 ms here vs 6.6 ms there = 2.44x."""
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
X=rng.normal(0.155,0.097,size=(96,20,32,64,3)).astype(np.float32); X[...,2]=0
base=load("submission_ROBUST.zip","tb0")       # FNO + constant bounds only
full=load("submission_FULLSTACK_v5.zip","tb1")
for m,nm in ((base,"FNO+const"),(full,"FULLSTACK_v4")):
    m.predict(X[:16],metadata={})
    t=time.time(); m.predict(X,metadata={}); d=(time.time()-t)/96
    print("  %-14s %6.2f ms/sample   -> A800 est %6.2f ms  -> N=5140 %5.0fs"%(nm,d*1000,d*1000*2.44,d*2.44*5140))
# fp16 head
z=np.load(f"{B}/local_harness/head_assets.npz")
h=full._get_head("cuda")
ft=torch.randn(16,int(z["nf"]),32,64,device="cuda")
with torch.no_grad():
    h(ft); torch.cuda.synchronize()
    t=time.time()
    for _ in range(20): h(ft)
    torch.cuda.synchronize(); t32=(time.time()-t)/20/16
    h16=h.half()
    ft16=ft.half(); h16(ft16); torch.cuda.synchronize()
    t=time.time()
    for _ in range(20): h16(ft16)
    torch.cuda.synchronize(); t16=(time.time()-t)/20/16
print("  head fp32 %.3f ms/window | fp16 %.3f ms/window | speedup %.1fx"%(t32*1000,t16*1000,t32/t16))
