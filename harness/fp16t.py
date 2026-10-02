"""ROUTE H: does fp16 inference speed the FNO up enough to matter, and does it cost
accuracy? time_score = 100/(1+sqrt(t/0.72896)); we are at 91.36 (t=6.59ms on A800).
No contamination risk -- this is purely an inference-speed change."""
import os,sys,time,math,zipfile,shutil
import importlib.util as iu
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
dev="cuda"
m,_=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=dev); m=m.to(dev).eval()
mi,si,mt,st=[torch.tensor(a).to(dev) for a in (MI,SI,MT,ST)]
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
Y=d["Y"].astype(np.float32)[:256]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
import json
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
wins=[]
for i in range(0,len(lens),5):
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
wins=wins[:256]
Wl=np.stack([np.concatenate([np.asarray(X[s:s+40]),np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xi,Yt=Wl[:,:20],Wl[:,20:]
def run(half,bs):
    # autocast keeps the COMPLEX spectral weights intact; a naive .half() does not
    o=[]
    torch.cuda.synchronize(); t0=time.time()
    with torch.no_grad():
        for i in range(0,len(Xi),bs):
            xb=torch.from_numpy(np.ascontiguousarray(Xi[i:i+bs])).to(dev)
            with torch.autocast("cuda",dtype=torch.float16,enabled=half):
                y=m((xb-mi)/si)*st+mt
            o.append(y.float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(o,0).astype(np.float32),(time.time()-t0)/len(Xi)
def ts(t): return 100.0/(1.0+math.sqrt(t/0.72896))
P32,t32=run(False,16); P32b,t32b=run(False,64)
P16,t16=run(True,64)
base=min(t32,t32b)
print("fp32 bs16 %.2f ms | fp32 bs64 %.2f ms | fp16 bs64 %.2f ms"%(t32*1e3,t32b*1e3,t16*1e3))
print("speedups vs fp32/bs16: bs64 %.2fx | fp16+bs64 %.2fx"%(t32/t32b,t32/t16))
r=S.rel_l2_per_sample(P16,Yt,2).mean(); r32=S.rel_l2_per_sample(P32,Yt,2).mean()
print("rel_l2 local: fp32 %.4f  fp16 %.4f  (delta %+.5f)"%(S.score_error(float(r32)),S.score_error(float(r)),
      S.score_error(float(r))-S.score_error(float(r32))))
print("max |fp16-fp32| %.2e"%np.abs(P16-P32).max())
A800=6.59e-3   # our measured real per-sample time for fp32/bs16
for nm,sp_ in (("bs64",t32/t32b),("fp16+bs64",t32/t16)):
    t=A800/sp_
    print("  %-10s -> A800 %.2f ms -> time_score %.2f (%+.2f) -> final %+.3f"%(
        nm,t*1e3,ts(t),ts(t)-91.362375,0.100*(ts(t)-91.362375)))
