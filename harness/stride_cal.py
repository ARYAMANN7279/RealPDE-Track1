"""Calibrate the proxy: find the temporal STRIDE at which the FNO's subscores on
held-out trajectories match its REAL leaderboard subscores (94.168/74.026/92.836).
Then the proxy can be trusted for persistence/hybrid/bounds decisions.
"""
import os,sys,glob,json,time
import numpy as np, torch
A="/SML_DISK_24TB/rajeshr/Aryamann"; D=f"{A}/UGP"; KIT=f"{A}/starting_kit"
SD=f"{D}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
from load_baseline import load_baseline
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
from datasets import Dataset
DEV="cuda"
MI=torch.tensor([0.154960856,-0.000513992854,0.0],device=DEV);SI=torch.tensor([0.0968056545,0.015960684,1.0],device=DEV)
MT=torch.tensor([0.154962569,-0.000517793698,0.0],device=DEV);ST=torch.tensor([0.0968104079,0.0159636438,1.0],device=DEV)
def dec(b,t,h,w):
    for dt in (np.float32,np.float16,np.float64):
        if len(b)==t*h*w*np.dtype(dt).itemsize: return np.frombuffer(b,dt).reshape(t,h,w).astype(np.float32)
trajs={}
for sp in sorted(glob.glob(f"{SD}/*.arrow")):
    r=Dataset.from_file(sp)[0]; sid=r["sim_id"]
    u=dec(r["u"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64]
    v=dec(r["v"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64]
    trajs[sid]=(u,v); print("loaded",sid,u.shape,flush=True)
model,_=load_baseline(f"{D}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV); model.eval()
@torch.no_grad()
def fno(X):
    out=[]
    for i in range(0,X.shape[0],32):
        xb=torch.from_numpy(X[i:i+32]).float().to(DEV)
        out.append((model((xb-MI)/SI)*ST+MT).cpu().numpy())
    P=np.concatenate(out,0).astype(np.float32); P[...,2]=0.0; return P
print(f"\n{'stride':>7s}{'nwin':>6s}{'rel_l2':>9s}{'tke':>8s}{'mvpe':>9s}   (REAL: 94.17 / 74.03 / 92.84)")
res={}
for s in (1,2,3,4,5,6,8,10):
    xs,ys=[],[]
    span=40*s
    for sid,(u,v) in trajs.items():
        T=u.shape[0]
        for t0 in range(0,T-span,span):
            idx=np.arange(t0,t0+span,s)[:40]
            uu,vv=u[idx],v[idx]
            tr=np.stack([uu,vv,np.zeros_like(uu)],-1)
            xs.append(tr[:20]); ys.append(tr[20:])
    X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32)
    P=fno(X); C=2
    r=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(P,Y,C))))
    t=scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(P,Y,C))))
    m=scoring.score_error(scoring.mvpe_rel_l2(P,Y))
    res[s]=(r,t,m,X,Y,P)
    print(f"{s:7d}{X.shape[0]:6d}{r:9.2f}{t:8.2f}{m:9.2f}",flush=True)
best=min(res,key=lambda s: abs(res[s][1]-74.026)+abs(res[s][0]-94.168))
print(f"\n=> BEST-MATCHING STRIDE = {best}  (rel_l2 {res[best][0]:.2f} tke {res[best][1]:.2f} mvpe {res[best][2]:.2f})")
X,Y,P=res[best][3],res[best][4],res[best][5]
np.savez(f"{D}/local_harness/calib_eval.npz",X=X,Y=Y,P=P,stride=best)
print("saved calib_eval.npz")
