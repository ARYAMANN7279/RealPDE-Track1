import os,sys,glob,importlib.util
import numpy as np, torch
A="/SML_DISK_24TB/rajeshr/Aryamann"; D=f"{A}/UGP"; KIT=f"{A}/starting_kit"
SD=f"{D}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
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
    r=Dataset.from_file(sp)[0]
    u=dec(r["u"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64]
    v=dec(r["v"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64]
    trajs[r["sim_id"]]=(u,v)
model,_=load_baseline(f"{D}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV); model.eval()
@torch.no_grad()
def fno(X):
    o=[]
    for i in range(0,X.shape[0],32):
        xb=torch.from_numpy(X[i:i+32]).float().to(DEV); o.append((model((xb-MI)/SI)*ST+MT).cpu().numpy())
    P=np.concatenate(o,0).astype(np.float32); P[...,2]=0.0; return P
C=2
def sc(P,Y): return (scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(P,Y,C)))),
                     scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(P,Y,C)))),
                     scoring.score_error(scoring.mvpe_rel_l2(P,Y)))
print(f"{'stride':>7s} {'method':28s}{'rel_l2':>8s}{'tke':>8s}{'mvpe':>8s}{'est.final':>10s}")
def fin(r,t,m,tm,s): return 0.30605*r+0.16274*t+0.21812*m+0.10011*tm+0.21704*s+0.00961
for s in (1,4,6,10,15):
    xs,ys=[],[]; span=40*s
    for sid,(u,v) in trajs.items():
        T=u.shape[0]
        for t0 in range(0,T-span,span):
            idx=np.arange(t0,t0+span,s)[:40]
            tr=np.stack([u[idx],v[idx],np.zeros_like(u[idx])],-1)
            xs.append(tr[:20]); ys.append(tr[20:])
    X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32)
    P=fno(X); per=np.repeat(X[:,-1:],20,axis=1).astype(np.float32); per[...,2]=0.0
    hyb=(per+(P-P.mean(axis=1,keepdims=True))).astype(np.float32); hyb[...,2]=0.0
    for nm,PR,tm in [("FNO",P,91.3),("persistence",per,99.0),("persist+FNO fluctuation",hyb,91.3)]:
        r,t,m=sc(PR,Y); print(f"{s:7d} {nm:28s}{r:8.2f}{t:8.2f}{m:8.2f}{fin(r,t,m,tm,29.84):10.2f}")
    print()
