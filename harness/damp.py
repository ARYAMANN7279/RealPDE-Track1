import os,sys,glob,importlib.util
import numpy as np, torch
A="/SML_DISK_24TB/rajeshr/Aryamann"; D=f"{A}/UGP"; KIT=f"{A}/starting_kit"; SD=f"{D}/data/real_hf/foil/hf_dataset/real"
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
    trajs[r["sim_id"]]=(dec(r["u"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64],
                        dec(r["v"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64])
model,_=load_baseline(f"{D}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV); model.eval()
@torch.no_grad()
def fno(X):
    o=[]
    for i in range(0,X.shape[0],32):
        xb=torch.from_numpy(X[i:i+32]).float().to(DEV); o.append((model((xb-MI)/SI)*ST+MT).cpu().numpy())
    P=np.concatenate(o,0).astype(np.float32); P[...,2]=0.0; return P
s=6; span=40*s; xs,ys=[],[]
for sid,(u,v) in trajs.items():
    T=u.shape[0]
    for t0 in range(0,T-span,span):
        idx=np.arange(t0,t0+span,s)[:40]
        tr=np.stack([u[idx],v[idx],np.zeros_like(u[idx])],-1); xs.append(tr[:20]); ys.append(tr[20:])
X=np.stack(xs).astype(np.float32); Y=np.stack(ys).astype(np.float32); P=fno(X); C=2
# does the FNO OVER-predict temporal fluctuation?
def tstd(a): return float(a[...,:2].std(axis=1).mean())
print(f"temporal std: FNO={tstd(P):.5f}  target={tstd(Y):.5f}  ratio={tstd(P)/tstd(Y):.2f}x  (>1 => FNO over-fluctuates)")
def ktke(Pp):
    kp=scoring.kinetic_energy(Pp[...,:C]); kt=scoring.kinetic_energy(Y[...,:C])
    return float(np.linalg.norm(kp)/np.linalg.norm(kt))
print(f"TKE magnitude ratio ||TKE_pred||/||TKE_target|| = {ktke(P):.3f}  (1.0 = perfect magnitude)\n")
print(f"{'damping d':>10s}{'rel_l2':>9s}{'tke':>8s}{'mvpe':>8s}{'TKEratio':>10s}")
mean=P.mean(axis=1,keepdims=True)
for d in (1.0,0.9,0.77,0.6,0.4,0.2,0.0):
    Pd=(mean+d*(P-mean)).astype(np.float32); Pd[...,2]=0.0
    r=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(Pd,Y,C))))
    t=scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(Pd,Y,C))))
    m=scoring.score_error(scoring.mvpe_rel_l2(Pd,Y))
    print(f"{d:10.2f}{r:9.2f}{t:8.2f}{m:8.2f}{ktke(Pd):10.3f}")
