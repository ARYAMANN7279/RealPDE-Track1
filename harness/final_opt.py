import os,sys,glob,importlib.util
import numpy as np, torch
A="/SML_DISK_24TB/rajeshr/Aryamann"; D=f"{A}/UGP"; KIT=f"{A}/starting_kit"; SD=f"{D}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
from datasets import Dataset
DEV="cuda";SIG=scoring.SIGMA_GLOBAL
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
s=6;span=40*s;xs,ys=[],[]
for sid,(u,v) in trajs.items():
    T=u.shape[0]
    for t0 in range(0,T-span,span):
        idx=np.arange(t0,t0+span,s)[:40]
        tr=np.stack([u[idx],v[idx],np.zeros_like(u[idx])],-1);xs.append(tr[:20]);ys.append(tr[20:])
X=np.stack(xs).astype(np.float32);Y=np.stack(ys).astype(np.float32);P=fno(X);C=2
mean=P.mean(axis=1,keepdims=True);per=np.repeat(X[:,-1:],20,axis=1).astype(np.float32);per[...,2]=0.0
def one_m_pm(sco):
    e=(100.0/sco-1.0)/0.5; return 1-e/(0.5+e)
def best_bounds(Pp,wmax):
    ab=np.abs(Y[...,:C]-Pp[...,:C]);sc=(Y[...,:C]!=0.0)
    bb=None
    for hu in np.linspace(0.004,0.06,29):
        for hv in np.linspace(0.002,0.03,29):
            h=np.zeros((1,1,1,1,C),np.float32);h[...,0]=hu;h[...,1]=hv
            ins=(ab<=h);nil=np.broadcast_to(2*h/SIG,ins.shape)
            eff=float(np.mean(np.where(sc,np.exp(-nil)*ins,0.0))/np.mean(sc))
            if bb is None or eff>bb[0]: bb=(eff,hu,hv)
    return bb
print(f"{'candidate':30s}{'rel_l2':>8s}{'mvpe':>8s}{'tkeCERT':>9s}{'eff':>7s}{'SPS':>7s}{'bounds':>16s}")
MVPE_OFF=95.90-92.836
out={}
for nm,Pp,tkecert,tm in [("PURE PERSISTENCE",per,66.67,99.0),
                          ("persistence + 0.5*FNOfluct",(per+0.5*(P-mean)).astype(np.float32),None,91.3),
                          ("persistence + 0.767*FNOfluct",(per+0.767*(P-mean)).astype(np.float32),None,91.3)]:
    Pp=Pp.copy();Pp[...,2]=0.0
    rel=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(Pp,Y,C))))
    mv=scoring.score_error(scoring.mvpe_rel_l2(Pp,Y))-MVPE_OFF
    tk=tkecert if tkecert else 74.03   # conservative: assume no TKE gain
    wmax=100*(0.5*one_m_pm(rel)+0.3*one_m_pm(tk)+0.2*one_m_pm(mv))
    eff,hu,hv=best_bounds(Pp,wmax); sps=wmax*eff
    fin=0.30605*rel+0.16274*tk+0.21812*mv+0.10011*tm+0.21704*sps+0.00961
    out[nm]=(fin,rel,mv,tk,sps,hu,hv,tm)
    print(f"{nm:30s}{rel:8.2f}{mv:8.2f}{tk:9.2f}{eff:7.3f}{sps:7.2f}  [{hu:.3f},{hv:.3f}]  EST FINAL={fin:.2f}")
print(f"\n(current best = 77.20; TKE shown CONSERVATIVELY: 66.67 certain for persistence, 74.03 = no-gain assumption for hybrids)")
import json; json.dump({k:[float(x) for x in v] for k,v in out.items()},open(f"{D}/local_harness/final_opt.json","w"))
