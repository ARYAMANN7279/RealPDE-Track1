import os,sys,glob,importlib.util
import numpy as np, torch
A="/SML_DISK_24TB/rajeshr/Aryamann"; D=f"{A}/UGP"; KIT=f"{A}/starting_kit"; SD=f"{D}/data/real_hf/foil/hf_dataset/real"
sys.path.insert(0,KIT)
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
from datasets import Dataset
SIG=scoring.SIGMA_GLOBAL
def dec(b,t,h,w):
    for dt in (np.float32,np.float16,np.float64):
        if len(b)==t*h*w*np.dtype(dt).itemsize: return np.frombuffer(b,dt).reshape(t,h,w).astype(np.float32)
trajs={}
for sp in sorted(glob.glob(f"{SD}/*.arrow")):
    r=Dataset.from_file(sp)[0]
    trajs[r["sim_id"]]=(dec(r["u"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64],
                        dec(r["v"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::4,::4][:,:32,:64])
C=2
print(f"{'stride':>7s}{'hu':>7s}{'hv':>7s}{'eff':>7s}{'cov':>7s}{'SPS':>7s}{'est final':>11s}")
def one_m_pm(sco):
    e=(100.0/sco-1.0)/0.5; return 1-e/(0.5+e)
for s in (4,6,10):
    span=40*s;xs,ys=[],[]
    for sid,(u,v) in trajs.items():
        T=u.shape[0]
        for t0 in range(0,T-span,span):
            idx=np.arange(t0,t0+span,s)[:40]
            tr=np.stack([u[idx],v[idx],np.zeros_like(u[idx])],-1);xs.append(tr[:20]);ys.append(tr[20:])
    X=np.stack(xs).astype(np.float32);Y=np.stack(ys).astype(np.float32)
    per=np.repeat(X[:,-1:],20,axis=1).astype(np.float32);per[...,2]=0.0
    rel=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(per,Y,C))))
    mv=scoring.score_error(scoring.mvpe_rel_l2(per,Y))-3.06
    wmax=100*(0.5*one_m_pm(rel)+0.3*one_m_pm(66.67)+0.2*one_m_pm(mv))
    ab=np.abs(Y[...,:C]-per[...,:C]);sc=(Y[...,:C]!=0.0)
    for hu,hv in [(0.004,0.006),(0.008,0.008),(0.012,0.010),(0.020,0.014),(0.030,0.020)]:
        h=np.zeros((1,1,1,1,C),np.float32);h[...,0]=hu;h[...,1]=hv
        ins=(ab<=h);nil=np.broadcast_to(2*h/SIG,ins.shape)
        eff=float(np.mean(np.where(sc,np.exp(-nil)*ins,0.0))/np.mean(sc))
        cov=float(np.mean(np.where(sc,ins,False))/np.mean(sc))*100
        sps=wmax*eff
        fin=0.30605*rel+0.16274*66.67+0.21812*mv+0.10011*99.0+0.21704*sps+0.00961
        print(f"{s:7d}{hu:7.3f}{hv:7.3f}{eff:7.3f}{cov:6.0f}%{sps:7.2f}{fin:11.2f}")
    print()
