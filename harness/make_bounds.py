"""Refit the per-location bound map on ALL 17 held-out trajectories (the model
never trained on them), and quantify the cost of a safety margin.

Bounds slightly too TIGHT score exactly zero for that element; too wide decays
only as exp(-2h/sigma). Given calibration uncertainty, buying margin is cheap
insurance -- this prints the E cost of each multiplier so the choice is explicit.
"""
import json, os, sys
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda:1"; C=2; IN=20
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
mi,si,mt,st=[torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
wins=np.array(wins)
def run(ck):
    m,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
    if ck: m.load_state_dict(torch.load(ck,map_location=DEV))
    m=m.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(wins),32):
            w=np.stack([np.asarray(X[s:s+40]) for s in wins[i:i+32]])
            z=np.zeros(w.shape[:-1]+(1,),np.float32); w=np.concatenate([w,z],-1)
            o.append((m((torch.from_numpy(w[:,:IN]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
Y=[]
for i in range(0,len(wins),32):
    w=np.stack([np.asarray(X[s:s+40]) for s in wins[i:i+32]])
    z=np.zeros(w.shape[:-1]+(1,),np.float32); Y.append(np.concatenate([w,z],-1)[:,IN:])
Y=np.concatenate(Y,0).astype(np.float32); SCM=(Y[...,:C]!=0.0)
# lambda anchored on the ORIGINAL model over these same windows
P0=run(None); e0=np.abs(P0[...,:C]-Y[...,:C])
def E_at(e,lam):
    eu=e[...,0][SCM[...,0]]/lam; ev=e[...,1][SCM[...,1]]/lam
    return float((np.exp(-2*0.030/SIG)*(eu<=0.030)).sum()+(np.exp(-2*0.010/SIG)*(ev<=0.010)).sum())/(eu.size+ev.size)
lo,hi=0.02,20.0
for _ in range(60):
    mid=(lo+hi)/2
    if E_at(e0,mid)>0.4399: hi=mid
    else: lo=mid
mid=(lo+hi)/2; LAM=1/mid
print("lambda: local errors x %.3f to reach real scale (check E=%.4f)"%(LAM,E_at(e0,mid)),flush=True)
CK=f"{B}/local_harness/ft_w015_best.pth"
P=run(CK); ERR=np.abs(P[...,:C]-Y[...,:C])*LAM
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
HL=np.zeros((32,64,C),np.float32)
for ci in range(C):
    E=ERR[...,ci]; M=SCM[...,ci]
    for i in range(32):
        for j in range(64): HL[i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]])
def sc(H):
    h=np.broadcast_to(H[None,None],ERR.shape); ok=(ERR<=h)&SCM
    return float((np.exp(-2*h/SIG)*ok).sum()/SCM.sum()), float(ok.sum()/SCM.sum())
print("\nsafety margin cost (fit on all 17 held-out trajectories):")
print("  %-10s %8s %9s %10s"%("mult","E","coverage","real sps"))
for m_ in [1.0,1.1,1.2,1.35,1.5]:
    E,cv=sc(HL*m_); print("  x%-9.2f %8.4f %9.3f %10.2f"%(m_,E,cv,100*0.7716*E))
CHOICE=1.15
np.save(f"{B}/local_harness/bounds_final.npy",(HL*CHOICE).astype(np.float32))
E,cv=sc(HL*CHOICE)
print("\nchose x%.2f -> E=%.4f coverage=%.3f ; saved bounds_final.npy %s"%(CHOICE,E,cv,HL.shape))
