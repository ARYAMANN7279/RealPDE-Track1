"""Refit the per-location bound map for the SOUP model on ALL 17 held-out
trajectories, apply the x1.15 safety margin, and save for packaging."""
import json, os, sys
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda:2"; C=2; IN=20
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
mi,si,mt,st=[torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
Wall=np.stack([np.concatenate([np.asarray(X[s:s+40]),
        np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
m,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
def run(sd):
    m.load_state_dict(sd); m.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
ORIG=torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV); ORIG=ORIG.get("model_state_dict",ORIG)
P0=run(ORIG); e0=np.abs(P0[...,:C]-Y[...,:C])
def E_at(e,lam):
    eu=e[...,0][SCM[...,0]]/lam; ev=e[...,1][SCM[...,1]]/lam
    return float((np.exp(-2*0.030/SIG)*(eu<=0.030)).sum()+(np.exp(-2*0.010/SIG)*(ev<=0.010)).sum())/(eu.size+ev.size)
lo,hi=0.02,20.0
for _ in range(60):
    mid=(lo+hi)/2
    if E_at(e0,mid)>0.4399: hi=mid
    else: lo=mid
LAM=1/((lo+hi)/2)
assert abs(E_at(e0,1/LAM)-0.4399)<0.01, "calibration assertion failed"
print("lambda x%.3f [assertion OK]"%LAM,flush=True)
P=run(torch.load(f"{B}/local_harness/soup_best.pth",map_location=DEV))
ERR=np.abs(P[...,:C]-Y[...,:C])*LAM
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
print("  %-8s %8s %9s"%("mult","E","coverage"))
for m_ in [1.0,1.15,1.3]:
    E,cv=sc(HL*m_); print("  x%-7.2f %8.4f %9.3f"%(m_,E,cv))
np.save(f"{B}/local_harness/soup_bounds_final.npy",(HL*1.15).astype(np.float32))
E,cv=sc(HL*1.15)
print("saved soup_bounds_final.npy (x1.15) E=%.4f coverage=%.3f"%(E,cv),flush=True)
