"""Dump flat scored-element arrays once so the distribution fit can iterate fast."""
import json, os, sys
import numpy as np, torch, torch.nn as nn
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"; SUB=f"{B}/_tmp/soupv1"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor")); sys.path.insert(0,f"{B}/train_soup")
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py")); S=iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
from head_common import feats
DEV="cuda:0"; C=2
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
mi,si,mt,st=[torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
X=np.load(f"{LH}/tr_frames.npy",mmap_mode="r"); meta=json.load(open(f"{LH}/tr_meta.json"))
off,lens,names=meta["off"],meta["lens"],meta["names"]; ntraj=len(lens)
wins,wt=[],[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0); wt.append(i)
wins=np.array(wins); wt=np.array(wt)
W=np.stack([np.concatenate([np.asarray(X[s:s+40]),np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=W[:,:20],W[:,20:]; SCM=(Y[...,:C]!=0.0)
model,_=load_baseline(f"{SUB}/sim_real_fno_fp16.pth",device=DEV); model=model.to(DEV).eval()
P=[]
with torch.no_grad():
    for i in range(0,len(Xin),32):
        xb=torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
        P.append((model((xb-mi)/si)*st+mt).cpu().numpy())
P=np.concatenate(P,0).astype(np.float32); P[...,2]=0.0
ERR_RAW=np.abs(P[...,:C]-Y[...,:C]).astype(np.float32)     # UNSCALED local |error|
z=np.load(f"{SUB}/head_assets.npz"); NF=int(z["nf"]); fmu=z["fmu"]; fsd=z["fsd"]
class Head(nn.Module):
    def __init__(s,nf,w=128):
        super().__init__()
        s.n=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
            nn.Conv2d(w,w,3,padding=8,dilation=8),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1),nn.GELU(),nn.Conv2d(w,2,1))
    def forward(s,x): return s.n(x)
head=Head(NF).to(DEV).eval()
head.load_state_dict({k[2:]:torch.from_numpy(z["w_"+k[2:]]) for k in z.files if k.startswith("w_")})
FT=(feats(P)-fmu)/fsd
def mu_pf():
    o=[]
    with torch.no_grad():
        for i in range(0,len(P),8):
            f=torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            o.append(head(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(o,0)
def mu_ta():
    o=[]
    with torch.no_grad():
        for i in range(0,len(P),8):
            f=torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV)
            m1=head(f.mean(dim=1).permute(0,3,1,2)).permute(0,2,3,1).cpu().numpy()
            o.append(np.repeat(m1[:,None],20,axis=1))
    return np.concatenate(o,0)
MUf,MUt=mu_pf(),mu_ta()
HELD=np.isin(wt,sorted(set(range(0,ntraj,5))))
RE_w=np.array([int(names[i].split("_")[0]) for i in wt],np.int32)
AOA_w=np.array([int(names[i].split("_")[1].replace(".h5","")) for i in wt],np.int16)
REB=np.broadcast_to(RE_w[:,None,None,None,None],SCM.shape)
AOB=np.broadcast_to(AOA_w[:,None,None,None,None],SCM.shape)
HB=np.broadcast_to(HELD[:,None,None,None,None],SCM.shape)
TB=np.broadcast_to(np.arange(SCM.shape[1],dtype=np.int8)[None,:,None,None,None],SCM.shape)
out={}
for ci,nm in ((0,"u"),(1,"v")):
    m=SCM[...,ci]
    out[f"err_{nm}"]=ERR_RAW[...,ci][m].astype(np.float32)
    out[f"muf_{nm}"]=MUf[...,ci][m].astype(np.float32)
    out[f"mut_{nm}"]=MUt[...,ci][m].astype(np.float32)
    out[f"held_{nm}"]=HB[...,ci][m]
    out[f"re_{nm}"]=REB[...,ci][m].astype(np.int32)
    out[f"aoa_{nm}"]=AOB[...,ci][m].astype(np.int16)
    out[f"t_{nm}"]=TB[...,ci][m].astype(np.int8)
np.savez_compressed(f"{B}/train_mvpe/runs/arrays3.npz",**out)
for k,v in out.items(): print(k,v.shape,v.dtype)
print("saved arrays.npz")
