import argparse, hashlib, json, os, shutil, subprocess, sys, time, zipfile
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
LH=f"{B}/local_harness"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
SIG=0.0563870259

ap=argparse.ArgumentParser()
ap.add_argument("--gpu",type=int,default=0)
ap.add_argument("--cache",required=True)
ap.add_argument("--fno_fp16",required=True)
ap.add_argument("--unet",required=True)
ap.add_argument("--tag",required=True)
ap.add_argument("--alpha",type=float,default=0.75)
a=ap.parse_args(); DEV=f"cuda:{a.gpu}"

meta=json.load(open(f"{LH}/tr_meta.json")); names=meta["names"]
d=np.load(f"{B}/train_es/{a.cache}"); wt=d["wt"]

XI=d["XI"]; PR=d["PR"]; RS=d["RS"]; SC=d["SC"]; del d

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return F.gelu(s.n2(s.c2(x)))

class UNet(nn.Module):
    def __init__(s,ci,co,w):
        super().__init__()
        s.e1=Blk(ci,w); s.e2=Blk(w,2*w); s.e3=Blk(2*w,4*w); s.b=Blk(4*w,4*w)
        s.d3=Blk(8*w,2*w); s.d2=Blk(4*w,w); s.d1=Blk(2*w,w); s.out=nn.Conv2d(w,co,1); s.pool=nn.AvgPool2d(2)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(s.pool(e1)); e3=s.e3(s.pool(e2)); b=s.b(s.pool(e3))
        u=F.interpolate(b,size=e3.shape[-2:],mode="bilinear",align_corners=False); d3=s.d3(torch.cat([u,e3],1))
        u=F.interpolate(d3,size=e2.shape[-2:],mode="bilinear",align_corners=False); d2=s.d2(torch.cat([u,e2],1))
        u=F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False); d1=s.d1(torch.cat([u,e1],1))
        return s.out(d1)

ck=torch.load(f"{B}/train_es/{a.unet}",map_location=DEV)
W=int(ck["w"]); net=UNet(80,80,W).to(DEV); net.load_state_dict(ck["sd"]); net.eval()

def flat(z): return torch.from_numpy(np.ascontiguousarray(z)).permute(0,1,4,2,3).reshape(z.shape[0],-1,32,64)
Cc=np.zeros_like(RS); Ww=np.zeros_like(RS)

with torch.no_grad():
    for i in range(0,len(RS),16):
        o=net(torch.cat([flat(XI[i:i+16]),flat(PR[i:i+16])],1).to(DEV))
        Cc[i:i+16]=o[:,:40].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
        Ww[i:i+16]=o[:,40:].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
del XI,PR

EFF=np.abs(RS-a.alpha*Cc).astype(np.float32); del RS,Cc
NB=24; LUT=np.zeros((NB,2),np.float32); ED=np.zeros((2,NB-1),np.float32); SCALE=(2.3199,1.1982)

for ci in range(2):
    m=SC[...,ci]
    s=Ww[...,ci][m][::4].astype(np.float64); e=EFF[...,ci][m][::4].astype(np.float64)*SCALE[ci]
    q=np.quantile(s,np.linspace(0,1,NB+1)[1:-1]); ED[ci]=q
    b=np.digitize(s,q); o=np.argsort(b,kind="stable"); bs=b[o]; es=e[o]
    cut=np.searchsorted(bs,np.arange(NB+1))
    for k in range(NB):
        v=np.sort(es[cut[k]:cut[k+1]])
        if v.size<200: LUT[k,ci]=0.012; continue
        kk=np.arange(1,v.size+1)/v.size
        LUT[k,ci]=float(v[np.argmax(np.exp(-2*v/SIG)*kk)])

OUT_NPZ=f"{B}/train_es/bounds_assets_{a.tag}.npz"
sd={k:v.cpu().numpy().astype(np.float32) for k,v in net.state_dict().items()}
np.savez_compressed(OUT_NPZ,LUT=LUT,ED=ED,w=np.int32(W),alpha=np.float32(a.alpha),
                    mu=np.float32(-6.0),sd_=np.float32(1.5),
                    **{"w_"+k:v for k,v in sd.items()})

BASE=f"{B}/submissions/submission_SOUP_v1.zip"
CK_BYTES=open(f"{B}/train_es/{a.fno_fp16}","rb").read()
OUT=f"{B}/submissions/submission_{a.tag}.zip"

zin=zipfile.ZipFile(BASE); zout=zipfile.ZipFile(OUT,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    if "pycache" in it.filename or it.filename.endswith(".pyc"): continue
    if it.filename=="submission.py":
        zout.writestr(it,open(f"{B}/train_es/submission_shift.py","rb").read())
    elif it.filename=="head_assets.npz":
        i2=zipfile.ZipInfo("bounds_assets.npz",date_time=it.date_time)
        i2.compress_type=it.compress_type; i2.external_attr=it.external_attr
        zout.writestr(i2,open(OUT_NPZ,"rb").read())
    elif it.filename=="sim_real_fno_fp16.pth":
        zout.writestr(it,CK_BYTES)
    else:
        zout.writestr(it,zin.read(it.filename))
zout.close()
