"""Build the off-centre-bounds submission: FNO checkpoint + joint 80->80 U-Net
(centre correction + width score) + 24-bin LUT fitted on the SHIFTED residual.

--ckpt soup   : copies the checkpoint entry byte-for-byte out of submission_SOUP_v1.zip
                (predictions then provably bit-identical to the artifact that scored)
--ckpt w15lr3 : packs local_harness/ft_long_w15lr3_best.pth to fp16 with the kit's own
                pack_ckpt_fp16.py (complex spectral tensors via view_as_real, see TEST.md)
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, zipfile
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
SIG=0.0563870259
ap=argparse.ArgumentParser()
ap.add_argument("--gpu",type=int,default=0); ap.add_argument("--joint",required=True)
ap.add_argument("--cache",required=True); ap.add_argument("--ckpt",default="soup",)
ap.add_argument("--alpha",type=float,default=1.0); ap.add_argument("--nb",type=int,default=24)
ap.add_argument("--au",type=float,default=2.3199); ap.add_argument("--av",type=float,default=1.1982)
ap.add_argument("--sub",type=int,default=4); ap.add_argument("--fitset",default="all",choices=["all","nolohi"])
ap.add_argument("--tag",required=True)
a=ap.parse_args(); DEV=f"cuda:{a.gpu}"
meta=json.load(open(f"{LH}/tr_meta.json")); names=meta["names"]
d=np.load(f"{B}/train_es/{a.cache}"); wt=d["wt"]
RE=np.array([int(names[i].split("_")[0]) for i in wt])
FITW=np.ones(len(wt),bool) if a.fitset=="all" else ~np.isin(RE,[3750,5025,25425,26700])
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
ck=torch.load(f"{B}/train_es/{a.joint}",map_location=DEV,weights_only=False)
W=int(ck["w"]); net=UNet(80,120,W).to(DEV); net.load_state_dict(ck["sd"]); net.eval()
NP=sum(p.numel() for p in net.parameters())
print("[net] width %d  %.2fM params (%.1f MB fp32) | alpha %.2f"%(W,NP/1e6,NP*4/1e6,a.alpha),flush=True)
def flat(z): return torch.from_numpy(np.ascontiguousarray(z)).permute(0,1,4,2,3).reshape(z.shape[0],-1,32,64)
Cc=np.zeros_like(RS); Ww=np.zeros_like(RS)
t0=time.time()
Ww_d=np.zeros_like(RS); Ww_u=np.zeros_like(RS)
with torch.no_grad():
    for i in range(0,len(RS),16):
        o=net(torch.cat([flat(XI[i:i+16]),flat(PR[i:i+16])],1).to(DEV))
        Cc[i:i+16]=o[:,:40].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
        Ww_d[i:i+16]=o[:,40:80].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
        Ww_u[i:i+16]=o[:,80:].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()
print("[scored] %.0fs"%(time.time()-t0),flush=True)
del XI,PR
EFF_D=np.maximum(-(RS-a.alpha*Cc), 0).astype(np.float32)
EFF_U=np.maximum(RS-a.alpha*Cc, 0).astype(np.float32)
del RS,Cc
NB=a.nb; LUT_D=np.zeros((NB,2),np.float32); ED_D=np.zeros((2,NB-1),np.float32); SCALE=(a.au,a.av)
LUT_U=np.zeros((NB,2),np.float32); ED_U=np.zeros((2,NB-1),np.float32)
print("[fit] %d windows (%s) | NB %d | err scale u x%.4f v x%.4f"%(FITW.sum(),a.fitset,NB,*SCALE),flush=True)


def fit_lut(Ww, EFF):
    LUT=np.zeros((NB,2),np.float32); ED=np.zeros((2,NB-1),np.float32)
    for ci in range(2):
        m=np.zeros(EFF.shape[:-1],bool); m[FITW]=True; m&=SC[...,ci]
        s=Ww[...,ci][m][::a.sub].astype(np.float64); e=EFF[...,ci][m][::a.sub].astype(np.float64)*SCALE[ci]
        q=np.quantile(s,np.linspace(0,1,NB+1)[1:-1]); ED[ci]=q
        b=np.digitize(s,q); o=np.argsort(b,kind="stable"); bs=b[o]; es=e[o]
        cut=np.searchsorted(bs,np.arange(NB+1))
        for k in range(NB):
            v=np.sort(es[cut[k]:cut[k+1]])
            if v.size<200: LUT[k,ci]=0.012; continue
            
            # Conformal Prediction: Use 95th percentile
            idx = int(0.99 * v.size)
            if idx >= v.size: idx = v.size - 1
            LUT[k,ci] = float(v[idx])
            
        print("   ch%d %d elem | LUT min %.5f med %.5f max %.5f | %d/%d non-monotone"
              %(ci,s.size,LUT[:,ci].min(),np.median(LUT[:,ci]),LUT[:,ci].max(),
                int((np.diff(LUT[:,ci])<0).sum()),NB-1),flush=True)
    return LUT, ED


print("Fitting LUT_D:")
LUT_D, ED_D = fit_lut(Ww_d, EFF_D)
print("Fitting LUT_U:")
LUT_U, ED_U = fit_lut(Ww_u, EFF_U)

assert np.isfinite(LUT_D).all() and np.isfinite(ED_D).all() and (LUT_D>=0).all() and LUT_D.max()<0.10
assert np.isfinite(LUT_U).all() and np.isfinite(ED_U).all() and (LUT_U>=0).all() and LUT_U.max()<0.10
sd={k:v.cpu().numpy().astype(np.float32) for k,v in net.state_dict().items()}
OUT_NPZ=f"{B}/train_es/bounds_assets_{a.tag+"_q99"}.npz"
np.savez_compressed(OUT_NPZ,LUT_D=LUT_D,ED_D=ED_D,LUT_U=LUT_U,ED_U=ED_U,w=np.int32(W),alpha=np.float32(a.alpha),
                    mu=np.float32(ck["mu"]),sd_=np.float32(ck["sd_"]),
                    **{"w_"+k:v for k,v in sd.items()})
print("[assets] %s  %.2f MB"%(OUT_NPZ,os.path.getsize(OUT_NPZ)/1e6),flush=True)
# ---- checkpoint ----
BASE=f"{B}/submissions/submission_SOUP_v1.zip"
CK_BYTES=None
if a.ckpt=="soup_v3":
    CK_BYTES=open(f"{B}/train_es/soup_v3_fp16.pth","rb").read()
    print("[ckpt] loaded soup_v3_fp16.pth")
elif a.ckpt=="soup_v2":
    CK_BYTES=open(f"{LH}/soup_v2_fp16.pth","rb").read()
    print("[ckpt] loaded soup_v2_fp16.pth")
elif a.ckpt=="soup":
    CK_BYTES=zipfile.ZipFile(BASE).read("sim_real_fno_fp16.pth")
    print("[ckpt] copied byte-for-byte from SOUP_v1 (md5 %s)"%hashlib.md5(CK_BYTES).hexdigest())
else:
    src=f"{LH}/ft_long_w15lr3_best.pth"; dst=f"{B}/train_es/w15lr3_fp16.pth"
    if not os.path.exists(dst):
        r=subprocess.run([sys.executable,f"{KIT}/pack_ckpt_fp16.py",src,dst],capture_output=True,text=True)
        print("[pack]",r.stdout.strip(),r.stderr.strip()[:400])
    raw=torch.load(dst,map_location="cpu",weights_only=False)
    print("[ckpt] packed: %d tensors, %d complex, %.1f MB"
          %(len(raw["state_fp16"]),len(raw["complex_keys"]),os.path.getsize(dst)/1e6))
    assert len(raw["complex_keys"])==16, "expected 16 COMPLEX spectral tensors, got %d"%len(raw["complex_keys"])
    CK_BYTES=open(dst,"rb").read()
OUT=f"{B}/submissions/submission_{a.tag+"_q99"}.zip"
if os.path.exists(OUT): os.remove(OUT)
zin=zipfile.ZipFile(BASE); zout=zipfile.ZipFile(OUT,"w",zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    assert "__pycache__" not in it.filename and not it.filename.endswith(".pyc")
    if it.filename=="submission.py":
        zout.writestr(it,open(f"{B}/train_es/submission_asym.py","rb").read())
    elif it.filename=="head_assets.npz":
        i2=zipfile.ZipInfo("bounds_assets.npz",date_time=it.date_time)
        i2.compress_type=it.compress_type; i2.external_attr=it.external_attr
        zout.writestr(i2,open(OUT_NPZ,"rb").read())
    elif it.filename=="sim_real_fno_fp16.pth":
        zout.writestr(it,CK_BYTES)
    else:
        zout.writestr(it,zin.read(it.filename))
zout.close()
zz=zipfile.ZipFile(OUT); nl=[i.filename for i in zz.infolist()]
assert nl.count("sim_real_fno_fp16.pth")==1 and nl.count("submission.py")==1
assert not [n for n in nl if n.endswith(".pyc") or "__pycache__" in n or n.endswith("/")]
tot=sum(i.file_size for i in zz.infolist()); old=[i.filename for i in zipfile.ZipFile(BASE).infolist()]
md5=hashlib.md5(open(OUT,"rb").read()).hexdigest()
print("[zip] %s\n  entries %d (base %d) | added %s | removed %s"
      %(OUT,len(nl),len(old),sorted(set(nl)-set(old)),sorted(set(old)-set(nl))))
print("  extracted %.1f MB (cap 256) | zipped %.1f MB | md5 %s"%(tot/1048576,os.path.getsize(OUT)/1048576,md5))
assert tot/1048576<256, "OVER THE 256 MB CAP"
json.dump({"alpha":a.alpha,"au":a.au,"av":a.av,"nb":NB,"ckpt":a.ckpt,"joint":a.joint,
           "params":int(NP),"md5":md5,"extracted_mb":tot/1048576,
           "lut_med_d":[float(np.median(LUT_D[:,0])),float(np.median(LUT_D[:,1]))]},
          open(f"{B}/train_es/build_{a.tag+"_q99"}.json","w"),indent=1)
print("[done]")
