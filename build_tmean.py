"""Build the time-mean candidate.

prediction += mp_alpha * mean_t(mp(ui))   <- the NEW, live-untested lever
lower/upper                               <- computed from the UNCORRECTED yb, so the
                                             bound geometry that scored 79.3418 is kept

Space: the main bounds net is stored float32 (30.5 MB) while the two ensemble extras are
already float16 (15.3 MB each). Casting the main net to fp16 frees exactly 15.25 MB --
enough for the W96 mean predictor -- so NO ensemble member is dropped.
"""
import os, shutil, subprocess, sys
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; D=f"{B}/_bld"
z=dict(np.load(f"{D}/bounds_assets.npz"))
before=sum(v.nbytes for v in z.values())
for k in list(z):
    if k.startswith("w_") and z[k].dtype==np.float32:
        z[k]=z[k].astype(np.float16)
after_main=sum(v.nbytes for v in z.values())
ck=torch.load(f"{B}/train_es/mean_predictor_W96_d01.pth",map_location="cpu",weights_only=False)
sd=ck.get("sd",ck)
for k,v in sd.items():
    z["mp_"+k]=v.detach().cpu().numpy().astype(np.float16)
z["mp_w"]=np.int32(96)
z["mp_alpha"]=np.float32(0.75)          # best alpha on the honest re_lohi split
after=sum(v.nbytes for v in z.values())
print("bounds_assets: %.1f MB -> %.1f MB after fp16 main -> %.1f MB with mean predictor"
      %(before/1e6, after_main/1e6, after/1e6))
np.savez_compressed(f"{D}/bounds_assets.npz", **z)

s=open(f"{D}/submission.py").read()
anchor='''    return _state["net"], _state["lut"], _state["ed"]'''
assert anchor in s
s=s.replace(anchor,'''        if "mp_w" in z.files:
            mp=_MeanNet(80,2,int(z["mp_w"]))
            mp.load_state_dict({k[3:]: torch.from_numpy(z[k].astype(np.float32))
                                for k in z.files if k.startswith("mp_")
                                and k not in ("mp_w","mp_alpha")})
            _state["mp"]=mp.to(device).eval()
            _state["mp_alpha"]=float(z["mp_alpha"])
        else:
            _state["mp"]=None; _state["mp_alpha"]=0.0
'''+anchor,1)

# a 2-output U-Net identical in body to _UNet
mean_cls='''

class _MeanNet(nn.Module):
    """Predicts mean_t(residual): (B,80,32,64) -> (B,2,32,64). A time-CONSTANT correction
    leaves u - mean_t(u) unchanged, so tke is algebraically invariant (verified 1.8e-06)."""
    def __init__(s, ci, co, w):
        super().__init__()
        s.e1=_Blk(ci,w); s.e2=_Blk(w,2*w); s.e3=_Blk(2*w,4*w); s.b=_Blk(4*w,4*w)
        s.d3=_Blk(8*w,2*w); s.d2=_Blk(4*w,w); s.d1=_Blk(2*w,w)
        s.out=nn.Conv2d(w,co,1); s.pool=nn.AvgPool2d(2)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(s.pool(e1)); e3=s.e3(s.pool(e2)); b=s.b(s.pool(e3))
        u=F.interpolate(b,size=e3.shape[-2:],mode="bilinear",align_corners=False); d3=s.d3(torch.cat([u,e3],1))
        u=F.interpolate(d3,size=e2.shape[-2:],mode="bilinear",align_corners=False); d2=s.d2(torch.cat([u,e2],1))
        u=F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False); d1=s.d1(torch.cat([u,e1],1))
        return s.out(d1)
'''
s=s.replace("def _get_net(device):", mean_cls+"\n\ndef _get_net(device):",1)

# apply AFTER the bounds are built from the uncorrected yb -> bound geometry preserved
old='''            yb[..., 2] = 0.0                # p is unmeasured in real data'''
new='''            if _state.get("mp") is not None and _state["mp_alpha"] != 0.0:
                for k in range(0, j - i, CH):
                    m = min(k + CH, j - i)
                    ui = torch.cat([_fold(xb_raw[k:m, :, :, :, :2]),
                                    _fold(yb[k:m, :, :, :, :2])], 1)
                    mc = _state["mp"](ui) * _state["mp_alpha"]       # (b,2,32,64)
                    yb[k:m, :, :, :, :2] += mc.permute(0, 2, 3, 1).unsqueeze(1)
            yb[..., 2] = 0.0                # p is unmeasured in real data'''
assert old in s; s=s.replace(old,new,1)
open(f"{D}/submission.py","w").write(s)
print("patched submission.py: _MeanNet + prediction correction AFTER bounds")
subprocess.run([sys.executable,"-m","py_compile",f"{D}/submission.py"],check=True)
print("compiles OK")
out=f"{B}/submissions/submission_TMEAN.zip"
if os.path.exists(out): os.remove(out)
subprocess.run(["zip","-q","-r","-X","-D",out,"."],cwd=D,check=True)
ext=int(subprocess.run(["unzip","-l",out],capture_output=True,text=True).stdout.strip().split("\n")[-1].split()[0])
print("built %s : extracted %d B = %.2f%% of cap  %s"%(os.path.basename(out),ext,100*ext/268435456,
      "PASS" if ext<268435456 else "*** OVER ***"))
