"""Rebuild the uncertainty head assets WITHOUT the ranking-destroying transform.

The bug: the LUT was fitted on errors multiplied per-element by (alpha+beta*|pred|),
which REORDERS elements (~1.6x to ~2.75x across the field) and destroys the ranking
the head exists to provide.

The fix: build the LUT on CLEAN local errors, then apply a single GLOBAL per-channel
scale so the half-widths sit on the real error scale. A per-channel constant cannot
reorder anything within a channel, so the ranking survives.

Global scale comes from the REAL anchors: a Weibull(k, lam) fit to the two constant
anchors gives the real median |err| per channel; divide by the local median.
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; DEV="cuda"
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P0=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
z_old=np.load(f"{B}/local_harness/head_assets.npz")
gt=z_old["gt"]
Fq=np.fft.rfft(P0[...,:C],axis=1)*gt[None,:,None,None,:]
P=P0.copy(); P[...,:C]=np.fft.irfft(Fq,n=20,axis=1)
SCM=(Y[...,:C]!=0.0)
ERR=np.abs(P[...,:C]-Y[...,:C]).astype(np.float32)     # CLEAN. no per-element transform.
# --- real error scale from the two CONSTANT anchors (Weibull, shared k) ---
def Ec(k,lu,lv,hu,hv):
    return 0.5*(np.exp(-2*hu/SIG)*(1-np.exp(-(hu/lu)**k))+np.exp(-2*hv/SIG)*(1-np.exp(-(hv/lv)**k)))
best=None
for k in np.linspace(0.6,2.2,81):
    for lu in np.linspace(0.006,0.030,97):
        for lv in np.linspace(0.003,0.016,53):
            r=abs(Ec(k,lu,lv,0.030,0.010)-0.4399)+abs(Ec(k,lu,lv,0.0129,0.0098)-0.4876)
            if best is None or r<best[0]: best=(r,k,lu,lv)
r,k,lu,lv=best
med_real=[lu*np.log(2)**(1/k), lv*np.log(2)**(1/k)]
med_loc=[float(np.median(ERR[...,0][SCM[...,0]])), float(np.median(ERR[...,1][SCM[...,1]]))]
SCALE=[med_real[0]/med_loc[0], med_real[1]/med_loc[1]]
print("Weibull on real anchors: k=%.2f lam_u=%.4f lam_v=%.4f (resid %.4f)"%(k,lu,lv,r))
print("median |err|  real %.5f/%.5f   local %.5f/%.5f"%(med_real[0],med_real[1],med_loc[0],med_loc[1]))
print("GLOBAL per-channel scale: u x%.3f  v x%.3f   (constants; cannot reorder)"%(SCALE[0],SCALE[1]),flush=True)
# --- head (reuse the trained net; it regresses log|err| and is unaffected) ---
def feats(Pw):
    u,v=Pw[...,0],Pw[...,1]
    gux=np.gradient(u,axis=3);guy=np.gradient(u,axis=2);gvx=np.gradient(v,axis=3);gvy=np.gradient(v,axis=2)
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2); vo=np.abs(gvx-guy)
    lap=np.abs(np.gradient(gux,axis=3)+np.gradient(guy,axis=2))
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tvv=np.repeat(v.std(axis=1,keepdims=True),v.shape[1],axis=1)
    ke=0.5*(u*u+v*v); dev=np.abs(u-u.mean(axis=1,keepdims=True)); dv=np.abs(v-v.mean(axis=1,keepdims=True))
    yy,xx=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,64),indexing="ij")
    yy=np.broadcast_to(yy,u.shape); xx=np.broadcast_to(xx,u.shape)
    tf=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    return np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(lap+1e-6),np.log(tv+1e-6),
        np.log(tvv+1e-6),np.log(ke+1e-9),np.log(dev+1e-6),np.log(dv+1e-6),u,v,xx,yy,tf],-1).astype(np.float32)
class Net(nn.Module):
    def __init__(s,nf,w=128):
        super().__init__()
        s.n=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
            nn.Conv2d(w,w,3,padding=8,dilation=8),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1),nn.GELU(),nn.Conv2d(w,2,1))
    def forward(s,x): return s.n(x)
ck=torch.load(f"{B}/local_harness/unc3_best.pth",map_location=DEV,weights_only=False)
NF=ck["nf"]; net=Net(NF).to(DEV); net.load_state_dict(ck["sd"]); net.eval()
FT=(feats(P)-ck["mu"])/ck["std"]
MU=[]
with torch.no_grad():
    for i in range(0,len(P),8):
        f=torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        MU.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
MU=np.concatenate(MU,0)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.012
    kk=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*kk)])
NB=24; LUT=np.zeros((NB,C),np.float32); ED=np.zeros((C,NB-1),np.float32)
for ci in range(C):
    m=SCM[...,ci]; mu=MU[...,ci][m]; er=ERR[...,ci][m]*SCALE[ci]   # global scale only
    q=np.quantile(mu,np.linspace(0,1,NB+1)[1:-1]); ED[ci]=q
    b=np.digitize(mu,q)
    for kk in range(NB):
        s_=b==kk; LUT[kk,ci]=best_h(er[s_][::3]) if s_.sum()>200 else 0.013
print("\nNEW LUT h_u %.4f .. %.4f (median %.4f)"%(LUT[:,0].min(),LUT[:,0].max(),np.median(LUT[:,0])))
print("OLD LUT h_u %.4f .. %.4f (median %.4f)"%(z_old["LUT"][:,0].min(),z_old["LUT"][:,0].max(),np.median(z_old["LUT"][:,0])))
print("best real CONSTANT is ~0.0150 -- the new median should sit near it, the old was far below")
sd={kk:v.cpu().numpy() for kk,v in net.state_dict().items()}
np.savez_compressed(f"{B}/local_harness/head_assets_v2.npz",gt=gt,LUT=LUT,ED=ED,
    fmu=ck["mu"],fsd=ck["std"],nf=np.int32(NF),**{"w_"+kk:v for kk,v in sd.items()})
print("saved head_assets_v2.npz")
