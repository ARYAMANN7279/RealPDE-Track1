"""Fine-tune the kit FNO WITH AUGMENTATION on a genuinely condition-disjoint holdout.

sec14 established a ceiling across 7 axes (hyperparams, steps, data amount, holdout size,
init, soup size/diversity) -- but never varied AUGMENTATION, and every run saw <2 epochs
of a 67k-window set with a 100M-parameter model.  Augmentations here, all of which keep
every sample traceable to the released data:
  --aug phase : sample one of the 4 valid 2x-subsample phases of the native 64x128 field
  --noise s   : additive Gaussian noise on the INPUT window, s * per-channel std
  --tshift    : (implicit) stride-1 window starts, as before
Validation is always phase (0,0), the phase the competition evaluates.
"""
import argparse, json, os, sys, time
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=3e-5); ap.add_argument("--gpu",type=int,default=0)
ap.add_argument("--steps",type=int,default=40000); ap.add_argument("--bs",type=int,default=16)
ap.add_argument("--wtke",type=float,default=0.15); ap.add_argument("--tag",required=True)
ap.add_argument("--split",default="re_lohi"); ap.add_argument("--aug",default="phase")
ap.add_argument("--noise",type=float,default=0.0); ap.add_argument("--wd",type=float,default=1e-6)
ap.add_argument("--init",default=""); ap.add_argument("--ema",type=float,default=0.0)
ap.add_argument("--dropout",type=float,default=0.0)
ap.add_argument("--amse",type=float,default=0.0)   # amplitude/coherence-decoupled spectral loss
ap.add_argument("--nbins",type=int,default=10)
ap.add_argument("--evalevery",type=int,default=2000); ap.add_argument("--seed",type=int,default=1234)
ap.add_argument("--snap_at",default=""); ap.add_argument("--stop_at",type=int,default=0)   # FITBASE: snapshots / early stop
a=ap.parse_args(); DEV=f"cuda:{a.gpu}"; C=2
# FITBASE: outputs go to agents/fitbase/ckpt.  sv3_3e5_10 (banked) is ftmv.py's STEP-2000 save of a 16000-step
# OneCycle run (its evaluate() returned 0 on split none, so only the first eval ever saved) -> --snap_at 2000.
OUTD=f"{B}/agents/fitbase/ckpt"; SNAP=set(int(s) for s in a.snap_at.split(",") if s.strip()); STOPPED=0
torch.manual_seed(a.seed); np.random.seed(a.seed)
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]
meta=json.load(open(f"{LH}/tr_meta.json")); off,lens,names=meta["off"],meta["lens"],meta["names"]
ntraj=len(lens)
RE=np.array([int(n.split("_")[0]) for n in names])
AOA=np.array([n.split("_")[1].split(".")[0] for n in names])
if   a.split=="re_lohi": VT=set(np.where(np.isin(RE,[3750,5025,25425,26700]))[0].tolist())
elif a.split=="aoa15":   VT=set(np.where(AOA=="15")[0].tolist())
elif a.split=="aoa0":    VT=set(np.where(AOA=="0")[0].tolist())
elif a.split=="every5":  VT=set(range(0,ntraj,5))
elif a.split=="none":    VT=set()          # 100% DATA: train on every trajectory.
else: raise SystemExit("unknown split")
USE_FULL = ("phase" in a.aug)
if USE_FULL: XF=np.load(f"{LH}/tr_full64.npy",mmap_mode="r")
X32=np.load(f"{LH}/tr_frames.npy",mmap_mode="r")
starts_tr=[]; starts_va=[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39):
        (starts_va if i in VT else starts_tr).append(t0)
starts_tr=np.array(starts_tr); starts_va=np.array(starts_va)
if len(starts_va)==0:
    # split=none: nothing is held out, so the monitor set is the standard re_lohi windows.
    # LEAKY BY CONSTRUCTION and intended: it only serves as a progress monitor here.
    # sec127/sec128: do NOT rank candidates by it.
    RE_=np.array([int(n.split("_")[0]) for n in names])
    VT_=set(np.where(np.isin(RE_,[3750,5025,25425,26700]))[0].tolist())
    starts_va=np.array([t0 for i in range(ntraj) if i in VT_
                        for t0 in range(off[i],off[i]+lens[i]-39)])
rng=np.random.default_rng(0)
va_sub=rng.choice(starts_va,size=min(900,len(starts_va)),replace=False)
print("[data] train %d win | val %d win (%d traj, split %s) | aug=%s noise=%.3f"
      %(len(starts_tr),len(va_sub),len(VT),a.split,a.aug,a.noise),flush=True)
SDU=float(np.asarray(X32[::97,...,0]).std()); SDV=float(np.asarray(X32[::97,...,1]).std())
NOI=torch.tensor([SDU,SDV,0.0]).to(DEV)*a.noise
def batch_tr(idx,g):
    if USE_FULL:
        py=g.integers(0,2,len(idx)); px=g.integers(0,2,len(idx))
        w=np.stack([np.asarray(XF[s:s+40,py[j]::2,px[j]::2]) for j,s in enumerate(idx)])
    else:
        w=np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z=np.zeros(w.shape[:-1]+(1,),np.float32); w=np.concatenate([w,z],-1).astype(np.float32)
    x=torch.from_numpy(w[:,:20]).to(DEV); y=torch.from_numpy(w[:,20:]).to(DEV)
    if a.noise>0: x=x+torch.randn_like(x)*NOI
    return x,y
def batch_va(idx):
    w=np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z=np.zeros(w.shape[:-1]+(1,),np.float32); w=np.concatenate([w,z],-1).astype(np.float32)
    return torch.from_numpy(w[:,:20]).to(DEV), torch.from_numpy(w[:,20:]).to(DEV)
init = a.init if a.init else f"{B}/data/comp_real/sim_real_fno.pth"
model,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
if a.init:
    sd=torch.load(a.init,map_location=DEV)
    if isinstance(sd,dict) and "model_state_dict" in sd: sd=sd["model_state_dict"]
    model.load_state_dict(sd)
model=model.to(DEV)

if a.dropout>0:
    import torch.nn.functional as _F
    def _mk(pp):
        def _h(mod,inp,out): return _F.dropout(out,p=pp,training=mod.training)
        return _h
    _n=0
    for _m in model.modules():
        if "Spectral" in type(_m).__name__:
            _m.register_forward_hook(_mk(a.dropout)); _n+=1
    print(f"[dropout] p={a.dropout} on {_n} spectral convs",flush=True)
_EMA=None
if a.ema>0:
    # sec82.1: .float() DROPS the imaginary part of the complex spectral weights.
    _EMA={k:(v.detach().clone() if v.is_complex() else v.detach().clone().float())
          for k,v in model.state_dict().items()}
    print(f"[ema] decay={a.ema}",flush=True)
def _ema_update():
    if _EMA is None: return
    sd=model.state_dict()
    for k,v in sd.items():
        if v.is_complex():
            _EMA[k].mul_(a.ema).add_(v.detach(),alpha=1-a.ema)
        elif v.dtype.is_floating_point:
            _EMA[k].mul_(a.ema).add_(v.detach().float(),alpha=1-a.ema)
        else:
            _EMA[k]=v.detach().clone().float()
import contextlib
@contextlib.contextmanager
def _use_ema():
    if _EMA is None:
        yield; return
    bak={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict({k:_EMA[k].to(v.dtype) for k,v in bak.items()})
    try: yield
    finally: model.load_state_dict(bak)
def fwd(x): return model((x-MI)/SI)*ST+MT
def rel_l2(p,t):
    p=p[...,:C].reshape(p.shape[0],-1); t=t[...,:C].reshape(t.shape[0],-1)
    return (torch.linalg.norm(p-t,dim=1)/torch.linalg.norm(t,dim=1).clamp(min=1e-8)).mean()
_WY=_WX=_IDX=_NB=None
def _amse_setup(H,W,dev,nb):
    """Hann windows (domain is NOT periodic) + radial band indices, built once."""
    global _WY,_WX,_IDX,_NB
    if _IDX is not None and _NB==nb: return
    _WY=torch.hann_window(H,periodic=False,device=dev).view(1,1,-1,1,1)
    _WX=torch.hann_window(W,periodic=False,device=dev).view(1,1,1,-1,1)
    ky=torch.fft.fftfreq(H,device=dev).view(-1,1)*H
    kx=torch.fft.rfftfreq(W,device=dev).view(1,-1)*W
    kr=torch.sqrt(ky**2+kx**2).flatten()
    edges=torch.linspace(0.0,float(kr.max())+1e-6,nb+1,device=dev)
    _IDX=torch.clamp(torch.bucketize(kr,edges[1:-1],right=False),0,nb-1); _NB=nb

def amse_loss(p,t,nb):
    """AMSE (Subich et al. 2501.19374), on the TEMPORAL FLUCTUATION field.
    MSE's cross term is 2*sqrt(Pp*Pt)*Coh, so where coherence is low the optimiser is REWARDED for
    shrinking amplitude -- that is the double penalty, and sec114 shows it caps tke at rho=cos0.
    AMSE replaces it:  (sqrt(Pp)-sqrt(Pt))^2  +  2*max(Pp,Pt)*(1-Coh)
    The first term drives amplitude to truth unconditionally; the second drives COHERENCE (= cos0,
    the only lever sec116 says is big enough) and is weighted by the LARGER power so it cannot be
    gamed by shrinking. Parameter-free apart from the band count."""
    p=p[...,:C]; t=t[...,:C]
    pf=p-p.mean(1,keepdim=True); tf=t-t.mean(1,keepdim=True)
    H,W=p.shape[2],p.shape[3]; _amse_setup(H,W,p.device,nb)
    Pf=torch.fft.rfft2(pf*_WY*_WX,dim=(2,3)); Tf=torch.fft.rfft2(tf*_WY*_WX,dim=(2,3))
    Pf=Pf.reshape(Pf.shape[0],Pf.shape[1],-1,Pf.shape[-1])
    Tf=Tf.reshape(Tf.shape[0],Tf.shape[1],-1,Tf.shape[-1])
    Pp=(Pf.real**2+Pf.imag**2); Pt=(Tf.real**2+Tf.imag**2)
    cr=Pf*Tf.conj()                                   # cross-spectrum
    def band(x):                                      # sum within radial bands -> (nb,)
        y=x.sum(dim=(0,1,3))                          # over batch, time, channel
        return torch.zeros(nb,device=x.device,dtype=y.dtype).index_add_(0,_IDX,y)
    bp=band(Pp); bt=band(Pt)
    bcr=torch.zeros(nb,device=p.device,dtype=cr.dtype).index_add_(0,_IDX,cr.sum(dim=(0,1,3)))
    eps=1e-12
    coh=bcr.abs()/torch.sqrt(bp*bt+eps).clamp(min=eps)          # magnitude-squared coherence^(1/2)
    amp=(torch.sqrt(bp+eps)-torch.sqrt(bt+eps))**2
    inc=2.0*torch.maximum(bp,bt)*(1.0-coh.clamp(0,1))
    return ((amp+inc).sum()/(bt.sum()+eps))

def tke_l2(p,t):
    def ke(z):
        u,v=z[...,0],z[...,1]
        return 0.5*(((u-u.mean(1,keepdim=True))**2).mean(1)+((v-v.mean(1,keepdim=True))**2).mean(1))
    pk=ke(p[...,:C]).reshape(p.shape[0],-1); tk=ke(t[...,:C]).reshape(t.shape[0],-1)
    return (torch.linalg.norm(pk-tk,dim=1)/torch.linalg.norm(tk,dim=1).clamp(min=1e-8)).mean()
@torch.no_grad()
def evaluate():
    model.eval(); P=[];T=[]
    for i in range(0,len(va_sub),16):
        x,y=batch_va(va_sub[i:i+16]); P.append(fwd(x).cpu().numpy()); T.append(y.cpu().numpy())
    model.train()
    P=np.concatenate(P,0).astype(np.float32); T=np.concatenate(T,0).astype(np.float32)
    dm=S.rel_l2_per_sample(P,T,C); tk=S.tke_rel_l2_per_sample(P,T,C); mv=S.mvpe_rel_l2_per_sample(P,T)
    def _tkemap(z):
        f=z[...,:C]-z[...,:C].mean(axis=1,keepdims=True)
        return 0.5*(f[...,0]**2).mean(1)+0.5*(f[...,1]**2).mean(1)
    Tp=_tkemap(P).astype(np.float64); Tt=_tkemap(T).astype(np.float64)
    nn=lambda a: np.sqrt((a**2).sum())
    global LAST_COS,LAST_RHO
    LAST_COS=float((Tp*Tt).sum()/max(nn(Tp)*nn(Tt),1e-12))
    LAST_RHO=float(nn(Tp)/max(nn(Tt),1e-12))
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
LAST_COS=0.0; LAST_RHO=0.0
MV=dict(rel_l2=0.669,tke=0.157,mvpe=0.170)   # CORRECTED: incl. W->sps channel
base=evaluate()
def dacc(s): return MV['rel_l2']*(s[0]-base[0])+MV['tke']*(s[1]-base[1])+MV['mvpe']*(s[2]-base[2])
print("[baseline val] rel_l2 %.4f tke %.4f mvpe %.4f"%base,flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.wd)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.steps,pct_start=0.05)
g=np.random.default_rng(a.seed)
best=-9e9; t0=time.time(); HIST=[]; model.train()
for step in range(1,a.steps+1):
    idx=g.choice(starts_tr,size=a.bs,replace=False)
    x,y=batch_tr(idx,g)
    loss=None
    p=fwd(x); loss=rel_l2(p,y)+a.wtke*tke_l2(p,y)
    if a.amse>0: loss=loss+a.amse*amse_loss(p,y,a.nbins)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
    _ema_update()
    if step%a.evalevery==0 or step==a.steps:
        with _use_ema():
            r=evaluate()
        d=dacc(r); flag=""
        if d>best:
            best=d
            with _use_ema(): torch.save(model.state_dict(),f"{OUTD}/ftaug_{a.tag}.pth")
            flag=" *"
        print("step %6d loss %7.4f | rel_l2 %.4f (%+.4f) tke %.4f (%+.4f) mvpe %.4f (%+.4f) | cos0 %.4f rho %.4f | d_acc %+.4f | %.0fs%s"
              %(step,float(loss.detach()),r[0],r[0]-base[0],r[1],r[1]-base[1],r[2],r[2]-base[2],LAST_COS,LAST_RHO,d,time.time()-t0,flag),flush=True)
        json.dump({"base":base,"best_dacc":best,"args":vars(a)},open(f"{OUTD}/ftaug_{a.tag}.json","w"),default=float,indent=1)
    # ---- FITBASE additions: pure I/O after the optimiser step; the optimisation above is untouched ----
    if step in SNAP:
        with _use_ema(): torch.save(model.state_dict(),f"{OUTD}/ftaug_{a.tag}_s{step}.pth")
        print("[snap] step %d -> ftaug_%s_s%d.pth | next lr %.4e"%(step,a.tag,step,sch.get_last_lr()[0]),flush=True)
    if a.stop_at and step>=a.stop_at:
        STOPPED=step; break
if STOPPED:
    print("[done] stopped at step %d by --stop_at (OneCycle total_steps=%d); no _final saved  %.0fs"%(STOPPED,a.steps,time.time()-t0),flush=True)
else:
    with _use_ema(): torch.save(model.state_dict(),f"{OUTD}/ftaug_{a.tag}_final.pth")
    print("[done] best d_acc %+.4f  %.0fs"%(best,time.time()-t0),flush=True)
