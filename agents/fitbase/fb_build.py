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
a=ap.parse_args(); DEV=f"cuda:{a.gpu}"; C=2
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

# =====================================================================================================
# FITBASE screen / build  (agents/fitbase, 11 Sep 2026)
# Lines 1-197 above are r50_build.py lines 1-197 VERBATIM (= r45_train.py's data / model / evaluate()).
# Invoke exactly like run_r50.sh did (the trainer args only select the 900 re_lohi monitor windows):
#   FB_MODE=screen CUDA_VISIBLE_DEVICES=0 $P fb_build.py --tag fb --split re_lohi --aug none --steps 1 --evalevery 1
#   FB_MODE=build FB_ARM=<arm label> FB_ZIPTAG=<tag>  <same args>  -> submissions/submission_FITBASE_<tag>.zip
# Screening rule (brief): local-negative => dead; advance only if 0 < dfinal < 0.20; FLAG leakage-signature
# if 9.5146*dcos0 >= dfinal; cos0 ingredient ceiling 0.8734 (SHARED_CONTEXT sec4.5).
# Evaluations are cached in screen_cache.json keyed on leaf (name, size, mtime).  BANKED is re-evaluated on
# every run and the whole cache is discarded if it does not reproduce to 1e-6.
# =====================================================================================================
import zipfile, hashlib, math, shutil, time as _time
LH=f"{B}/local_harness"; FB=f"{B}/agents/fitbase"; CK=f"{FB}/ckpt"
MODE=os.environ.get("FB_MODE","screen")
PRICE=dict(rel=0.6133,tke=0.1628,mvpe=0.1700)   # effective final per LOCAL point (SHARED_CONTEXT sec4.8)
LEAKSLOPE=9.5146                                  # sec128.3: dfinal = 9.5146*cos0 - 8.2480 (r2 0.9885)
COSGATE=0.8734                                    # sec127.3 ingredient ceiling (= soup_v2's cos0)
KITBASE=(95.4738,75.8957,96.0945)                 # r12_eval.BASE (selftest)
T0=_time.time()

bad=[i for i in range(3) if abs(base[i]-KITBASE[i])>1e-3]
print("\n=== HARNESS GATE: kit base %.4f / %.4f / %.4f  (r12 selftest wants %.4f / %.4f / %.4f)  ->  %s"
      %(base[0],base[1],base[2],KITBASE[0],KITBASE[1],KITBASE[2],"FAIL" if bad else "PASS"),flush=True)
if bad: raise SystemExit("HARNESS GATE FAILED - screen nothing")
print("  monitor windows: %d (re_lohi, rng 0)"%len(va_sub),flush=True)

# ---- r50_build.py helpers, verbatim ----
def load_sd(p):
    sd=torch.load(p,map_location="cpu")
    if isinstance(sd,dict) and "model_state_dict" in sd: sd=sd["model_state_dict"]
    if isinstance(sd,dict) and "state_fp16" in sd:
        ck=set(sd.get("complex_keys",[])); out={}
        for k,v in sd["state_fp16"].items():
            v=v.float()
            if k in ck: v=torch.view_as_complex(v)
            out[k]=v
        return out
    return sd
def lincomb(terms):
    ref=terms[0][1]; out={}
    for k,v in ref.items():
        if "num_batches_tracked" in k:
            out[k]=v.clone()            # BN counter stored as float32 -> COPY, never average
        elif v.is_complex() or v.dtype.is_floating_point:
            dt=torch.complex64 if v.is_complex() else torch.float32
            acc=None
            for c,sd in terms:
                t=c*sd[k].to(dt); acc=t if acc is None else acc+t
            out[k]=acc.to(v.dtype)
        else: out[k]=v.clone()
    return out
def maxdiff(a,b):
    m=0.0
    for k in a:
        if "num_batches_tracked" in k: continue     # counters, not weights
        x,y=a[k],b[k]
        if x.is_complex(): x=torch.view_as_real(x.to(torch.complex64)); y=torch.view_as_real(y.to(torch.complex64))
        m=max(m,float((x.float()-y.float()).abs().max()))
    return m
# ---- FITBASE helpers ----
def reldist(a,b):
    """||a-b||_2 / ||b||_2 over every float/complex tensor except BN counters (complex as real pairs)."""
    num=den=0.0
    for k,y in b.items():
        if "num_batches_tracked" in k or not (y.is_complex() or y.dtype.is_floating_point): continue
        x=a[k]
        if y.is_complex(): x=torch.view_as_real(x.to(torch.complex64)); y=torch.view_as_real(y.to(torch.complex64))
        x=x.double(); y=y.double(); num+=float(((x-y)**2).sum()); den+=float((y*y).sum())
    return math.sqrt(num/max(den,1e-300))
REFK={k:v.detach().clone() for k,v in model.state_dict().items()}      # the KIT weights (model untouched so far)
KITSD={k:v.detach().clone().cpu() for k,v in REFK.items()}
def ev(sd,label):
    model.load_state_dict({k:sd[k].to(dtype=v.dtype) for k,v in REFK.items()},strict=True)
    model.eval(); r=evaluate()
    print("  %-42s rel_l2 %.4f  tke %.4f  mvpe %.4f | cos0 %.4f rho %.4f"%(label,r[0],r[1],r[2],LAST_COS,LAST_RHO),flush=True)
    return dict(rel=float(r[0]),tke=float(r[1]),mvpe=float(r[2]),cos0=float(LAST_COS),rho=float(LAST_RHO))
def pr_cached(label,r):
    print("  %-42s rel_l2 %.4f  tke %.4f  mvpe %.4f | cos0 %.4f rho %.4f  (cached)"%(label,r["rel"],r["tke"],r["mvpe"],r["cos0"],r["rho"]),flush=True)
def deltas(r,ref):
    d=dict(d_rel=r["rel"]-ref["rel"],d_tke=r["tke"]-ref["tke"],d_mvpe=r["mvpe"]-ref["mvpe"],d_cos0=r["cos0"]-ref["cos0"])
    d["d_final"]=PRICE["rel"]*d["d_rel"]+PRICE["tke"]*d["d_tke"]+PRICE["mvpe"]*d["d_mvpe"]
    d["leak"]=LEAKSLOPE*d["d_cos0"]
    d["frac_cos0"]=d["leak"]/d["d_final"] if abs(d["d_final"])>1e-12 else float("nan")
    return d

# ---- leaves + cache ----
OLD=[f"{LH}/ft_all_w15_lr1_final.pth",f"{LH}/ft_all_w15_lr3_final.pth",
     f"{LH}/ft_all_w33_lr1_final.pth",f"{LH}/ft_all_w33_lr3_final.pth"]
LEAF=dict(w15_lr1=OLD[0],w15_lr3=OLD[1],w33_lr1=OLD[2],w33_lr3=OLD[3],sv3=f"{B}/train_es/ftaug_sv3_3e5_10.pth")
BASE4=["w15_lr1","w15_lr3","w33_lr1","w33_lr3"]
CFG=json.load(open(f"{FB}/fb_arms.json"))
for k,p in CFG["leaves"].items(): LEAF[k]=p.format(B=B,LH=LH,CK=CK)
_SD={}
def leaf(n):
    if n not in _SD: _SD[n]=load_sd(LEAF[n])
    return _SD[n]
CACHEF=f"{FB}/screen_cache.json"
try: CACHE=json.load(open(CACHEF))
except Exception: CACHE={}
def save_cache(): json.dump(CACHE,open(CACHEF,"w"),indent=1)
def fkey(names): return "|".join("%s:%d:%d"%(n,os.path.getsize(LEAF[n]),int(os.path.getmtime(LEAF[n]))) for n in names)

print("\n=== GATE 1: mean(4 original ft_all_*) must equal soup_v2.pth bit-exactly ===",flush=True)
s2_file=load_sd(f"{LH}/soup_v2.pth")
s2_calc=lincomb([(0.25,leaf(n)) for n in BASE4])
d1=maxdiff(s2_file,s2_calc); print("  max|d| = %.6e  ->  %s"%(d1,"PASS" if d1<1e-6 else "FAIL"),flush=True)
if d1>=1e-6: raise SystemExit("GATE 1 FAILED")
print("\n=== GATE 2: 0.5*soup_v2 + 0.5*sv3 must equal the BANKED backbone ===",flush=True)
bank_file=load_sd(f"{B}/train_es/r19_screen_best.pth")
bank_calc=lincomb([(0.5,s2_file),(0.5,leaf("sv3"))])
TOL2=1e-3   # r19_screen_best.json souped soup_v2_FP16.pth, we use full-precision soup_v2.pth
d2=maxdiff(bank_file,bank_calc); print("  max|d| = %.6e  (tol %.0e, fp16-quantisation of the soup_v2 ingredient)  ->  %s"%(d2,TOL2,"PASS" if d2<TOL2 else "FAIL"),flush=True)
if d2>=TOL2: raise SystemExit("GATE 2 FAILED - the decomposition is wrong; do NOT build.")
print("  ** the ENTIRE banked artifact is reproduced from its 5 leaf checkpoints.",flush=True)
del s2_calc

SLOTS=dict(a="w15_lr1",b="w15_lr3",c="w33_lr1",d="w33_lr3",s="sv3")
def build(sl):
    half=lincomb([(0.25,leaf(sl[x])) for x in "abcd"])           # same two-stage construction as r50
    return half, lincomb([(0.5,half),(0.5,leaf(sl["s"]))])
PRIMARY="BANKED"
print("\n=== REFERENCES ===",flush=True)
REF={}
REF["BANKED"]=ev(bank_file,"BANKED r19_screen_best (live 79.484440)")
old=CACHE.get("REF:BANKED")
if old is not None and max(abs(old[k]-REF["BANKED"][k]) for k in ("rel","tke","mvpe","cos0","rho"))>1e-6:
    print("  !! BANKED no longer reproduces its cached value -> cache discarded",flush=True); CACHE={}
CACHE["REF:BANKED"]=dict(REF["BANKED"]); save_cache()
for nm,key,sdv,lab in (("BANKCALC","REF:BANKCALC|"+fkey(BASE4+["sv3"]),bank_calc,"BANKCALC 0.5*soup_v2+0.5*sv3 (fp32 leaves)"),
                       ("soup_v2","REF:soup_v2|"+fkey(BASE4),s2_file,"soup_v2 (banked soup half)")):
    if key in CACHE: REF[nm]=dict(CACHE[key]); pr_cached(lab,REF[nm])
    else: REF[nm]=ev(sdv,lab); CACHE[key]=dict(REF[nm]); save_cache()

print("\n=== MEMBERS (single leaf checkpoints; cos0 ingredient ceiling %.4f) ==="%COSGATE,flush=True)
MEM={}
for n,p in LEAF.items():
    if not os.path.exists(p): continue
    key="MEM|"+fkey([n]); cp=CFG["counterpart"].get(n)
    if key in CACHE:
        MEM[n]=dict(CACHE[key]); pr_cached("member %s"%n,MEM[n])
    else:
        MEM[n]=ev(leaf(n),"member %s"%n)
        MEM[n]["dist_kit"]=reldist(leaf(n),KITSD)
        if cp: MEM[n]["counterpart"]=cp; MEM[n]["dist_counterpart"]=reldist(leaf(n),leaf(cp))
        CACHE[key]=dict(MEM[n]); save_cache()
    msg="      |w-w_kit|/|w_kit| = %.4e"%MEM[n]["dist_kit"]
    if cp: msg+="   |w-w_%s|/|w_%s| = %.4e"%(cp,cp,MEM[n]["dist_counterpart"])
    print(msg,flush=True)
key="YARD|"+fkey(BASE4)
if key not in CACHE:
    CACHE[key]={"w15_lr3 vs w15_lr1":reldist(leaf("w15_lr3"),leaf("w15_lr1")),
                "w33_lr3 vs w33_lr1":reldist(leaf("w33_lr3"),leaf("w33_lr1")),
                "w15_lr3 vs w33_lr3":reldist(leaf("w15_lr3"),leaf("w33_lr3"))}; save_cache()
YARD=CACHE[key]
for k,v in YARD.items(): print("  yardstick |d|/|w| %-22s %.4e"%(k,v),flush=True)

def verdict(lab,d,ingr):
    if lab.startswith("C"): return "CONTROL (replicate: noise floor)"
    if d["d_final"]<=0: return "DEAD (local-negative)"
    if d["d_final"]>=0.20: return "REJECT (magnitude law: dfinal >= 0.20)"
    if d["leak"]>0 and d["leak"]>=d["d_final"]:
        return "FLAG leakage-signature (%.0f%% of dfinal explained by the cos0 rise) - not built"%(100*d["leak"]/d["d_final"])
    hi=["%s %.4f"%(k,c) for k,c in ingr.items() if c>COSGATE]
    if hi: return "REJECT (cos0 ingredient gate > %.4f: %s)"%(COSGATE,", ".join(hi))
    return "ADVANCE"

print("\n=== ARMS: backbone = 0.5*mean(4 ft_all slots) + 0.5*(sv3 slot); deltas vs %s ==="%PRIMARY,flush=True)
ARM={}
for lab,ov in CFG["arms"].items():
    sl=dict(SLOTS); sl.update(ov)
    names=[sl[x] for x in "abcds"]
    miss=[n for n in names if not os.path.exists(LEAF[n])]
    if miss: print("  %-42s SKIP (missing: %s)"%(lab,", ".join(miss)),flush=True); continue
    ch_soup=any(sl[x]!=SLOTS[x] for x in "abcd"); ch_sv3=(sl["s"]!=SLOTS["s"])
    ka="ARM|"+fkey(names); kh="HALF|"+fkey(names[:4])
    ingr={}
    if ka in CACHE and ((not ch_soup) or kh in CACHE):
        r=dict(CACHE[ka]); pr_cached(lab,r)
        if ch_soup: ingr["soup_half"]=CACHE[kh]["cos0"]; pr_cached("   ingredient: soup half",CACHE[kh])
    else:
        half,bb=build(sl)
        r=ev(bb,lab); r["dist_vs_bankcalc"]=reldist(bb,bank_calc); CACHE[ka]=dict(r)
        if ch_soup:
            hr=ev(half,"   ingredient: soup half"); CACHE[kh]=dict(hr); ingr["soup_half"]=hr["cos0"]
        save_cache(); del half,bb
    if ch_sv3: ingr["sv3_half"]=MEM[sl["s"]]["cos0"]
    newm={sl[x]:MEM[sl[x]]["cos0"] for x in "abcds" if sl[x]!=SLOTS[x]}
    d=deltas(r,REF[PRIMARY]); dc=deltas(r,REF["BANKCALC"])
    ARM[lab]=dict(slots=sl,score=r,d=d,d_vs_bankcalc=dc,ingredient_cos0=ingr,new_member_cos0=newm,
                  dist_vs_bankcalc=r.get("dist_vs_bankcalc"))
    ARM[lab]["verdict"]=verdict(lab,d,ingr)
    fr=("%.0f%%"%(100*d["frac_cos0"])) if d["frac_cos0"]==d["frac_cos0"] else "n/a"
    print("      d_rel %+.4f d_tke %+.4f d_mvpe %+.4f d_cos0 %+.4f | dfinal %+.4f (vs BANKCALC %+.4f) | 9.5146*dcos0 %+.4f = %s of dfinal"
          %(d["d_rel"],d["d_tke"],d["d_mvpe"],d["d_cos0"],d["d_final"],dc["d_final"],d["leak"],fr),flush=True)
    print("      |bb-bankcalc|/|bankcalc| %.4e | ingredient cos0 %s | new-member cos0 %s"
          %(ARM[lab]["dist_vs_bankcalc"],{k:round(v,4) for k,v in ingr.items()},{k:round(v,4) for k,v in newm.items()}),flush=True)
    print("      VERDICT: %s"%ARM[lab]["verdict"],flush=True)

R0=REF[PRIMARY]
print("\n=== SUMMARY vs %s: rel %.4f tke %.4f mvpe %.4f cos0 %.4f ==="%(PRIMARY,R0["rel"],R0["tke"],R0["mvpe"],R0["cos0"]),flush=True)
print("  %-28s %8s %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s %6s | %s"%("arm","rel","tke","mvpe","cos0","rho","d_rel","d_tke","d_mvpe","d_cos0","dfinal","9.51dcos","frac","verdict"))
for lab,A_ in ARM.items():
    s=A_["score"]; d=A_["d"]; fr=("%5.0f%%"%(100*d["frac_cos0"])) if d["frac_cos0"]==d["frac_cos0"] else "   n/a"
    print("  %-28s %8.4f %8.4f %8.4f %8.4f %8.4f | %+8.4f %+8.4f %+8.4f %+8.4f | %+8.4f %+8.4f %6s | %s"
          %(lab,s["rel"],s["tke"],s["mvpe"],s["cos0"],s["rho"],d["d_rel"],d["d_tke"],d["d_mvpe"],d["d_cos0"],d["d_final"],d["leak"],fr,A_["verdict"]))
cand=[A_ for l,A_ in ARM.items() if not l.startswith("C")]
if len(cand)>=3:
    x=np.array([A_["d"]["d_cos0"] for A_ in cand]); y=np.array([A_["d"]["d_final"] for A_ in cand])
    rr=float(np.corrcoef(x,y)[0,1]); sl_=float(np.polyfit(x,y,1)[0])
    print("  corr(dfinal, dcos0) over %d candidate arms = %+.3f (r2 %.3f), OLS slope %.3f  [sec128.4.3 check]"%(len(cand),rr,rr*rr,sl_),flush=True)
OUTJ=dict(ref=REF,members=MEM,arms=ARM,yard=YARD,primary=PRIMARY,gates=dict(gate1=d1,gate2=d2,kit_base=list(base)),
          when=_time.strftime("%Y-%m-%d %H:%M:%S"))
json.dump(OUTJ,open(f"{FB}/screen_results.json","w"),indent=1,default=float)
shutil.copy(f"{FB}/screen_results.json",f"{FB}/screen_results_%s.json"%_time.strftime("%m%d_%H%M%S"))

if MODE=="build":
    lab=os.environ["FB_ARM"]; ztag=os.environ["FB_ZIPTAG"]
    if lab not in ARM: raise SystemExit("REFUSE: arm %s was not screened in this run"%lab)
    if ARM[lab]["verdict"]!="ADVANCE": raise SystemExit("REFUSE to build %s: %s"%(lab,ARM[lab]["verdict"]))
    DSTN=f"FITBASE_{ztag}"; DST=f"{B}/submissions/submission_{DSTN}.zip"
    if os.path.exists(DST): raise SystemExit("REFUSE: %s already exists - never overwrite a zip"%DST)
    _,new=build(ARM[lab]["slots"])
    packed,ck={},[]
    for k,v in new.items():
        if v.is_complex(): ck.append(k); packed[k]=torch.view_as_real(v.to(torch.complex64)).half()
        elif v.dtype.is_floating_point: packed[k]=v.half()
        else: packed[k]=v
    bp=f"{FB}/{DSTN}_fp16.pth"; torch.save({"state_fp16":packed,"complex_keys":ck},bp)
    raw=torch.load(bp,map_location="cpu"); cks=set(raw["complex_keys"]); mx=0.0
    for k,v in raw["state_fp16"].items():
        if "num_batches_tracked" in k: continue
        w=torch.view_as_complex(v.float()) if k in cks else v.float()
        mx=max(mx,(w-new[k].to(w.dtype)).abs().max().item())
    print("\npacked -> %s B (%d complex) | fp16 round-trip max|d| = %.3e"%(format(os.path.getsize(bp),","),len(ck),mx),flush=True)
    r16=ev(load_sd(bp),"packed fp16 CANDIDATE")
    zs=zipfile.ZipFile(f"{B}/submissions/submission_SCREEN.zip")
    sbp=f"{FB}/_SCREEN_backbone_fp16.pth"; open(sbp,"wb").write(zs.read("sim_real_fno_fp16.pth"))
    try:
        b16=ev(load_sd(sbp),"packed fp16 BANKED (SCREEN zip)"); d16=deltas(r16,b16)
        print("  fp16-vs-fp16: d_rel %+.4f d_tke %+.4f d_mvpe %+.4f d_cos0 %+.4f | dfinal %+.4f"
              %(d16["d_rel"],d16["d_tke"],d16["d_mvpe"],d16["d_cos0"],d16["d_final"]),flush=True)
    except Exception as e: print("  (could not evaluate the SCREEN backbone: %r)"%(e,),flush=True)
    SRC=f"{B}/submissions/submission_SV2.zip"
    zin=zipfile.ZipFile(SRC); nb=open(bp,"rb").read()
    zo=zipfile.ZipFile(DST,"w",zipfile.ZIP_DEFLATED); n=0
    for it in zin.infolist():
        dd=nb if it.filename=="sim_real_fno_fp16.pth" else zin.read(it.filename)
        if it.filename=="sim_real_fno_fp16.pth": n+=1
        zi=zipfile.ZipInfo(it.filename,date_time=it.date_time)
        zi.compress_type=it.compress_type; zi.external_attr=it.external_attr
        zo.writestr(zi,dd)
    zo.close(); assert n==1
    zc=zipfile.ZipFile(DST); ext=sum(i.file_size for i in zc.infolist())
    same=all(zc.read(x)==zin.read(x) for x in zin.namelist() if x!="sim_real_fno_fp16.pth")
    same_screen=all(zc.read(x)==zs.read(x) for x in zs.namelist() if x!="sim_real_fno_fp16.pth")
    npyc=sum(1 for x in zc.namelist() if x.endswith(".pyc"))
    ok=os.system("unzip -tq "+DST+" >/dev/null")==0
    print("\n"+DST,flush=True)
    print("  other entries byte-identical to SV2: %s | to SCREEN: %s | entry list identical: %s | .pyc %d"
          %(same,same_screen,zc.namelist()==zin.namelist(),npyc),flush=True)
    print("  extracted %s B = %.2f%% of cap (< 268,435,456: %s) | unzip -t %s"%(format(ext,","),100*ext/268435456,ext<268435456,ok),flush=True)
    print("  backbone md5 %s"%hashlib.md5(zc.read("sim_real_fno_fp16.pth")).hexdigest(),flush=True)
    print("  zip md5      %s"%hashlib.md5(open(DST,"rb").read()).hexdigest(),flush=True)
    if not (same and same_screen and npyc==0 and ok and ext<268435456 and zc.namelist()==zin.namelist()):
        raise SystemExit("BUILD VERIFICATION FAILED")
print("\n[done] fitbase %s  %.0fs"%(MODE,_time.time()-T0),flush=True)
