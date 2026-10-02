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
ap.add_argument("--init",default="")
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
else: raise SystemExit("unknown split")
USE_FULL = ("phase" in a.aug)
if USE_FULL: XF=np.load(f"{LH}/tr_full64.npy",mmap_mode="r")
X32=np.load(f"{LH}/tr_frames.npy",mmap_mode="r")
starts_tr=[]; starts_va=[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39):
        (starts_va if i in VT else starts_tr).append(t0)
starts_tr=np.array(starts_tr); starts_va=np.array(starts_va)
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
def fwd(x): return model((x-MI)/SI)*ST+MT
def rel_l2(p,t):
    p=p[...,:C].reshape(p.shape[0],-1); t=t[...,:C].reshape(t.shape[0],-1)
    return (torch.linalg.norm(p-t,dim=1)/torch.linalg.norm(t,dim=1).clamp(min=1e-8)).mean()
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
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
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
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
    if step%a.evalevery==0 or step==a.steps:
        r=evaluate(); d=dacc(r); flag=""
        if d>best:
            best=d; torch.save(model.state_dict(),f"{B}/train_es/ftaug_{a.tag}.pth"); flag=" *"
        print("step %6d loss %7.4f | rel_l2 %.4f (%+.4f) tke %.4f (%+.4f) mvpe %.4f (%+.4f) | d_acc %+.4f | %.0fs%s"
              %(step,float(loss.detach()),r[0],r[0]-base[0],r[1],r[1]-base[1],r[2],r[2]-base[2],d,time.time()-t0,flag),flush=True)
        json.dump({"base":base,"best_dacc":best,"args":vars(a)},open(f"{B}/train_es/ftaug_{a.tag}.json","w"),default=float,indent=1)
print("[done] best d_acc %+.4f  %.0fs"%(best,time.time()-t0))
