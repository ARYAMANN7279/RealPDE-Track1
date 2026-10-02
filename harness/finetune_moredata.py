"""Fine-tune the kit FNO on the CORRECT real data (train_real).

Memory 6 recorded that fine-tuning degrades the model, but that was measured
through the broken input path (out-of-distribution inputs => meaningless
gradients). This is the first attempt on correct data.

Loss weights follow the score's own sensitivity:
  d(final)/d(err_rel_l2) = 0.306 * -50/(1+0.5*0.1238)^2 = -13.6
  d(final)/d(err_tke)    = 0.163 * -50/(1+0.5*0.702)^2  = -4.47
  => ratio ~3:1, so L = err_rel_l2 + 0.33 * err_tke
Honest split: 65 train trajectories / 16 val, disjoint. Best-on-val is kept.
"""
import argparse, json, os, sys, time
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=3e-5); ap.add_argument("--gpu",type=int,default=1)
ap.add_argument("--steps",type=int,default=6000); ap.add_argument("--bs",type=int,default=8)
ap.add_argument("--wtke",type=float,default=0.33); ap.add_argument("--tag",default="a")
ap.add_argument("--holdout_stride",type=int,default=5)
a=ap.parse_args()
DEV=f"cuda:{a.gpu}"; C=2; IN=20; OUT=20
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]; names=meta["names"]
ntraj=len(lens); vidx=set(range(0,ntraj,a.holdout_stride))  # configurable holdout for more-data experiments
starts_tr=[];starts_va=[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39):
        (starts_va if i in vidx else starts_tr).append(t0)
starts_tr=np.array(starts_tr); starts_va=np.array(starts_va)
print("train windows %d | val windows %d (%d val trajectories)"%(len(starts_tr),len(starts_va),len(vidx)),flush=True)
rng=np.random.default_rng(0)
va_sub=rng.choice(starts_va,size=min(600,len(starts_va)),replace=False)

def batch(idx):
    w=np.stack([np.asarray(X[s:s+40]) for s in idx])       # (B,40,32,64,2)
    z=np.zeros(w.shape[:-1]+(1,),np.float32)
    w=np.concatenate([w,z],-1)
    return torch.from_numpy(w[:,:IN]).to(DEV), torch.from_numpy(w[:,IN:]).to(DEV)

model,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
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
        x,y=batch(va_sub[i:i+16]); P.append(fwd(x).cpu().numpy()); T.append(y.cpu().numpy())
    model.train()
    P=np.concatenate(P,0).astype(np.float32); T=np.concatenate(T,0).astype(np.float32)
    dm=S.rel_l2_per_sample(P,T,C); tk=S.tke_rel_l2_per_sample(P,T,C); mv=S.mvpe_rel_l2_per_sample(P,T)
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
n_=lambda x:x/(0.5+x)
def proj(r):
    # delta final vs the pre-training baseline, using the solved score weights
    return 0.306*r[0]+0.163*r[1]+0.218*r[2]
base=evaluate(); b0=proj(base)
print("baseline val: rel_l2 %.2f tke %.2f mvpe %.2f"%base,flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-6)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.steps,pct_start=0.1)
best=(b0,base,0); t0=time.time(); model.train()
for step in range(1,a.steps+1):
    idx=rng.choice(starts_tr,size=a.bs,replace=False)
    x,y=batch(idx)
    p=fwd(x)
    loss=rel_l2(p,y)+a.wtke*tke_l2(p,y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sched.step()
    if step%500==0 or step==a.steps:
        r=evaluate(); d=proj(r)-b0
        flag=""
        if d>best[0]-b0:
            best=(proj(r),r,step)
            torch.save(model.state_dict(),f"{B}/local_harness/ft_{a.tag}_best.pth"); flag=" *saved"
        print("step %5d loss %.4f | val rel_l2 %.2f tke %.2f mvpe %.2f | d_acc %+.3f | %.0fs%s"%(
            step,float(loss),r[0],r[1],r[2],d,time.time()-t0,flag),flush=True)
print("BEST at step %d: rel_l2 %.2f tke %.2f mvpe %.2f  (d_acc %+.3f)"%(
    best[2],best[1][0],best[1][1],best[1][2],best[0]-b0),flush=True)
json.dump(dict(lr=a.lr,wtke=a.wtke,base=base,best=best[1],step=best[2],d_acc=best[0]-b0),
          open(f"{B}/local_harness/ft_{a.tag}_result.json","w"),indent=1)
