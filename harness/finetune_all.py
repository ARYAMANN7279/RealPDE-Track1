import argparse, json, os, sys, time
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline

ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=3e-5); ap.add_argument("--gpu",type=int,default=1)
ap.add_argument("--steps",type=int,default=6000); ap.add_argument("--bs",type=int,default=8)
ap.add_argument("--wtke",type=float,default=0.33); ap.add_argument("--tag",default="all")
a=ap.parse_args()

DEV=f"cuda:{a.gpu}"; C=2; IN=20
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]

X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
ntraj=len(lens)

starts_tr=[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39):
        starts_tr.append(t0)
starts_tr=np.array(starts_tr)
print("train windows %d (ALL %d trajectories)"%(len(starts_tr),ntraj),flush=True)

rng=np.random.default_rng(0)

def batch(idx):
    w=np.stack([np.asarray(X[s:s+40]) for s in idx])
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

opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-6)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.steps,pct_start=0.1)
t0=time.time(); model.train()
for step in range(1,a.steps+1):
    idx=rng.choice(starts_tr,size=a.bs,replace=False)
    x,y=batch(idx)
    p=fwd(x)
    loss=rel_l2(p,y)+a.wtke*tke_l2(p,y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sched.step()
    if step%1000==0:
        print("step %5d loss %.4f | %.0fs"%(step,float(loss),time.time()-t0),flush=True)

torch.save(model.state_dict(),f"{B}/local_harness/ft_{a.tag}_final.pth")
print("Saved ft_%s_final.pth" % a.tag)
