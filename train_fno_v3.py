import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))

import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline

ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=1e-5)
ap.add_argument("--gpu",type=int,default=2)
ap.add_argument("--steps",type=int,default=6000)
ap.add_argument("--bs",type=int,default=16)
ap.add_argument("--wtke",type=float,default=0.33)
ap.add_argument("--tag",default="soup_v3")
a=ap.parse_args()

DEV=f"cuda:{a.gpu}"; C=2; IN=20; OUT=20
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]

X_npy=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json"))
off=meta["off"]; lens=meta["lens"]; names=meta["names"]
ntraj=len(lens)

# re_lohi indices
RE=np.array([int(names[i].split("_")[0]) for i in range(ntraj)])
is_lohi = np.isin(RE,[3750,5025,25425,26700])

starts_tr=[];starts_va=[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39):
        if is_lohi[i]:
            starts_va.append(t0)
        else:
            starts_tr.append(t0)

starts_tr=np.array(starts_tr); starts_va=np.array(starts_va)
print(f"train windows {len(starts_tr)} | val windows (re_lohi) {len(starts_va)}", flush=True)

rng=np.random.default_rng(0)
va_sub=rng.choice(starts_va,size=min(1200,len(starts_va)),replace=False)

def batch(idx):
    w=np.stack([np.asarray(X_npy[s:s+40]) for s in idx])
    z=np.zeros(w.shape[:-1]+(1,),np.float32)
    w=np.concatenate([w,z],-1)
    return torch.from_numpy(w[:,:IN]).to(DEV), torch.from_numpy(w[:,IN:]).to(DEV)

# Loading baseline model to init
model,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)

# Let's see what model we should load.
if os.path.exists(f"{B}/train_es/soup_fno_fp16.pth"):
    # Wait, it's fp16, we might need to cast.
    sd = torch.load(f"{B}/train_es/soup_fno_fp16.pth", map_location=DEV)
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    print("Loaded soup_fno_fp16.pth")
else:
    print("Loaded baseline sim_real_fno.pth")

model=model.to(DEV)

def fwd_fno(x): return model((x-MI)/SI)*ST+MT

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
    model.eval(); P=[]; T=[]
    for i in range(0,len(va_sub),16):
        x,y=batch(va_sub[i:i+16])
        p = fwd_fno(x)
        P.append(p.cpu().numpy()); T.append(y.cpu().numpy())
    model.train()
    P=np.concatenate(P,0).astype(np.float32); T=np.concatenate(T,0).astype(np.float32)
    dm=S.rel_l2_per_sample(P,T,C); tk=S.tke_rel_l2_per_sample(P,T,C); mv=S.mvpe_rel_l2_per_sample(P,T)
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))

def proj(r): return 0.306*r[0]+0.163*r[1]+0.218*r[2]

base=evaluate(); b0=proj(base)
print("baseline val: rel_l2 %.2f tke %.2f mvpe %.2f"%base,flush=True)

params = list(model.parameters())
opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=1e-6)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.steps,pct_start=0.1)

best=(b0,base,0); t0=time.time(); model.train()

for step in range(1,a.steps+1):
    idx=rng.choice(starts_tr,size=a.bs,replace=False)
    x,y=batch(idx)
    
    p = fwd_fno(x)
    
    loss_fno = rel_l2(p, y) + a.wtke * tke_l2(p, y)
    loss = loss_fno
    
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    
    if step % 500 == 0 or step == a.steps:
        r = evaluate(); d = proj(r) - b0
        flag = ""
        # We want to maximize the score, meaning d > best[0] - b0 (since proj is a score where higher is better? Wait, let's check S.score_error)
        if proj(r) > best[0]:
            best = (proj(r), r, step)
            torch.save(model.state_dict(), f"{B}/train_es/{a.tag}.pth")
            flag = " *saved"
        print("step %5d loss %.4f | val rel_l2 %.2f tke %.2f mvpe %.2f | proj %.3f | %.0fs%s" % (
            step, float(loss), r[0], r[1], r[2], proj(r), time.time() - t0, flag), flush=True)
        if r[1] > 80.0:
            print("Target TKE > 80 achieved!", flush=True)
            # if we wanted we could stop, but we just train till end.

print("BEST at step %d: rel_l2 %.2f tke %.2f mvpe %.2f  (proj %.3f)" % (
    best[2], best[1][0], best[1][1], best[1][2], best[0]), flush=True)

