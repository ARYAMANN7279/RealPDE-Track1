"""M55: fine-tune the SIM-ONLY checkpoint on 55 trajectories, holding out 26.

Why this exists: all 82 released trajectories are in-sample for the official
sim_real checkpoint, so its out-of-sample error can never be measured on released
data. sim_fno.pth has never seen real data, so a model fine-tuned from it has 26
genuinely unseen trajectories -- the first honest error signal we can get.

Rules-compliant: official Drive checkpoint + released training data only.
"""
import json, os, sys, time
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
DEV="cuda"; C=2; IN=20
torch.manual_seed(1234); np.random.seed(1234)
MI=torch.tensor([0.154960856,-0.000513992854,0.0]).to(DEV)
SI=torch.tensor([0.0968056545,0.015960684,1.0]).to(DEV)
MT=torch.tensor([0.154962569,-0.000517793698,0.0]).to(DEV)
ST=torch.tensor([0.0968104079,0.0159636438,1.0]).to(DEV)
F=np.load(f"{B}/train_work/frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/train_work/meta.json")); off=meta["off"]; lens=meta["lens"]; names=meta["names"]
NT=len(lens)
rng=np.random.default_rng(1234); order=rng.permutation(NT)
TRAJ_TR=set(order[:55].tolist()); TRAJ_VA=set(order[55:].tolist())
print("trajectories: %d train / %d HELD-OUT (never seen by M55)"%(len(TRAJ_TR),len(TRAJ_VA)),flush=True)
HOR=40
def windows(tset,stride=20):
    w=[]
    for i in tset:
        for t0 in range(off[i],off[i]+lens[i]-HOR+1,stride): w.append(t0)
    return np.array(w)
Wtr=windows(TRAJ_TR,20); Wva=windows(TRAJ_VA,40)
print("windows: train %d / val %d"%(len(Wtr),len(Wva)),flush=True)
def batch(idx):
    a=np.stack([np.concatenate([np.asarray(F[s:s+HOR]),
        np.zeros((HOR,32,64,1),np.float32)],-1) for s in idx]).astype(np.float32)
    return torch.from_numpy(a[:,:IN]).to(DEV), torch.from_numpy(a[:,IN:]).to(DEV)
model,_=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)
ck=torch.load(f"{B}/data/comp_real/sim_fno.pth",map_location=DEV,weights_only=False)
model.load_state_dict(ck.get("model_state_dict",ck)); model=model.to(DEV).train()
print("loaded SIM-ONLY weights (iteration %s)"%ck.get("iteration"),flush=True)
def fwd(x): return model((x-MI)/SI)*ST+MT
def rel(a,b):
    n=(a-b).flatten(1).norm(dim=1); d=b.flatten(1).norm(dim=1).clamp(min=1e-8)
    return (n/d).mean()
def tkeloss(p,t):
    def ke(z):
        u,v=z[...,0],z[...,1]
        return 0.5*(((u-u.mean(1,keepdim=True))**2).mean(1)+((v-v.mean(1,keepdim=True))**2).mean(1))
    return rel(ke(p).unsqueeze(-1),ke(t).unsqueeze(-1))
opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
STEPS=6000; BS=8; WTKE=0.12
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS,eta_min=5e-6)
@torch.no_grad()
def evaluate():
    model.eval(); dm=[];tk=[];mv=[]
    for i in range(0,len(Wva),16):
        x,y=batch(Wva[i:i+16]); p=fwd(x)
        pn,yn=p.cpu().numpy(),y.cpu().numpy()
        dm.append(S.rel_l2_per_sample(pn,yn,C)); tk.append(S.tke_rel_l2_per_sample(pn,yn,C))
        mv.append(S.mvpe_rel_l2_per_sample(pn,yn))
    model.train()
    return (S.score_error(float(np.concatenate(dm).mean())),
            S.score_error(float(np.concatenate(tk).mean())),
            S.score_error(float(np.concatenate(mv).mean())))
print("HELD-OUT before fine-tuning: rel_l2 %.2f tke %.2f mvpe %.2f"%evaluate(),flush=True)
best=-1; t0=time.time()
for step in range(1,STEPS+1):
    idx=Wtr[np.random.randint(0,len(Wtr),BS)]
    x,y=batch(idx); p=fwd(x)
    loss=rel(p[...,:C],y[...,:C])+WTKE*tkeloss(p[...,:C],y[...,:C])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
    if step%1000==0:
        a=evaluate(); sc=0.306*a[0]+0.163*a[1]+0.218*a[2]
        star=""
        if sc>best:
            best=sc; torch.save(model.state_dict(),f"{B}/local_harness/m55.pth"); star=" *"
        print("  step %5d loss %.4f | HELD-OUT rel_l2 %.2f tke %.2f mvpe %.2f | %.1f min%s"%(
            step,float(loss),a[0],a[1],a[2],(time.time()-t0)/60,star),flush=True)
print("saved m55.pth")
