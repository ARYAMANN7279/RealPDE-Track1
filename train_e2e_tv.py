import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
LH=f"{B}/local_harness"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))

import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline

ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=3e-5)
ap.add_argument("--gpu",type=int,default=0)
ap.add_argument("--steps",type=int,default=2000)
ap.add_argument("--bs",type=int,default=8)
ap.add_argument("--wtke",type=float,default=0.33)
ap.add_argument("--wtv",type=float,default=10.0)
ap.add_argument("--u_width",type=int,default=64)
ap.add_argument("--tag",default="e2e_tv")
a=ap.parse_args()

DEV=f"cuda:{a.gpu}"; C=2; IN=20; OUT=20
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]

X_npy=np.load(f"{LH}/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{LH}/tr_meta.json"))
off=meta["off"]; lens=meta["lens"]; names=meta["names"]
ntraj=len(lens); vidx=set(range(0,ntraj,5))
starts_tr=[];starts_va=[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39):
        (starts_va if i in vidx else starts_tr).append(t0)
starts_tr=np.array(starts_tr); starts_va=np.array(starts_va)
print(f"train windows {len(starts_tr)} | val windows {len(starts_va)}", flush=True)

rng=np.random.default_rng(0)
va_sub=rng.choice(starts_va,size=min(600,len(starts_va)),replace=False)

def batch(idx):
    w=np.stack([np.asarray(X_npy[s:s+40]) for s in idx])
    z=np.zeros(w.shape[:-1]+(1,),np.float32)
    w=np.concatenate([w,z],-1)
    return torch.from_numpy(w[:,:IN]).to(DEV), torch.from_numpy(w[:,IN:]).to(DEV)

model,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
try:
    soup_sd = torch.load(f"{B}/train_es/soup_v2.pth", map_location=DEV)
    model.load_state_dict(soup_sd.get("model_state_dict", soup_sd))
    print("Loaded soup_v2.pth")
except:
    print("Loaded baseline sim_real_fno.pth")
model=model.to(DEV)

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return F.gelu(s.n2(s.c2(x)))

class UNet(nn.Module):
    def __init__(s,ci,co,w):
        super().__init__()
        s.e1=Blk(ci,w); s.e2=Blk(w,2*w); s.e3=Blk(2*w,4*w); s.b=Blk(4*w,4*w)
        s.d3=Blk(8*w,2*w); s.d2=Blk(4*w,w); s.d1=Blk(2*w,w); s.out=nn.Conv2d(w,co,1)
        s.pool=nn.AvgPool2d(2); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(s.pool(e1)); e3=s.e3(s.pool(e2)); b=s.b(s.pool(e3))
        u=F.interpolate(b,size=e3.shape[-2:],mode="bilinear",align_corners=False); d3=s.d3(torch.cat([u,e3],1))
        u=F.interpolate(d3,size=e2.shape[-2:],mode="bilinear",align_corners=False); d2=s.d2(torch.cat([u,e2],1))
        u=F.interpolate(d2,size=e1.shape[-2:],mode="bilinear",align_corners=False); d1=s.d1(torch.cat([u,e1],1))
        return s.out(d1)

unet = UNet(80,80,a.u_width).to(DEV)
print(f"UNet initialized with width {a.u_width}")

def fwd_fno(x): return model((x-MI)/SI)*ST+MT

def format_unet_in(x, p):
    x2 = x[..., :2].permute(0, 1, 4, 2, 3).reshape(x.shape[0], -1, 32, 64)
    p2 = p[..., :2].permute(0, 1, 4, 2, 3).reshape(p.shape[0], -1, 32, 64)
    return torch.cat([x2, p2], dim=1)

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
    model.eval(); unet.eval(); P=[]; T=[]; W=[]
    for i in range(0,len(va_sub),16):
        x,y=batch(va_sub[i:i+16])
        p_base = fwd_fno(x)
        u_in = format_unet_in(x, p_base)
        o = unet(u_in)
        c = o[:, :40].view(o.shape[0], 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        c_pad = torch.zeros(c.shape[:-1] + (1,), device=DEV)
        c_val = torch.cat([c, c_pad], dim=-1)
        p_corr = p_base + c_val
        P.append(p_corr.cpu().numpy()); T.append(y.cpu().numpy())
    model.train(); unet.train()
    P=np.concatenate(P,0).astype(np.float32); T=np.concatenate(T,0).astype(np.float32)
    dm=S.rel_l2_per_sample(P,T,C); tk=S.tke_rel_l2_per_sample(P,T,C); mv=S.mvpe_rel_l2_per_sample(P,T)
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))

def proj(r): return 0.306*r[0]+0.163*r[1]+0.218*r[2]

base=evaluate(); b0=proj(base)
print("baseline val (with untrained unet): rel_l2 %.2f tke %.2f mvpe %.2f"%base,flush=True)

params = list(model.parameters()) + list(unet.parameters())
opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=1e-6)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.steps,pct_start=0.1)

best=(b0,base,0); t0=time.time(); model.train(); unet.train()

for step in range(1,a.steps+1):
    idx=rng.choice(starts_tr,size=a.bs,replace=False)
    x,y=batch(idx)
    
    p_base = fwd_fno(x)
    u_in = format_unet_in(x, p_base)
    o = unet(u_in)
    
    c_out = o[:, :40]
    w_out = o[:, 40:]
    
    c_res = c_out.view(c_out.shape[0], 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
    c_pad = torch.zeros(c_res.shape[:-1] + (1,), device=DEV)
    c_val = torch.cat([c_res, c_pad], dim=-1)
    
    p_corr = p_base + c_val
    
    # Standard metrics
    l_fno = rel_l2(p_corr, y) + a.wtke * tke_l2(p_corr, y)
    
    # TV Penalty on c_val
    # c_val shape: (B, 20, 32, 64, 3)
    dx = c_val[:, :, 1:, :, :] - c_val[:, :, :-1, :, :]
    dy = c_val[:, :, :, 1:, :] - c_val[:, :, :, :-1, :]
    l_tv = a.wtv * ((dx**2).mean() + (dy**2).mean())
    
    loss_fno = l_fno + l_tv
    
    # Width training (log-abs err)
    r_flat = y[..., :2].permute(0, 1, 4, 2, 3).reshape(y.shape[0], -1, 32, 64)
    p_corr_flat = p_corr[..., :2].permute(0, 1, 4, 2, 3).reshape(p_corr.shape[0], -1, 32, 64)
    m = (r_flat != 0).float()
    
    tgt = (torch.log((r_flat - p_corr_flat.detach()).abs() + 1e-6) - -6.0) / 1.5
    l_w = ((w_out - tgt).abs() * m).sum() / m.sum().clamp(min=1.0)
    
    loss = loss_fno + 0.1 * l_w
    
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    
    if step % 200 == 0 or step == a.steps:
        r = evaluate(); d = proj(r) - b0
        flag = ""
        if d > best[0] - b0:
            best = (proj(r), r, step)
            torch.save({"fno": model.state_dict(), "unet": unet.state_dict()}, f"{B}/train_es/{a.tag}_best.pth")
            flag = " *saved"
        print("step %5d loss %.4f (fno %.4f tv %.4f w %.4f) | val rel_l2 %.2f tke %.2f mvpe %.2f | d_acc %+.3f | %.0fs%s" % (
            step, float(loss), float(l_fno), float(l_tv), float(l_w), r[0], r[1], r[2], d, time.time() - t0, flag), flush=True)

print("BEST at step %d: rel_l2 %.2f tke %.2f mvpe %.2f  (d_acc %+.3f)" % (
    best[2], best[1][0], best[1][1], best[1][2], best[0] - b0), flush=True)
