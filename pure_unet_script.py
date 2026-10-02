import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S)

ap=argparse.ArgumentParser()
ap.add_argument("--lr",type=float,default=1e-5); ap.add_argument("--gpu",type=int,default=0)
ap.add_argument("--steps",type=int,default=8000); ap.add_argument("--bs",type=int,default=16)
ap.add_argument("--wtke",type=float,default=0.08); ap.add_argument("--tag",default="pure_unet")
ap.add_argument("--split",default="re_lohi"); ap.add_argument("--aug",default="phase")
ap.add_argument("--wd",type=float,default=1e-6)
ap.add_argument("--wtv",type=float,default=0.1) # Added wtv
ap.add_argument("--u_width",type=int,default=64)
ap.add_argument("--evalevery",type=int,default=1000)
a=ap.parse_args(); DEV=f"cuda:{a.gpu}"; C=2
torch.manual_seed(1234); np.random.seed(1234)

# Data loading exactly as ftaug.py
meta=json.load(open(f"{LH}/tr_meta.json")); off,lens,names=meta["off"],meta["lens"],meta["names"]
ntraj=len(lens)
RE=np.array([int(n.split("_")[0]) for n in names])
AOA=np.array([n.split("_")[1].split(".")[0] for n in names])

# Evaluate on re_lohi
VT=set(np.where(np.isin(RE,[3750,5025,25425,26700]))[0].tolist())

USE_FULL = ("phase" in a.aug)
if USE_FULL: XF=np.load(f"{LH}/tr_full64.npy",mmap_mode="r")
X32=np.load(f"{LH}/tr_frames.npy",mmap_mode="r")

starts_tr=[]; starts_va=[]
for i in range(ntraj):
    for t0 in range(off[i],off[i]+lens[i]-39):
        if a.split == "none":
            starts_tr.append(t0)
            if i in VT:
                starts_va.append(t0)
        else:
            (starts_va if i in VT else starts_tr).append(t0)

starts_tr=np.array(starts_tr); starts_va=np.array(starts_va)
rng=np.random.default_rng(0)
va_sub=rng.choice(starts_va,size=min(900,len(starts_va)),replace=False)

print("[data] train %d win | val %d win (%d traj, split %s) | aug=%s"
      %(len(starts_tr),len(va_sub),len(VT),a.split,a.aug),flush=True)

def batch_tr(idx,g):
    if USE_FULL:
        py=g.integers(0,2,len(idx)); px=g.integers(0,2,len(idx))
        w=np.stack([np.asarray(XF[s:s+40,py[j]::2,px[j]::2]) for j,s in enumerate(idx)])
    else:
        w=np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z=np.zeros(w.shape[:-1]+(1,),np.float32); w=np.concatenate([w,z],-1).astype(np.float32)
    x=torch.from_numpy(w[:,:20]).to(DEV); y=torch.from_numpy(w[:,20:]).to(DEV)
    return x,y

def batch_va(idx):
    w=np.stack([np.asarray(X32[s:s+40]) for s in idx])
    z=np.zeros(w.shape[:-1]+(1,),np.float32); w=np.concatenate([w,z],-1).astype(np.float32)
    return torch.from_numpy(w[:,:20]).to(DEV), torch.from_numpy(w[:,20:]).to(DEV)

class CBAM(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(1, c//r), 1, bias=False)
        self.fc2 = nn.Conv2d(max(1, c//r), c, 1, bias=False)
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca
        avg_sp = torch.mean(x, 1, keepdim=True)
        max_sp, _ = torch.max(x, 1, keepdim=True)
        sa = torch.sigmoid(self.conv(torch.cat([avg_sp, max_sp], 1)))
        return x * sa

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
        s.cbam=CBAM(o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return s.cbam(F.gelu(s.n2(s.c2(x))))

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

# x is (B, 20, 32, 64, 3) -> only use first 2 channels
# input features: 20 * 2 = 40. output: 20 * 2 = 40.
model = UNet(40, 40, a.u_width).to(DEV)

def fwd(x):
    # flatten time and channel
    x2 = x[..., :2].permute(0, 1, 4, 2, 3).reshape(x.shape[0], -1, 32, 64)
    o = model(x2)
    o = o.view(x.shape[0], 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
    # output needs to be 3 channels for loss function
    z = torch.zeros(o.shape[:-1]+(1,), device=DEV)
    return torch.cat([o, z], -1)

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

MV=dict(rel_l2=0.467,tke=0.208,mvpe=0.290)
base=(45.71, 79.52, 60.10) # rough baseline numbers
def dacc(s): return MV['rel_l2']*(s[0]-base[0])+MV['tke']*(s[1]-base[1])+MV['mvpe']*(s[2]-base[2])
print("[baseline val] rel_l2 %.4f tke %.4f mvpe %.4f"%base,flush=True)

opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.wd)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.steps,pct_start=0.05)
g=np.random.default_rng(1234)

best=-9e9; t0=time.time(); model.train()
for step in range(1,a.steps+1):
    idx=g.choice(starts_tr,size=a.bs,replace=False)
    x,y=batch_tr(idx,g)
    
    p = fwd(x)
    l_fno = rel_l2(p, y) + a.wtke * tke_l2(p, y)
    
    # TV penalty on p
    dx = p[:, :, 1:, :, :2] - p[:, :, :-1, :, :2]
    dy = p[:, :, :, 1:, :2] - p[:, :, :, :-1, :2]
    l_tv = a.wtv * ((dx**2).mean() + (dy**2).mean())
    
    loss = l_fno + l_tv
    
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
    
    if step%a.evalevery==0 or step==a.steps:
        r=evaluate(); d=dacc(r); flag=""
        if d>best:
            best=d; torch.save(model.state_dict(),f"{B}/train_es/pure_unet_{a.tag}.pth"); flag=" *"
        print("step %6d loss %7.4f (fno %.4f tv %.4f) | rel_l2 %.4f (%+.4f) tke %.4f (%+.4f) mvpe %.4f (%+.4f) | d_acc %+.4f | %.0fs%s"
              %(step,float(loss.detach()), float(l_fno), float(l_tv), r[0],r[0]-base[0],r[1],r[1]-base[1],r[2],r[2]-base[2],d,time.time()-t0,flag),flush=True)

print("[done] best d_acc %+.4f  %.0fs"%(best,time.time()-t0))
