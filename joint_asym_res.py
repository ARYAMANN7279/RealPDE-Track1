import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"
SIG=0.0563870259
ap=argparse.ArgumentParser()
ap.add_argument("--gpu",type=int,default=0); ap.add_argument("--epochs",type=int,default=30)
ap.add_argument("--width",type=int,default=64); ap.add_argument("--bs",type=int,default=16)
ap.add_argument("--lr",type=float,default=2e-3); ap.add_argument("--cache",required=True)
ap.add_argument("--split",default="re_lohi"); ap.add_argument("--tag",required=True)
ap.add_argument("--wwidth",type=float,default=1.0)
a=ap.parse_args(); DEV=f"cuda:{a.gpu}"
torch.manual_seed(1234); np.random.seed(1234)
meta=json.load(open(f"{LH}/tr_meta.json")); names=meta["names"]
d=np.load(a.cache); wt=d["wt"]
RE=np.array([int(names[i].split("_")[0]) for i in wt])
VAL=np.array(["_15" in names[i] for i in wt]) if a.split=="aoa15" else np.isin(RE,[3750,5025,25425,26700])
TR=~VAL
XI=d["XI"]; PR=d["PR"]; RS=d["RS"]; SC=d["SC"]; del d
rs=float(RS[TR][SC[TR]].std())
LOGE=np.log(np.abs(RS)+1e-6).astype(np.float32)
mu_,sd_=float(LOGE[TR][SC[TR]].mean()),float(LOGE[TR][SC[TR]].std())
print(f"[data] {len(XI)} win | train {TR.sum()} val {VAL.sum()} | log|e| mu {mu_:.3f} sd {sd_:.3f}",flush=True)
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
net=UNet(80,120,a.width).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
itr=np.where(TR)[0]; iva=np.where(VAL)[0]
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.epochs*max(1,len(itr)//a.bs),pct_start=0.15)
def flat(z): return torch.from_numpy(np.ascontiguousarray(z)).permute(0,1,4,2,3).reshape(z.shape[0],-1,32,64)
def batch(ix):
    xi_flat = flat(XI[ix]).to(DEV)
    pr_flat = flat(PR[ix]).to(DEV)
    x=torch.cat([xi_flat,pr_flat],1)
    
    # Calculate local variance (std dev across 20 frames)
    # XI is (B, 20, H, W, C) -> flat is (B, 40, 32, 64)
    # Channel 0-19 is u, 20-39 is v
    u_std = xi_flat[:, :20].std(dim=1, keepdim=True).clamp(min=1e-4)
    v_std = xi_flat[:, 20:40].std(dim=1, keepdim=True).clamp(min=1e-4)
    local_std = torch.cat([u_std, v_std], 1) # (B, 2, 32, 64)
    
    r=flat(RS[ix]).to(DEV); m=flat(SC[ix].astype(np.float32)).to(DEV)
    return x,r,m,local_std

GRID=np.linspace(0.0002,0.06,1200)
def capt(score,err,scm,NB=24):
    tot=Eor=Ebc=Ep=0.0
    for ci in range(2):
        e=err[...,ci][scm[...,ci]]; s=score[...,ci][scm[...,ci]]
        if e.size==0: continue
        tot+=e.size; Eor+=float(np.exp(-2*e/SIG).sum())
        es=np.sort(e); k=np.searchsorted(es,GRID,side="right")
        Ebc+=float(np.max(np.exp(-2*GRID/SIG)*k))
        q=np.quantile(s,np.linspace(0,1,NB+1)[1:-1]); b=np.digitize(s,q)
        for kk in range(NB):
            ss=np.sort(e[b==kk])
            if ss.size==0: continue
            c=np.searchsorted(ss,GRID,side="right"); Ep+=float(np.max(np.exp(-2*GRID/SIG)*c))
    return 100*(Ep-Ebc)/(Eor-Ebc), Ebc/tot, Eor/tot

@torch.no_grad()
def evaluate():
    net.eval(); C=[]; W_D=[]; W_U=[]
    for i in range(0,len(iva),16):
        x,_,_,l_std=batch(iva[i:i+16]); o=net(x)
        c=o[:,:40].reshape(-1,20,2,32,64).permute(0,1,3,4,2)
        
        l_std = l_std.unsqueeze(1).expand(-1, 20, -1, -1, -1).permute(0,1,3,4,2)
        
        w_d = o[:,40:80].reshape(-1,20,2,32,64).permute(0,1,3,4,2)
        w_u = o[:,80:].reshape(-1,20,2,32,64).permute(0,1,3,4,2)
        
        w_d = (w_d * sd_ + mu_) + torch.log(l_std)
        w_u = (w_u * sd_ + mu_) + torch.log(l_std)
        
        C.append(c.cpu().numpy())
        W_D.append(w_d.cpu().numpy())
        W_U.append(w_u.cpu().numpy())
    net.train()
    C=np.concatenate(C,0); W=np.concatenate(W_D,0)
    R=RS[iva]; S=SC[iva]
    cu=np.corrcoef(R[...,0][S[...,0]],C[...,0][S[...,0]])[0,1]
    cv=np.corrcoef(R[...,1][S[...,1]],C[...,1][S[...,1]])[0,1]
    out={"corr_c":(cu,cv)}
    for al in (0.0,0.5,0.75,1.0):
        E=np.abs(R-al*C)
        f,bc,orc=capt(W,E,S)
        med=[float(np.median(E[...,ci][S[...,ci]])) for ci in range(2)]
        out["a%.2f"%al]=dict(capt=f,Ebc=bc,Eor=orc,med=med)
    return out

t0=time.time(); best=-1
for ep in range(1,a.epochs+1):
    perm=itr.copy(); np.random.default_rng(ep).shuffle(perm); tl=0.0; nb=0
    for i in range(0,len(perm)-a.bs+1,a.bs):
        x,r,m,l_std=batch(perm[i:i+a.bs]); o=net(x)
        c=o[:,:40]; w_d=o[:,40:80]; w_u=o[:,80:]
        l_c=((c-r).abs()*m).sum()/m.sum()/rs
        
        l_std_expanded = l_std.unsqueeze(1).expand(-1, 20, -1, -1, -1).reshape(-1, 40, 32, 64)
        
        tgt_d=(torch.log(torch.relu(-(r-c.detach()))+1e-6) - torch.log(l_std_expanded) - mu_)/sd_
        tgt_u=(torch.log(torch.relu(r-c.detach())+1e-6) - torch.log(l_std_expanded) - mu_)/sd_
        
        l_w=(((w_d-tgt_d).abs()+(w_u-tgt_u).abs())*m).sum()/m.sum()
        loss=l_c+a.wwidth*l_w
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        tl+=float(loss.detach()); nb+=1
    if ep%2==0 or ep==a.epochs:
        r=evaluate(); f0=r["a0.00"]; f1=r["a1.00"]; flag=""
        sc=f1["Ebc"]
        if sc>best:
            best=sc; torch.save({"sd":net.state_dict(),"w":a.width,"mu":mu_,"sd_":sd_,"rs":rs,
                                 "cache":a.cache,"split":a.split},
                                f"{B}/train_es/joint_{a.tag}.pth"); flag=" *"
        print("%5d %9.4f | Ebc %.4f->%.4f  capt %.1f%%  %ds%s"
              %(ep,tl/nb,f0["Ebc"],f1["Ebc"],f1["capt"],time.time()-t0,flag),flush=True)
