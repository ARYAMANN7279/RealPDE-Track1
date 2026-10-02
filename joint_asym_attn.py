"""ONE U-Net, 80 in -> 80 out: 40 channels of bound-CENTRE correction c (sec30.11) and
40 channels of log|residual after correction| for the width LUT. One forward pass for the
whole bounds stack.

Reports the calibration-free captured fraction, and -- more importantly -- the raw
coverage gain the centre shift buys at FIXED widths, which needs no error-scale model
at all (the width penalty exp(-2h/sigma) is unchanged, so only the indicator moves).
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"
SIG=0.0563870259
ap=argparse.ArgumentParser()
ap.add_argument("--gpu",type=int,default=0); ap.add_argument("--epochs",type=int,default=30)
ap.add_argument("--width",type=int,default=64); ap.add_argument("--bs",type=int,default=16)
ap.add_argument("--lr",type=float,default=2e-3); ap.add_argument("--cache",required=True)
ap.add_argument("--split",default="re_lohi"); ap.add_argument("--tag",required=True)
ap.add_argument("--wwidth",type=float,default=1.0,help="weight on the width-head loss")
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
print("[data] %d win | train %d val %d (%s) | resid sd %.6f | log|e| mu %.3f sd %.3f"
      %(len(XI),TR.sum(),VAL.sum(),a.split,rs,mu_,sd_),flush=True)

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(1, channels // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // r), channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
        s.se = SEBlock(o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return s.se(F.gelu(s.n2(s.c2(x))))
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
NP=sum(p.numel() for p in net.parameters())
print("[net] width %d  %.2fM params  (%.1f MB fp32)"%(a.width,NP/1e6,NP*4/1e6),flush=True)
opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
itr=np.where(TR)[0]; iva=np.where(VAL)[0]
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.epochs*max(1,len(itr)//a.bs),pct_start=0.15)
def flat(z): return torch.from_numpy(np.ascontiguousarray(z)).permute(0,1,4,2,3).reshape(z.shape[0],-1,32,64)
def batch(ix):
    x=torch.cat([flat(XI[ix]),flat(PR[ix])],1).to(DEV)
    r=flat(RS[ix]).to(DEV); m=flat(SC[ix].astype(np.float32)).to(DEV)
    return x,r,m
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
        x,_,_=batch(iva[i:i+16]); o=net(x)
        C.append(o[:,:40].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
        W_D.append(o[:,40:80].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
        W_U.append(o[:,80:].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy()) #(o[:,40:80].reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
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
print("\n%5s %9s %8s %8s | %s"%("ep","loss","corr_cu","corr_cv","Ebc/capt at alpha 0.00 / 1.00"),flush=True)
t0=time.time(); best=-1
for ep in range(1,a.epochs+1):
    perm=itr.copy(); np.random.default_rng(ep).shuffle(perm); tl=0.0; nb=0
    for i in range(0,len(perm)-a.bs+1,a.bs):
        x,r,m=batch(perm[i:i+a.bs]); o=net(x)
        c=o[:,:40]; w_d=o[:,40:80]; w_u=o[:,80:]
        l_c=((c-r).abs()*m).sum()/m.sum()/rs
        tgt_d=(torch.log(torch.relu(-(r-c.detach()))+1e-6)-mu_)/sd_
        tgt_u=(torch.log(torch.relu(r-c.detach())+1e-6)-mu_)/sd_
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
        print("%5d %9.4f %8.3f %8.3f | Ebc %.4f->%.4f  Eor %.4f->%.4f  capt %.1f%%->%.1f%%  med_u %.5f->%.5f  %.0fs%s"
              %(ep,tl/nb,r["corr_c"][0],r["corr_c"][1],f0["Ebc"],f1["Ebc"],f0["Eor"],f1["Eor"],
                f0["capt"],f1["capt"],f0["med"][0],f1["med"][0],time.time()-t0,flag),flush=True)
r=evaluate()
print("\n=== FINAL (val = %s, held out of training) ==="%a.split)
print("corr(c, residual): u %.3f  v %.3f"%r["corr_c"])
for al in (0.0,0.5,0.75,1.0):
    k=r["a%.2f"%al]
    print("  alpha %.2f : E_bc %.4f  E_oracle %.4f  captured %5.1f%%  median|res| u %.6f v %.6f"
          %(al,k["Ebc"],k["Eor"],k["capt"],k["med"][0],k["med"][1]))
json.dump({k:(v if not isinstance(v,tuple) else list(v)) for k,v in r.items()},
          open(f"{B}/train_es/joint_{a.tag}.json","w"),default=float,indent=1)
print("[done] %.0fs"%(time.time()-t0))
