"""Uncertainty head v3: REGRESS log|err| instead of optimising the step surrogate.

v2 optimised exp(-2h/s)*sigmoid((h-e)/tau) directly. At small tau that is nearly a
step function, so gradients vanish except where h ~ e; the net stayed near-constant
(sd(log h) = 0.36 vs the ~1.2 needed).

Regression on log|err| has well-conditioned gradients everywhere, so the net can
actually learn spatial structure. Then h = k*exp(mu) with a single global k tuned
to maximise E -- the analytic optimum for a log-scale error model.
"""
import json,os,sys,time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; DEV="cuda:2"; W=0.6784
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); sims=sorted(set(sim.tolist()))
SCM=(Y[...,:C]!=0.0); tr=np.isin(sim,sims[0::2]); te=~tr
EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
m0,m1=SCM[...,0],SCM[...,1]; SUB=40
eu=EL[...,0][m0][::SUB]; ev=EL[...,1][m1][::SUB]
au=AP[...,0][m0][::SUB]; av=AP[...,1][m1][::SUB]; NT=eu.size+ev.size
best=None
for al in np.linspace(1.0,6.0,26):
    for be in np.linspace(-12,2,29):
        su,sv=eu*(al+be*au),ev*(al+be*av)
        Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
        hu,hv=0.05*au,0.05*av
        Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
        r=abs(Ec-0.4399)+abs(Ep-0.2076)
        if best is None or r<best[0]: best=(r,al,be)
_,AL,BE=best
ERR=(EL*(AL+BE*AP)).astype(np.float32)
LOGE=np.log(ERR+1e-6).astype(np.float32)
def feats(Pw):
    u,v=Pw[...,0],Pw[...,1]
    gux=np.gradient(u,axis=3); guy=np.gradient(u,axis=2)
    gvx=np.gradient(v,axis=3); gvy=np.gradient(v,axis=2)
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2); vo=np.abs(gvx-guy)
    lap=np.abs(np.gradient(gux,axis=3)+np.gradient(guy,axis=2))
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tvv=np.repeat(v.std(axis=1,keepdims=True),v.shape[1],axis=1)
    ke=0.5*(u*u+v*v); dev=np.abs(u-u.mean(axis=1,keepdims=True))
    dv=np.abs(v-v.mean(axis=1,keepdims=True))
    yy,xx=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,64),indexing="ij")
    yy=np.broadcast_to(yy,u.shape); xx=np.broadcast_to(xx,u.shape)
    tf=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    return np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(lap+1e-6),np.log(tv+1e-6),
                     np.log(tvv+1e-6),np.log(ke+1e-9),np.log(dev+1e-6),np.log(dv+1e-6),
                     u,v,xx,yy,tf],axis=-1).astype(np.float32)
FT=feats(P); NF=FT.shape[-1]
mu_=FT[tr].reshape(-1,NF).mean(0); sd_=FT[tr].reshape(-1,NF).std(0)+1e-6
FT=(FT-mu_)/sd_
print("features %d ; train %d / test %d windows"%(NF,tr.sum(),te.sum()),flush=True)
class Net(nn.Module):
    def __init__(s,nf,w=128):
        super().__init__()
        s.n=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
            nn.Conv2d(w,w,3,padding=8,dilation=8),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,2,1))
    def forward(s,x): return s.n(x)
net=Net(NF).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
EP=26
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP,eta_min=5e-5)
def to_t(a): return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
def predict_mu(mask,bs=8):
    out=[]
    with torch.no_grad():
        for i in np.arange(0,mask.sum(),bs):
            j=np.where(mask)[0][int(i):int(i)+bs]
            f=to_t(FT[j]).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            out.append(net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out,0)
def bestE(mask,MU):
    e=ERR[mask]; m=SCM[mask]
    bb=(-1,1.0)
    for k in np.linspace(0.6,6.0,40):
        h=k*np.exp(MU)
        ok=(e<=h)&m
        E=float((np.exp(-2*h/SIG)*ok).sum()/m.sum())
        if E>bb[0]: bb=(E,k)
    return bb
BEST=-1
for ep in range(EP):
    idx=np.where(tr)[0]; np.random.default_rng(ep).shuffle(idx)
    for i in range(0,len(idx),8):
        j=idx[i:i+8]
        f=to_t(FT[j]).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        p=net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2)
        t=to_t(LOGE[j]); m=to_t(SCM[j].astype(np.float32))
        loss=((p-t).abs()*m).sum()/m.sum()          # L1 on log|err|
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    sched.step()
    if ep%4==3 or ep==EP-1:
        MUte=predict_mu(te); Ete,kte=bestE(te,MUte)
        r=np.corrcoef(MUte[SCM[te]],LOGE[te][SCM[te]])[0,1]
        star=""
        if Ete>BEST:
            BEST=Ete; torch.save({"sd":net.state_dict(),"mu":mu_,"std":sd_,"nf":NF,"k":kte},
                                 f"{B}/local_harness/unc3_best.pth"); star=" *"
        print("  ep %2d  L1 %.3f  corr(pred,true logerr) %.3f  sd(mu) %.2f  E_test %.4f (k=%.2f)%s"%(
            ep,float(loss),r,float(MUte[SCM[te]].std()),Ete,kte,star),flush=True)
RATIO=0.4876/0.5132
print("\nBASELINE constant: model E 0.5132 -> real 0.4876 (sps 33.08, final 78.07)")
print("v2 head        : model E 0.5586 -> real ~%.4f"%(0.5586*RATIO))
print("v3 head        : model E %.4f -> real ~%.4f -> sps ~%.2f -> DELTA final %+.2f -> ~%.2f"%(
    BEST,BEST*RATIO,100*W*BEST*RATIO,0.217*(100*W*BEST*RATIO-33.078742),
    78.068714+0.217*(100*W*BEST*RATIO-33.078742)))
print("top-20 real E 0.5896")
