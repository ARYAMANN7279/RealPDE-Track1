"""THE CRUX EXPERIMENT.

M55 was fine-tuned on 55 trajectories and has never seen the other 26. So:
  IN  = M55's errors on its 55 training trajectories   (in-sample, like train_real
        is for the official checkpoint)
  OUT = M55's errors on the 26 held-out trajectories   (honest, like the hidden eval)

Fit the bound policies on IN, score them on OUT. That reproduces exactly the
situation that made FULLSTACK_v6 score 76.61, but this time we can MEASURE it.

Questions:
  1. Does a head fitted in-sample beat CONSTANTS out-of-sample? (If not, dead.)
  2. How much does an in-sample fit over-estimate E out-of-sample? (the bias factor)
  3. Does a safety widening recover the loss?
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util as iu
sp=iu.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda"; C=2; IN=20; HOR=40
MI=torch.tensor([0.154960856,-0.000513992854,0.0]).to(DEV); SI=torch.tensor([0.0968056545,0.015960684,1.0]).to(DEV)
MT=torch.tensor([0.154962569,-0.000517793698,0.0]).to(DEV); ST=torch.tensor([0.0968104079,0.0159636438,1.0]).to(DEV)
F=np.load(f"{B}/train_work/frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/train_work/meta.json")); off=meta["off"]; lens=meta["lens"]
rng=np.random.default_rng(1234); order=rng.permutation(len(lens))
TR=set(order[:55].tolist()); VA=set(order[55:].tolist())
def wins(ts,stride):
    w=[]
    for i in ts:
        for t0 in range(off[i],off[i]+lens[i]-HOR+1,stride): w.append(t0)
    return np.array(w)
Win, Wout = wins(TR,60), wins(VA,40)
model,_=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)
model.load_state_dict(torch.load(f"{B}/local_harness/m55.pth",map_location=DEV,weights_only=False))
model=model.to(DEV).eval()
def preds(wl):
    P=[];Y=[]
    with torch.no_grad():
        for i in range(0,len(wl),16):
            a=np.stack([np.concatenate([np.asarray(F[s:s+HOR]),
                np.zeros((HOR,32,64,1),np.float32)],-1) for s in wl[i:i+16]]).astype(np.float32)
            x=torch.from_numpy(a[:,:IN]).to(DEV)
            P.append((model((x-MI)/SI)*ST+MT).cpu().numpy()); Y.append(a[:,IN:])
    return np.concatenate(P,0).astype(np.float32), np.concatenate(Y,0).astype(np.float32)
Pi,Yi = preds(Win); Po,Yo = preds(Wout)
Mi=(Yi[...,:C]!=0.0); Mo=(Yo[...,:C]!=0.0)
Ei=np.abs(Pi[...,:C]-Yi[...,:C]); Eo=np.abs(Po[...,:C]-Yo[...,:C])
print("IN  (in-sample)  %d windows | median |err_u| %.5f"%(len(Pi),np.median(Ei[...,0][Mi[...,0]])),flush=True)
print("OUT (held-out)   %d windows | median |err_u| %.5f  -> %.2fx LARGER"%(
    len(Po),np.median(Eo[...,0][Mo[...,0]]),
    np.median(Eo[...,0][Mo[...,0]])/np.median(Ei[...,0][Mi[...,0]])),flush=True)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.012
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
def Escore(h,E,M):
    ok=(E<=h)&M
    return float((np.exp(-2*h/SIG)*ok).sum()/M.sum()), float(ok.sum()/M.sum())
# ---- CONSTANT fitted on IN, scored on OUT ----
hc=np.zeros((1,1,1,1,C),np.float32)
for ci in range(C): hc[...,ci]=best_h(Ei[...,ci][Mi[...,ci]][::7])
Ec_in,_=Escore(np.broadcast_to(hc,Ei.shape),Ei,Mi)
Ec_out,cc=Escore(np.broadcast_to(hc,Eo.shape),Eo,Mo)
print("\nCONSTANT [%.4f,%.4f] fitted IN:"%(hc[...,0].item(),hc[...,1].item()))
print("   E on IN  %.4f   |   E on OUT %.4f (cov %.3f)   -> in-sample over-estimates by %+.4f"%(
    Ec_in,Ec_out,cc,Ec_in-Ec_out),flush=True)
# ---- HEAD fitted on IN, scored on OUT ----
def feats(Pw):
    u,v=Pw[...,0],Pw[...,1]
    gux=np.gradient(u,axis=3);guy=np.gradient(u,axis=2);gvx=np.gradient(v,axis=3);gvy=np.gradient(v,axis=2)
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2); vo=np.abs(gvx-guy)
    lap=np.abs(np.gradient(gux,axis=3)+np.gradient(guy,axis=2))
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tvv=np.repeat(v.std(axis=1,keepdims=True),v.shape[1],axis=1)
    ke=0.5*(u*u+v*v); dev=np.abs(u-u.mean(axis=1,keepdims=True)); dv=np.abs(v-v.mean(axis=1,keepdims=True))
    yy,xx=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,64),indexing="ij")
    yy=np.broadcast_to(yy,u.shape); xx=np.broadcast_to(xx,u.shape)
    tf=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    return np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(lap+1e-6),np.log(tv+1e-6),
        np.log(tvv+1e-6),np.log(ke+1e-9),np.log(dev+1e-6),np.log(dv+1e-6),u,v,xx,yy,tf],-1).astype(np.float32)
class Net(nn.Module):
    def __init__(s,nf,w=96):
        super().__init__()
        s.n=nn.Sequential(nn.Conv2d(nf,w,3,padding=1),nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4),nn.GELU(),
            nn.Conv2d(w,w,3,padding=1),nn.GELU(),nn.Conv2d(w,2,1))
    def forward(s,x): return s.n(x)
Fi=feats(Pi); NF=Fi.shape[-1]
mu_=Fi.reshape(-1,NF).mean(0); sd_=Fi.reshape(-1,NF).std(0)+1e-6
Fi=(Fi-mu_)/sd_; Fo=(feats(Po)-mu_)/sd_
LOG=np.log(Ei+1e-6).astype(np.float32)
net=Net(NF).to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=18,eta_min=5e-5)
for ep in range(18):
    idx=np.random.default_rng(ep).permutation(len(Fi))
    for i in range(0,len(idx),8):
        j=idx[i:i+8]
        f=torch.from_numpy(np.ascontiguousarray(Fi[j])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        p=net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2)
        t=torch.from_numpy(np.ascontiguousarray(LOG[j])).to(DEV)
        m=torch.from_numpy(Mi[j].astype(np.float32)).to(DEV)
        loss=((p-t).abs()*m).sum()/m.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
def mu_of(FT):
    o=[]
    with torch.no_grad():
        for i in range(0,len(FT),8):
            f=torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            o.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(o,0)
MUi=mu_of(Fi); MUo=mu_of(Fo)
NB=20; LUT=np.zeros((NB,C),np.float32); ED=[]
for ci in range(C):
    m=Mi[...,ci]; q=np.quantile(MUi[...,ci][m],np.linspace(0,1,NB+1)[1:-1]); ED.append(q)
    b=np.digitize(MUi[...,ci][m],q); er=Ei[...,ci][m]
    for k in range(NB):
        s_=b==k; LUT[k,ci]=best_h(er[s_][::3]) if s_.sum()>200 else 0.012
def head_h(MU,shape,mult=1.0):
    h=np.zeros(shape,np.float32)
    for ci in range(C): h[...,ci]=LUT[np.digitize(MU[...,ci],ED[ci]),ci]*mult
    return h
Eh_in,_=Escore(head_h(MUi,Ei.shape),Ei,Mi)
Eh_out,ch=Escore(head_h(MUo,Eo.shape),Eo,Mo)
print("\nLEARNED HEAD fitted IN:")
print("   E on IN  %.4f   |   E on OUT %.4f (cov %.3f)   -> in-sample over-estimates by %+.4f"%(
    Eh_in,Eh_out,ch,Eh_in-Eh_out),flush=True)
print("\n*** Q1: does the head beat CONSTANTS out-of-sample? ***")
print("   constant OUT %.4f  vs  head OUT %.4f   -> head is %s by %+.4f"%(
    Ec_out,Eh_out,"BETTER" if Eh_out>Ec_out else "WORSE",Eh_out-Ec_out))
print("\n*** Q3: does a safety widening recover it? ***")
for mm in (1.0,1.25,1.5,1.75,2.0,2.5):
    E_,c_=Escore(head_h(MUo,Eo.shape,mm),Eo,Mo)
    print("   head x%.2f -> E %.4f cov %.3f %s"%(mm,E_,c_,"<-- beats constant" if E_>Ec_out else ""))

# ---- SUSPECT 2: the alpha+beta*|pred| calibration transform ----
# FULLSTACK built its LUT on errors multiplied per-element by (alpha+beta*|pred|),
# a transform fitted only to match AGGREGATE E at two bound settings. If the true
# per-element relationship differs, that transform CORRUPTS the error ranking the
# LUT is built from. Test: build the LUT on transformed errors, score on real ones.
AL,BE=2.80,-8.00
APi=np.abs(Pi[...,:C]); ERRt=(Ei*(AL+BE*APi)).astype(np.float32)
LUT2=np.zeros((NB,C),np.float32); ED2=[]
for ci in range(C):
    m=Mi[...,ci]; q=np.quantile(MUi[...,ci][m],np.linspace(0,1,NB+1)[1:-1]); ED2.append(q)
    b=np.digitize(MUi[...,ci][m],q); er=ERRt[...,ci][m]
    for k in range(NB):
        s_=b==k; LUT2[k,ci]=best_h(er[s_][::3]) if s_.sum()>200 else 0.012
def head_h2(MU,shape,mult=1.0):
    h=np.zeros(shape,np.float32)
    for ci in range(C): h[...,ci]=LUT2[np.digitize(MU[...,ci],ED2[ci]),ci]*mult
    return h
E2,c2=Escore(head_h2(MUo,Eo.shape),Eo,Mo)
print(chr(10)+"*** SUSPECT 2: LUT built on alpha+beta*|pred|-transformed errors ***")
print("   clean LUT      -> E %.4f (cov %.3f)"%(Eh_out,ch))
print("   transformed LUT -> E %.4f (cov %.3f)   delta %+.4f"%(E2,c2,E2-Eh_out))
print("   constant        -> E %.4f"%Ec_out)
print("   => %s"%("TRANSFORM IS THE CULPRIT" if E2<Ec_out else "transform is NOT the culprit"))