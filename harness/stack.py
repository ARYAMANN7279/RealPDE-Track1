"""FULL STACK on held-out trajectories: model -> temporal spectral correction ->
learned uncertainty-head bounds. Measures the real interaction (accuracy changes W,
which scales sps; the correction also changes the error distribution the head sees).
Everything fit on TRAIN trajectories, scored on disjoint TEST ones.
"""
import json,os,sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT)
import importlib.util
sp=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
SIG=S.SIGMA_GLOBAL; C=2; DEV="cuda:2"
d=np.load(f"{B}/local_harness/comp_eval_cache.npz")
P0=d["P"].astype(np.float32); Y=d["Y"].astype(np.float32)
rows=json.load(open(f"{B}/local_harness/comp_anchor_rows.json"))
sim=np.concatenate([[r["sim"]]*r["n"] for r in rows]); sims=sorted(set(sim.tolist()))
tr=np.isin(sim,sims[0::2]); te=~tr; SCM=(Y[...,:C]!=0.0)
n_=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL0,TK0,MV0,TM=94.168150,74.025866,92.836278,91.362375
Wf=lambda rl,tk,mv:0.5*(1-n_(inv(rl)))+0.3*(1-n_(inv(tk)))+0.2*(1-n_(inv(mv)))
def sub(Q,m):
    dm=S.rel_l2_per_sample(Q[m],Y[m],C);tk=S.tke_rel_l2_per_sample(Q[m],Y[m],C);mv=S.mvpe_rel_l2_per_sample(Q[m],Y[m])
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
base=sub(P0,te)
# temporal spectral gain, fit on TRAIN
Fp=np.fft.rfft(P0[tr][...,:C],axis=1); Fy=np.fft.rfft(Y[tr][...,:C],axis=1)
gt=np.sqrt((np.abs(Fy)**2).mean(axis=(0,2,3))/np.maximum((np.abs(Fp)**2).mean(axis=(0,2,3)),1e-20))
gt=np.clip(gt,1/2.0,2.0)
def apply_t(Q):
    Fq=np.fft.rfft(Q[...,:C],axis=1)*gt[None,:,None,None,:]
    o=Q.copy(); o[...,:C]=np.fft.irfft(Fq,n=Q.shape[1],axis=1); return o
P1=apply_t(P0); corr=sub(P1,te)
print("held-out accuracy:")
print("  base      rel_l2 %.2f tke %.2f mvpe %.2f"%base)
print("  +spectral rel_l2 %.2f tke %.2f mvpe %.2f"%corr,flush=True)
# calibrate errors to real scale for EACH prediction set
def calib(P):
    EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
    m0,m1=SCM[...,0],SCM[...,1]; SB=40
    eu=EL[...,0][m0][::SB]; ev=EL[...,1][m1][::SB]; au=AP[...,0][m0][::SB]; av=AP[...,1][m1][::SB]
    NT=eu.size+ev.size; best=None
    for al in np.linspace(1.0,6.0,26):
        for be in np.linspace(-12,2,29):
            su,sv=eu*(al+be*au),ev*(al+be*av)
            Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
            hu,hv=0.05*au,0.05*av
            Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
            r=abs(Ec-0.4399)+abs(Ep-0.2076)
            if best is None or r<best[0]: best=(r,al,be)
    _,AL,BE=best
    return (EL*(AL+BE*AP)).astype(np.float32)
def feats(Pw):
    u,v=Pw[...,0],Pw[...,1]
    gux=np.gradient(u,axis=3); guy=np.gradient(u,axis=2)
    gvx=np.gradient(v,axis=3); gvy=np.gradient(v,axis=2)
    g=np.sqrt(gux**2+guy**2+gvx**2+gvy**2); vo=np.abs(gvx-guy)
    lap=np.abs(np.gradient(gux,axis=3)+np.gradient(guy,axis=2))
    tv=np.repeat(u.std(axis=1,keepdims=True),u.shape[1],axis=1)
    tvv=np.repeat(v.std(axis=1,keepdims=True),v.shape[1],axis=1)
    ke=0.5*(u*u+v*v); dev=np.abs(u-u.mean(axis=1,keepdims=True)); dv=np.abs(v-v.mean(axis=1,keepdims=True))
    yy,xx=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,64),indexing="ij")
    yy=np.broadcast_to(yy,u.shape); xx=np.broadcast_to(xx,u.shape)
    tf=np.broadcast_to(np.linspace(0,1,u.shape[1])[None,:,None,None],u.shape)
    return np.stack([np.log(g+1e-6),np.log(vo+1e-6),np.log(lap+1e-6),np.log(tv+1e-6),
                     np.log(tvv+1e-6),np.log(ke+1e-9),np.log(dev+1e-6),np.log(dv+1e-6),
                     u,v,xx,yy,tf],axis=-1).astype(np.float32)
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
ck=torch.load(f"{B}/local_harness/unc3_best.pth",map_location=DEV,weights_only=False)
NF=ck["nf"]; net=Net(NF).to(DEV); net.load_state_dict(ck["sd"]); net.eval()
def mu_of(P,mask,bs=8):
    FT=(feats(P)-ck["mu"])/ck["std"]; out=[]; idx=np.where(mask)[0]
    with torch.no_grad():
        for i in range(0,len(idx),bs):
            j=idx[i:i+bs]
            f=torch.from_numpy(np.ascontiguousarray(FT[j])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
            out.append(net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out,0)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.012
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
RATIO=0.4876/0.5132
def run(P,tag,acc):
    ERR=calib(P); MUtr=mu_of(P,tr); MUte=mu_of(P,te)
    NB=24; LUT=np.zeros((NB,C),np.float32); ED=[]
    for ci in range(C):
        m=SCM[tr][...,ci]; mu=MUtr[...,ci][m]; er=ERR[tr][...,ci][m]
        q=np.quantile(mu,np.linspace(0,1,NB+1)[1:-1]); ED.append(q)
        b=np.digitize(mu,q)
        for k in range(NB):
            s_=b==k; LUT[k,ci]=best_h(er[s_][::3]) if s_.sum()>200 else 0.012
    tot=0.0;cnt=0.0
    for ci in range(C):
        m=SCM[te][...,ci]; mu=MUte[...,ci][m]; er=ERR[te][...,ci][m]
        h=LUT[np.digitize(mu,ED[ci]),ci]
        tot+=float((np.exp(-2*h/SIG)*(er<=h)).sum()); cnt+=float(m.sum())
    E=tot/cnt; Ereal=E*RATIO
    rl,tk,mv=RL0+(acc[0]-base[0]),TK0+(acc[1]-base[1]),MV0+(acc[2]-base[2])
    sps=100*Wf(rl,tk,mv)*Ereal
    dfin=0.306*(rl-RL0)+0.163*(tk-TK0)+0.218*(mv-MV0)+0.217*(sps-33.078742)
    print("  %-26s rel_l2 %.2f tke %.2f mvpe %.2f | E %.4f sps %.2f | final ~%.2f (%+.2f)"%(
        tag,rl,tk,mv,Ereal,sps,78.068714+dfin,dfin),flush=True)
print("\n=== FULL STACK (all deltas vs the banked 78.07) ===")
run(P0,"model + head bounds",base)
run(P1,"model + spectral + head",corr)
