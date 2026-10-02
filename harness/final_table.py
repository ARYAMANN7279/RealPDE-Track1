"""FINAL TABLE under the two-anchor-corrected error model.
err_real = err_local * (alpha + beta*|pred|), alpha/beta fitted to BOTH real anchors.
Every model x every bounds policy. Bounds fit on half the held-out trajectories,
scored on the other half. Accuracy measured directly -> deltas onto the real 77.20.
"""
import json, os, sys
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda:2"; C=2; IN=20
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
mi,si,mt,st=[torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
Wall=np.stack([np.concatenate([np.asarray(X[s:s+40]),
        np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
mdl,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
def run(sd):
    mdl.load_state_dict(sd); mdl.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            o.append((mdl((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
ORIG=torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV); ORIG=ORIG.get("model_state_dict",ORIG)
MODELS={"original":ORIG,"soup(all6)":torch.load(f"{B}/local_harness/soup_best.pth",map_location=DEV)}
PR={k:run(v) for k,v in MODELS.items()}
def acc(P,m):
    dm=S.rel_l2_per_sample(P[m],Y[m],C);tk=S.tke_rel_l2_per_sample(P[m],Y[m],C);mv=S.mvpe_rel_l2_per_sample(P[m],Y[m])
    return (S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
# --- fit alpha,beta on ORIGINAL against BOTH real anchors ---
P0=PR["original"]; EL=np.abs(P0[...,:C]-Y[...,:C]); AP=np.abs(P0[...,:C])
m0,m1=SCM[...,0],SCM[...,1]
eu,ev_=EL[...,0][m0],EL[...,1][m1]; au,av=AP[...,0][m0],AP[...,1][m1]; NT=eu.size+ev_.size
best=None
for al in np.linspace(1.0,6.0,51):
    for be in np.linspace(-12,2,57):
        su,sv=eu*(al+be*au),ev_*(al+be*av)
        Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
        hu,hv=0.05*au,0.05*av
        Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
        r=abs(Ec-0.4399)+abs(Ep-0.2076)
        if best is None or r<best[0]: best=(r,al,be)
r,AL,BE=best
print("two-anchor fit: alpha=%.3f beta=%.3f residual %.4f"%(AL,BE,r),flush=True)
n=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
Wf=lambda rl,tk,mv:0.5*(1-n(inv(rl)))+0.3*(1-n(inv(tk)))+0.2*(1-n(inv(mv)))
fin=lambda rl,tk,mv,sps:0.306*rl+0.163*tk+0.218*mv+0.100*TM+0.217*sps
BASE=fin(RL,TK,MV,29.84)
A0=acc(P0,np.ones(len(wins),bool))
h=len(wins)//2; fit=np.zeros(len(wins),bool); fit[:h]=True; ev=~fit
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
out=[]
for mk,P in PR.items():
    SCALE=AL+BE*np.abs(P[...,:C])
    ERR=np.abs(P[...,:C]-Y[...,:C])*SCALE          # corrected to real scale
    a=acc(P,ev); d=[a[i]-A0[i] for i in range(3)]
    rl,tk,mv=RL+d[0],TK+d[1],MV+d[2]; Wn=Wf(rl,tk,mv)
    hk=np.zeros((1,1,C),np.float32)
    for ci in range(C): hk[0,0,ci]=best_h(ERR[fit][...,ci][SCM[fit][...,ci]][::17])
    HL=np.zeros((32,64,C),np.float32)
    for ci in range(C):
        E=ERR[fit][...,ci]; M=SCM[fit][...,ci]
        for i in range(32):
            for j in range(64): HL[i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]])
    def sc(H):
        hh=np.broadcast_to(H[None,None],ERR[ev].shape); ok=(ERR[ev]<=hh)&SCM[ev]
        return float((np.exp(-2*hh/SIG)*ok).sum()/SCM[ev].sum()),float(ok.sum()/SCM[ev].sum())
    pol={"current [.030,.010]":np.array([[[0.030,0.010]]],np.float32)*np.ones((32,64,1),np.float32),
         "best constant [%.4f,%.4f]"%(hk[0,0,0],hk[0,0,1]):hk*np.ones((32,64,1),np.float32)}
    for mm in [1.00,1.15]: pol["per-location x%.2f"%mm]=HL*mm
    for pk,H in pol.items():
        E,cv=sc(H); sps=100*Wn*E
        out.append((mk,pk,rl,tk,mv,sps,cv,77.20+fin(rl,tk,mv,sps)-BASE))
    np.savez(f"{B}/local_harness/final_{mk.split('(')[0]}.npz",hk=hk,HL=HL)
out.sort(key=lambda r:-r[7])
print("\n%-11s %-28s %7s %7s %7s %7s %6s %9s"%("model","bounds","rel_l2","tke","mvpe","sps","cov","EST REAL"))
print("-"*98)
for mk,pk,rl,tk,mv,sps,cv,f in out:
    print("%-11s %-28s %7.2f %7.2f %7.2f %7.2f %6.3f %9.2f"%(mk,pk,rl,tk,mv,sps,cv,f))
print("-"*98)
print("known-true: original + current bounds = 77.20 | agent33 (identical model) = 78.64")
