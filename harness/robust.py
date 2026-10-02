"""Is per-timestep x location real, or is it fitting noise?
81920 free parameters fitted on ~336 windows is a lot. Test BOTH split
directions; a real effect survives the swap, an overfit does not.
Also sweeps a safety multiplier on each family (too-tight scores exactly 0).
"""
import json, os, sys, importlib.util
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; C=2; IN=20; DEV="cuda:2"
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
mi,si,mt,st=[torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
vidx=sorted(set(range(0,len(lens),5)))
wins=[];wt=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0); wt.append(i)
wt=np.array(wt)
Wall=np.stack([np.concatenate([np.asarray(X[s:s+40]),
        np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
mdl,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
sd=torch.load(f"{B}/local_harness/sim_real_fno_unwrapped.pth",map_location=DEV)
mdl.load_state_dict(sd); mdl.to(DEV).eval(); o=[]
with torch.no_grad():
    for i in range(0,len(Xin),32):
        o.append((mdl((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
P=np.concatenate(o,0).astype(np.float32)
EL=np.abs(P[...,:C]-Y[...,:C]); AP=np.abs(P[...,:C])
n=lambda x:x/(0.5+x); inv=lambda s:(100.0/s-1.0)*2.0
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
WR=0.5*(1-n(inv(RL)))+0.3*(1-n(inv(TK)))+0.2*(1-n(inv(MV)))
fin=lambda sps:0.306*RL+0.163*TK+0.218*MV+0.100*TM+0.217*sps
BASE=fin(29.84)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
def calib(fitm):
    m0,m1=SCM[fitm][...,0],SCM[fitm][...,1]
    eu,ev_=EL[fitm][...,0][m0],EL[fitm][...,1][m1]; au,av=AP[fitm][...,0][m0],AP[fitm][...,1][m1]
    NT=eu.size+ev_.size; best=None
    for al in np.linspace(1.0,6.0,51):
        for be in np.linspace(-12,2,57):
            su,sv=eu*(al+be*au),ev_*(al+be*av)
            Ec=(np.exp(-2*0.030/SIG)*(su<=0.030).sum()+np.exp(-2*0.010/SIG)*(sv<=0.010).sum())/NT
            hu,hv=0.05*au,0.05*av
            Ep=((np.exp(-2*hu/SIG)*(su<=hu)).sum()+(np.exp(-2*hv/SIG)*(sv<=hv)).sum())/NT
            r=abs(Ec-0.4399)+abs(Ep-0.2076)
            if best is None or r<best[0]: best=(r,al,be)
    return best[1],best[2]
def build(fitm,ERR):
    HL=np.zeros((1,1,32,64,C),np.float32); HTL=np.zeros((1,20,32,64,C),np.float32)
    for ci in range(C):
        E=ERR[fitm][...,ci]; M=SCM[fitm][...,ci]
        for i in range(32):
            for j in range(64): HL[0,0,i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]])
        for t in range(20):
            Et=ERR[fitm][:,t][...,ci]; Mt=SCM[fitm][:,t][...,ci]
            for i in range(32):
                for j in range(64): HTL[0,t,i,j,ci]=best_h(Et[:,i,j][Mt[:,i,j]])
    return HL,HTL
def sc(H,evm,ERR):
    h=np.broadcast_to(H,ERR[evm].shape); ok=(ERR[evm]<=h)&SCM[evm]
    E=float((np.exp(-2*h/SIG)*ok).sum()/SCM[evm].sum()); cov=float(ok.sum()/SCM[evm].sum())
    return E,cov,77.20+fin(100*WR*E)-BASE
A=np.isin(wt,list(set(vidx[0::2]))); Bm=~A
out={}
for tag,fitm,evm in [("fit A -> score B",A,Bm),("fit B -> score A",Bm,A)]:
    al,be=calib(fitm); ERR=EL*(al+be*AP)
    HL,HTL=build(fitm,ERR)
    print("\n%s   (alpha %.2f beta %.2f)"%(tag,al,be),flush=True)
    print("  %-30s %8s %8s %9s"%("policy","E","coverage","EST REAL"))
    for nm,Hb in [("per-location",HL),("per-timestep x location",HTL)]:
        for m_ in [1.00,1.10,1.20]:
            E,cv,f=sc(Hb*m_,evm,ERR)
            print("  %-30s %8.4f %8.3f %9.2f"%("%s x%.2f"%(nm,m_),E,cv,f),flush=True)
            out.setdefault("%s x%.2f"%(nm,m_),[]).append(f)
print("\n=== stability across both split directions ===")
print("  %-30s %8s %8s %8s"%("policy","A->B","B->A","spread"))
for k,v in sorted(out.items(),key=lambda x:-np.mean(x[1])):
    print("  %-30s %8.2f %8.2f %8.2f"%(k,v[0],v[1],abs(v[0]-v[1])))
