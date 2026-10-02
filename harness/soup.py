"""Model soup: average the weights of fine-tunes that share an initialization.
All runs started from sim_real_fno.pth, so averaging is valid. Often improves
both accuracy terms at once, which is exactly what our single fine-tunes fail to
do (they trade rel_l2 for tke).

Scored on the same held-out trajectories, with the same lambda calibration and
the +0.00 assertion on the original model.
"""
import json, os, sys, itertools
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
wins=[];wt=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0); wt.append(i)
wins=np.array(wins); wt=np.array(wt)
Wall=np.stack([np.concatenate([np.asarray(X[s:s+40]),
        np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
half=set(vidx[0::2]); fit=np.isin(wt,list(half)); ev=~fit
base_model,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
def run_sd(sd):
    m=base_model
    if sd is not None: m.load_state_dict(sd)
    else: m.load_state_dict(torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV).get(
        "model_state_dict", torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV)))
    m=m.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
ORIG=torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV)
ORIG=ORIG.get("model_state_dict",ORIG)
P0=run_sd(ORIG); e0=np.abs(P0[...,:C]-Y[...,:C])
def E_at(e,lam,m):
    eu=e[m][...,0][SCM[m][...,0]]/lam; ev_=e[m][...,1][SCM[m][...,1]]/lam
    return float((np.exp(-2*0.030/SIG)*(eu<=0.030)).sum()+(np.exp(-2*0.010/SIG)*(ev_<=0.010)).sum())/(eu.size+ev_.size)
lo,hi=0.02,20.0
for _ in range(60):
    mid=(lo+hi)/2
    if E_at(e0,mid,ev)>0.4399: hi=mid
    else: lo=mid
LAM=1/((lo+hi)/2)
assert abs(E_at(e0,1/LAM,ev)-0.4399)<0.01
print("lambda x%.3f  [assertion OK]"%LAM,flush=True)
def best_h(v):
    v=np.sort(v)
    if v.size==0: return 0.0
    k=np.arange(1,v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
def evaluate(P,tag):
    dm=S.rel_l2_per_sample(P[ev],Y[ev],C);tk=S.tke_rel_l2_per_sample(P[ev],Y[ev],C);mv=S.mvpe_rel_l2_per_sample(P[ev],Y[ev])
    acc=(S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean())))
    ERR=np.abs(P[...,:C]-Y[...,:C])*LAM
    HL=np.zeros((32,64,C),np.float32)
    for ci in range(C):
        E=ERR[fit][...,ci]; M=SCM[fit][...,ci]
        for i in range(32):
            for j in range(64): HL[i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]])
    h=np.broadcast_to(HL[None,None],ERR[ev].shape); ok=(ERR[ev]<=h)&SCM[ev]
    E=float((np.exp(-2*h/SIG)*ok).sum()/SCM[ev].sum())
    return acc,E,HL
n_=lambda x:x/(0.5+x)
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
fin=lambda rl,tk,mv,sps:0.306*rl+0.163*tk+0.218*mv+0.100*TM+0.217*sps
bf=fin(RL,TK,MV,29.84)
oacc,oE,_=evaluate(P0,"original")
print("original: rel_l2 %.2f tke %.2f mvpe %.2f E %.4f"%(oacc[0],oacc[1],oacc[2],oE),flush=True)
CK={t:f"{B}/local_harness/ft_{t}_best.pth" for t in ["w005","w010","w015","lr1e5","w060","lr3e5"]}
CK={k:v for k,v in CK.items() if os.path.exists(v)}
recipes={"soup w005+w010+w015":["w005","w010","w015"],
         "soup all 6":list(CK.keys()),
         "soup w010+w015":["w010","w015"]}
print("\n  %-24s %7s %7s %7s %8s %10s"%("model","rel_l2","tke","mvpe","E","d final"))
def report(tag,acc,E):
    d=[acc[k]-oacc[k] for k in range(3)]
    rl,tk,mv=RL+d[0],TK+d[1],MV+d[2]
    Wn=0.5*(1-n_((100/rl-1)*2))+0.3*(1-n_((100/tk-1)*2))+0.2*(1-n_((100/mv-1)*2))
    sps=100*Wn*E
    print("  %-24s %7.2f %7.2f %7.2f %8.4f %+10.2f"%(tag,acc[0],acc[1],acc[2],E,fin(rl,tk,mv,sps)-bf),flush=True)
    return fin(rl,tk,mv,sps)-bf
acc,E,_=evaluate(run_sd(torch.load(CK["w015"],map_location=DEV)),"w015"); report("single w015",acc,E)
best=(None,-9,None)
for name,keys in recipes.items():
    sds=[torch.load(CK[k],map_location=DEV) for k in keys if k in CK]
    if len(sds)<2: continue
    avg={}
    for k in sds[0]:
        v0=sds[0][k]
        # complex spectral weights must NOT go through .float() -- that drops the
        # imaginary part and silently destroys the model
        if torch.is_tensor(v0) and (v0.is_complex() or torch.is_floating_point(v0)):
            avg[k]=(sum(sd[k] for sd in sds)/len(sds)).to(v0.dtype)
        else:
            avg[k]=v0
    ncplx=sum(1 for k in avg if torch.is_tensor(avg[k]) and avg[k].is_complex())
    dmax=max(float((avg[k].float()-sds[0][k].float()).abs().max()) if not avg[k].is_complex()
             else float((avg[k]-sds[0][k]).abs().max())
             for k in avg if torch.is_tensor(avg[k]) and (avg[k].is_complex() or torch.is_floating_point(avg[k])))
    print("    [%s: %d complex tensors preserved, max|soup-member| %.2e]"%(name,ncplx,dmax),flush=True)
    acc,E,HL=evaluate(run_sd(avg),name); d=report(name,acc,E)
    if d>best[1]: best=(name,d,(avg,HL))
if best[0]:
    torch.save(best[2][0],f"{B}/local_harness/soup_best.pth")
    np.save(f"{B}/local_harness/soup_bounds.npy",best[2][1])
    print("\nbest: %s (%+.2f) -> soup_best.pth + soup_bounds.npy"%(best[0],best[1]),flush=True)
