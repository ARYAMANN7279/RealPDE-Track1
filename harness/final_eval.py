"""Honest end-to-end evaluation of the fine-tuned model + bounds.

Discipline:
  * Only the fine-tune's VAL trajectories (every 5th) are used -- the model
    never trained on them.
  * Those are split again: half to FIT bounds/gain, half to SCORE. No leakage.
  * lambda is re-derived ON THIS SUBSET so that the ORIGINAL model reproduces
    its real anchor E=0.4399 at [0.030,0.010]; the same lambda is then applied
    to the fine-tuned model. This keeps the local->real correction anchored to
    live leaderboard data.
"""
import json, os, sys
import numpy as np, torch
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda:1"; C=2; IN=20
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
mi,si,mt,st=[torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
X=np.load(f"{B}/local_harness/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{B}/local_harness/tr_meta.json")); off=meta["off"]; lens=meta["lens"]
ntraj=len(lens); vidx=sorted(set(range(0,ntraj,5)))
print("val trajectories (never trained on): %d"%len(vidx),flush=True)
wins=[];wtraj=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20):
        wins.append(t0); wtraj.append(i)
wins=np.array(wins); wtraj=np.array(wtraj)
print("val windows: %d"%len(wins),flush=True)
def run(ckpt):
    m,_=load_baseline(ckpt,device=DEV) if ckpt.endswith("sim_real_fno.pth") else (None,None)
    if m is None:
        m,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
        m.load_state_dict(torch.load(ckpt,map_location=DEV))
    m=m.to(DEV).eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(wins),32):
            w=np.stack([np.asarray(X[s:s+40]) for s in wins[i:i+32]])
            z=np.zeros(w.shape[:-1]+(1,),np.float32); w=np.concatenate([w,z],-1)
            x=torch.from_numpy(w[:,:IN]).to(DEV)
            out.append((m((x-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(out,0).astype(np.float32)
Yw=[]
for i in range(0,len(wins),32):
    w=np.stack([np.asarray(X[s:s+40]) for s in wins[i:i+32]])
    z=np.zeros(w.shape[:-1]+(1,),np.float32); Yw.append(np.concatenate([w,z],-1)[:,IN:])
Y=np.concatenate(Yw,0).astype(np.float32)
n_=lambda x:x/(0.5+x)
def subs(P,m):
    dm=S.rel_l2_per_sample(P[m],Y[m],C);tk=S.tke_rel_l2_per_sample(P[m],Y[m],C);mv=S.mvpe_rel_l2_per_sample(P[m],Y[m])
    return S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean()))
half=set(vidx[0::2]); fit=np.isin(wtraj,list(half)); ev=~fit
print("fit windows %d | score windows %d"%(fit.sum(),ev.sum()),flush=True)
SCM=(Y[...,:C]!=0.0)
P0=run(f"{B}/data/comp_real/sim_real_fno.pth")
print("\nORIGINAL model on score-half: rel_l2 %.2f tke %.2f mvpe %.2f"%subs(P0,ev),flush=True)
# lambda from the original model's real anchor on THIS subset
e0=np.abs(P0[...,:C]-Y[...,:C])
def E_at(e,lam,hu=0.030,hv=0.010,m=None):
    eu=e[m][...,0][SCM[m][...,0]]/lam; ev_=e[m][...,1][SCM[m][...,1]]/lam
    return float((np.exp(-2*hu/SIG)*(eu<=hu)).sum()+ (np.exp(-2*hv/SIG)*(ev_<=hv)).sum())/(eu.size+ev_.size)
lo,hi=0.02,20.0
for _ in range(60):
    mid=(lo+hi)/2
    if E_at(e0,mid,m=ev)>0.4399: hi=mid   # E rises with lam => need SMALLER lam
    else: lo=mid
mid=(lo+hi)/2
assert 0.03<mid<19.0, "lambda bisection saturated at %.3f"%mid
LAM=1/mid
print("  lambda (this subset, anchored to real E=0.4399): errors x %.3f"%(1/LAM),flush=True)
res={}
cands=[("original",f"{B}/data/comp_real/sim_real_fno.pth"),
       ("ft w0.05",f"{B}/local_harness/ft_w005_best.pth"),
       ("ft w0.10",f"{B}/local_harness/ft_w010_best.pth"),
       ("ft w0.15",f"{B}/local_harness/ft_w015_best.pth"),
       ("ft w0.33",f"{B}/local_harness/ft_lr1e5_best.pth"),
       ("ft w0.60",f"{B}/local_harness/ft_w060_best.pth"),
       ("ft lr3e-5",f"{B}/local_harness/ft_lr3e5_best.pth")]
cands=[(t,c) for t,c in cands if os.path.exists(c)]
for tag,ck in cands:
    P=P0 if tag=="original" else run(ck)
    acc=subs(P,ev)
    ERR=np.abs(P[...,:C]-Y[...,:C])*LAM      # LAM=1/mid : inflate local errors to real scale
    def best_h(v):
        v=np.sort(v)
        if v.size==0: return 0.0
        k=np.arange(1,v.size+1)/v.size
        return float(v[np.argmax(np.exp(-2*v/SIG)*k)])
    HL=np.zeros((32,64,C),np.float32)
    for ci in range(C):
        E=ERR[fit][...,ci]; M=SCM[fit][...,ci]
        for i in range(32):
            for j in range(64): HL[i,j,ci]=best_h(E[:,:,i,j][M[:,:,i,j]])
    def sc_(H):
        e=ERR[ev]; s=SCM[ev]; h=np.broadcast_to(H[None,None],e.shape)
        ok=(e<=h)&s; return float((np.exp(-2*h/SIG)*ok).sum()/s.sum())
    Hb=np.zeros((32,64,C),np.float32); Hb[...,0]=0.030; Hb[...,1]=0.010
    E_base,E_loc=sc_(Hb),sc_(HL)
    if tag=="original":
        assert abs(E_base-0.4399)<0.01, "CALIBRATION BROKEN: original E_base=%.4f, must be 0.4399"%E_base
        print("  [check] original E_base=%.4f == real anchor 0.4399 OK"%E_base,flush=True)
    W=0.5*(1-n_((100/acc[0]-1)*2))+0.3*(1-n_((100/acc[1]-1)*2))+0.2*(1-n_((100/acc[2]-1)*2))
    res[tag]=(acc,E_base,E_loc,W,HL)
    print("  %-10s rel_l2 %.2f tke %.2f mvpe %.2f | E_base %.4f E_loc %.4f | W %.4f"%(
        tag,acc[0],acc[1],acc[2],E_base,E_loc,W),flush=True)
RL,TK,MV,TM=94.168150,74.025866,92.836278,91.32
fin=lambda rl,tk,mv,sps:0.306*rl+0.163*tk+0.218*mv+0.100*TM+0.217*sps
o=res["original"]; base_f=fin(RL,TK,MV,29.84)
print("\n=== projected (deltas from original; the original+current row MUST be ~0.00) ===")
for tag,(acc,Eb,El,W,HL) in res.items():
    d=[acc[k]-o[0][k] for k in range(3)]
    rl,tk,mv=RL+d[0],TK+d[1],MV+d[2]
    Wn=0.5*(1-n_((100/rl-1)*2))+0.3*(1-n_((100/tk-1)*2))+0.2*(1-n_((100/mv-1)*2))
    for bn,E in [("current bounds",Eb),("per-loc bounds",El)]:
        sps=100*Wn*E*(o[1]and 1)  # E already absolute
        sps=100*Wn*E
        print("  %-10s + %-15s rel_l2 %.2f tke %.2f mvpe %.2f sps %.2f -> %+.2f final"%(
            tag,bn,rl,tk,mv,sps,fin(rl,tk,mv,sps)-base_f))
np.savez(f"{B}/local_harness/final_maps.npz",lam=np.array([LAM]),
         **{("hl_"+t.replace(" ","_").replace(".","")):r[4] for t,r in res.items()})
print("\nsaved final_maps.npz",flush=True)
