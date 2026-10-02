"""Close more of the bounds gap. Per-location captures WHERE error lives, but not
WHICH WINDOWS are hard. Both extra signals below are computable from the INPUT
at inference time, so they are legal in submission.

  A. per-location (current best)                          -> reference
  B. + window-level multiplier from the INPUT's temporal variance
  C. + per-element multiplier from the INPUT's LOCAL temporal variance (deciles)
  D. B and C combined
Fit on half the held-out trajectories, scored on the other half.
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
vidx=sorted(set(range(0,len(lens),5)))
wins=[];wt=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0); wt.append(i)
wins=np.array(wins); wt=np.array(wt)
Wall=[]
for s in wins:
    w=np.asarray(X[s:s+40]); z=np.zeros(w.shape[:-1]+(1,),np.float32)
    Wall.append(np.concatenate([w,z],-1))
Wall=np.stack(Wall).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
def run(ck):
    m,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
    if ck: m.load_state_dict(torch.load(ck,map_location=DEV))
    m=m.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
P0=run(None); e0=np.abs(P0[...,:C]-Y[...,:C])
def E_at(e,lam):
    eu=e[...,0][SCM[...,0]]/lam; ev=e[...,1][SCM[...,1]]/lam
    return float((np.exp(-2*0.030/SIG)*(eu<=0.030)).sum()+(np.exp(-2*0.010/SIG)*(ev<=0.010)).sum())/(eu.size+ev.size)
lo,hi=0.02,20.0
for _ in range(60):
    mid=(lo+hi)/2
    if E_at(e0,mid)>0.4399: hi=mid
    else: lo=mid
LAM=1/((lo+hi)/2)
print("lambda: x%.3f (check E=%.4f == 0.4399)"%(LAM,E_at(e0,(lo+hi)/2)),flush=True)
P=run(f"{B}/local_harness/ft_w015_best.pth")
ERR=np.abs(P[...,:C]-Y[...,:C])*LAM
half=set(vidx[0::2]); fit=np.isin(wt,list(half)); ev=~fit
print("fit %d / score %d windows"%(fit.sum(),ev.sum()),flush=True)

# --- INPUT-derived signals (legal at inference) ---
iu,iv=Xin[...,0],Xin[...,1]
win_tv=np.sqrt(0.5*(iu.var(axis=1).mean(axis=(1,2))+iv.var(axis=1).mean(axis=(1,2))))   # (N,)
loc_tv=np.sqrt(0.5*(iu.var(axis=1)+iv.var(axis=1)))                                      # (N,32,64)
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
def sc(H,m):
    e=ERR[m]; s=SCM[m]; ok=(e<=H)&s
    return float((np.exp(-2*H/SIG)*ok).sum()/s.sum()), float(ok.sum()/s.sum())
base=np.broadcast_to(HL[None,None],ERR.shape)
res={}
res["A per-location"]=sc(base[ev],ev)
# B: window-level multiplier by decile of win_tv
qw=np.quantile(win_tv[fit],np.linspace(0,1,11)[1:-1]); wb=np.digitize(win_tv,qw)
mw=np.ones((10,C),np.float32)
for ci in range(C):
    for b_ in range(10):
        sel=(wb[fit]==b_)
        if sel.sum()<3: continue
        e=ERR[fit][sel][...,ci][SCM[fit][sel][...,ci]]; bh=base[fit][sel][...,ci][SCM[fit][sel][...,ci]]
        best=(1.0,-1.0)
        for m_ in np.linspace(0.4,2.5,22):
            h=bh*m_; val=float((np.exp(-2*h/SIG)*(e<=h)).mean())
            if val>best[1]: best=(m_,val)
        mw[b_,ci]=best[0]
HB=base.copy()
for ci in range(C): HB[...,ci]=base[...,ci]*mw[wb,ci][:,None,None,None]
res["B + window-tv mult"]=sc(HB[ev],ev)
# C: per-element multiplier by decile of local input temporal variance
ql=np.quantile(loc_tv[fit][::7],np.linspace(0,1,11)[1:-1]); lb=np.digitize(loc_tv,ql)  # (N,32,64)
lb4=np.broadcast_to(lb[:,None,:,:],ERR.shape[:4])
ml=np.ones((10,C),np.float32)
for ci in range(C):
    for b_ in range(10):
        sel=(lb4[fit]==b_)&SCM[fit][...,ci]
        if sel.sum()<2000: continue
        e=ERR[fit][...,ci][sel]; bh=base[fit][...,ci][sel]
        best=(1.0,-1.0)
        for m_ in np.linspace(0.4,2.5,22):
            h=bh*m_; val=float((np.exp(-2*h/SIG)*(e<=h)).mean())
            if val>best[1]: best=(m_,val)
        ml[b_,ci]=best[0]
HC=base.copy()
for ci in range(C): HC[...,ci]=base[...,ci]*ml[lb4,ci]
res["C + local-tv mult"]=sc(HC[ev],ev)
HD=base.copy()
for ci in range(C): HD[...,ci]=base[...,ci]*ml[lb4,ci]*mw[wb,ci][:,None,None,None]
res["D both"]=sc(HD[ev],ev)
eo=ERR[ev][SCM[ev]]; Eor=float(np.mean(np.exp(-2*eo/SIG)))
print("\n=== TEST half (fine-tuned model) ===")
print("  %-22s %8s %9s %10s %8s"%("policy","E","coverage","real sps","%oracle"))
for k,(E,cv) in res.items():
    print("  %-22s %8.4f %9.3f %10.2f %7.0f%%"%(k,E,cv,100*0.7716*E,100*E/Eor))
print("  %-22s %8.4f %9.3f %10.2f %7.0f%%"%("ORACLE",Eor,1.0,100*0.7716*Eor,100))
print("\n  window-tv multipliers (u):",np.round(mw[:,0],2))
print("  local-tv  multipliers (u):",np.round(ml[:,0],2))
np.savez(f"{B}/local_harness/bounds2.npz",HL=HL,mw=mw,ml=ml,qw=qw,ql=ql)
