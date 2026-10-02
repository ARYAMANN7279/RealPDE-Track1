"""THE ANCHOR TEST on the real competition data (train_real, 64x128 H5).

Kit convention: raw PIV is 64x128, eval subsamples by 2 -> 32x64.
Window = 20 in / 20 out, non-overlapping.

REAL leaderboard anchors for this exact checkpoint:
    rel_l2 94.168   tke 74.026   mvpe 92.836   coverage@[0.030,0.010] 0.841
Caveat: train_real is FINE-TUNING data, so scores may be optimistic vs the
hidden eval. Direction and magnitude are what matter here.
"""
import glob, os, sys, json
import numpy as np, torch, h5py
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
TR=f"{B}/data/comp_real/train_real"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
import importlib.util
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG=S.SIGMA_GLOBAL; DEV="cuda:3"; C=2; IN=20; HOR=40
MI=np.array([0.154960856,-0.000513992854,0.0],np.float32); SI=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32); ST=np.array([0.0968104079,0.0159636438,1.0],np.float32)
model=load_baseline(f"{B}/local_harness/fno_model/sim_real_fno_fp16.pth",device=DEV)[0]; model.eval()
mi,si,mt,st=[torch.tensor(a).to(DEV) for a in (MI,SI,MT,ST)]
@torch.no_grad()
def run(A):
    o=[]
    for i in range(0,len(A),64):
        xb=torch.from_numpy(A[i:i+64]).to(DEV).float()
        o.append((model((xb-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)
n=lambda x:x/(0.5+x)
files=sorted(glob.glob(f"{TR}/*.h5"))
print("trajectories: %d"%len(files),flush=True)
rows=[]; gP=[]; gY=[]
for fp in files:
    sid=os.path.basename(fp)
    try:
        with h5py.File(fp,"r") as f:
            u=np.array(f["u"],dtype=np.float32)[:,::2,::2][:,:32,:64]
            v=np.array(f["v"],dtype=np.float32)[:,::2,::2][:,:32,:64]
    except Exception as ex:
        print("  SKIP %s (%s)"%(sid,ex),flush=True); continue
    xs,ys=[],[]
    for k in range((u.shape[0]-HOR)//HOR+1):
        t0=k*HOR; tr=np.stack([u[t0:t0+HOR],v[t0:t0+HOR],np.zeros_like(u[t0:t0+HOR])],-1)
        xs.append(tr[:IN]); ys.append(tr[IN:])
    if not xs: continue
    X=np.stack(xs); Y=np.stack(ys); P=run(X)
    dm=S.rel_l2_per_sample(P,Y,C); tk=S.tke_rel_l2_per_sample(P,Y,C); mv=S.mvpe_rel_l2_per_sample(P,Y)
    e=np.abs(P[...,:C]-Y[...,:C]); sc=(Y[...,:C]!=0.0)
    cov=0.5*(float((e[...,0]<=0.030)[sc[...,0]].mean())+float((e[...,1]<=0.010)[sc[...,1]].mean()))
    rows.append(dict(sim=sid,n=len(X),dm=float(dm.mean()),tk=float(tk.mean()),mv=float(mv.mean()),
                     cov=cov,zerofrac=float((~sc).mean())))
    gP.append(P.astype(np.float16)); gY.append(Y.astype(np.float16))
    print("  %-14s n=%2d zeros=%.3f | rl %6.2f tke %6.2f mvpe %6.2f cov %.3f"%(
        sid,len(X),rows[-1]["zerofrac"],S.score_error(rows[-1]["dm"]),S.score_error(rows[-1]["tk"]),
        S.score_error(rows[-1]["mv"]),cov),flush=True)
    del u,v,X,Y,P
N=sum(r["n"] for r in rows)
dm=sum(r["dm"]*r["n"] for r in rows)/N; tk=sum(r["tk"]*r["n"] for r in rows)/N
mv=sum(r["mv"]*r["n"] for r in rows)/N; cov=sum(r["cov"]*r["n"] for r in rows)/N
print("\n"+"="*74)
print("AGGREGATE  n=%d windows over %d trajectories"%(N,len(rows)))
print("            rel_l2   tke    mvpe    coverage")
print("  measured  %6.2f  %6.2f  %6.2f    %.3f"%(S.score_error(dm),S.score_error(tk),S.score_error(mv),cov))
print("  ANCHOR    %6.2f  %6.2f  %6.2f    %.3f"%(94.168,74.026,92.836,0.841))
print("  delta     %+6.2f  %+6.2f  %+6.2f    %+.3f"%(
    S.score_error(dm)-94.168,S.score_error(tk)-74.026,S.score_error(mv)-92.836,cov-0.841))
print("="*74)
json.dump(rows,open(f"{B}/local_harness/comp_anchor_rows.json","w"),indent=1)
np.savez_compressed(f"{B}/local_harness/comp_eval_cache.npz",P=np.concatenate(gP,0),Y=np.concatenate(gY,0))
print("cached -> comp_eval_cache.npz",flush=True)
