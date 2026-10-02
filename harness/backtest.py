"""BACKTEST: run the EXACT pipeline used to project the soup candidate against
the config we KNOW scored 77.20 on the real leaderboard, on the SAME held-out
trajectories used to validate the soup candidate. This is the assurance check.

Honesty notes, stated explicitly in the output:
  - rel_l2/tke/mvpe are DIRECT measurements, not calibrated -> genuine test.
  - lambda (the SPS error-scale) was fit using this exact model+bounds as ONE of
    its two anchors, so sps for THIS config is partially circular. The SECOND
    anchor (proportional bounds, sps=14.08) was NOT used to fit lambda and is
    reported separately as the true independent check.
  - The final-score FORMULA was fit externally on 257 real submissions, not by
    us, so plugging OUR measured subscores into it is a fair test of whether
    OUR subscores are right.
  - Whatever residual this backtest shows gets applied as a correction to the
    soup projection, not ignored.
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
vidx=sorted(set(range(0,len(lens),5)))          # the SAME 17 held-out trajectories used for soup validation
wins=[]
for i in vidx:
    for t0 in range(off[i],off[i]+lens[i]-39,20): wins.append(t0)
Wall=np.stack([np.concatenate([np.asarray(X[s:s+40]),
        np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
Xin,Y=Wall[:,:IN],Wall[:,IN:]; SCM=(Y[...,:C]!=0.0)
print("held-out trajectories: %d  windows: %d"%(len(vidx),len(wins)),flush=True)

def run(ckpt_path, sd=None):
    m,_=load_baseline(ckpt_path,device=DEV)
    if sd is not None: m.load_state_dict(sd)
    m=m.to(DEV).eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy())
    return np.concatenate(o,0).astype(np.float32)

n=lambda x:x/(0.5+x)
def W_of(rl,tk,mv):
    e=lambda s:(100.0/s-1.0)*2.0
    return 0.5*(1-n(e(rl)))+0.3*(1-n(e(tk)))+0.2*(1-n(e(mv)))
def formula(rl,tk,mv,tm,sps):
    return 0.306*rl+0.163*tk+0.218*mv+0.100*tm+0.217*sps

# ============================================================
# 1. THE KNOWN 77.20 CONFIG: original checkpoint, bounds [0.030,0.010]
# ============================================================
ORIG_SD=torch.load(f"{B}/data/comp_real/sim_real_fno.pth",map_location=DEV)
ORIG_SD=ORIG_SD.get("model_state_dict",ORIG_SD)
P0=run(f"{B}/data/comp_real/sim_real_fno.pth", ORIG_SD)
dm=S.rel_l2_per_sample(P0,Y,C); tk=S.tke_rel_l2_per_sample(P0,Y,C); mv=S.mvpe_rel_l2_per_sample(P0,Y)
rl_m,tk_m,mv_m=S.score_error(float(dm.mean())),S.score_error(float(tk.mean())),S.score_error(float(mv.mean()))
print("\n=== 1. ACCURACY subscores, DIRECT measurement (no calibration involved) ===")
print("  measured on held-out train_real:  rel_l2 %.2f   tke %.2f   mvpe %.2f"%(rl_m,tk_m,mv_m))
print("  REAL leaderboard (known-true):    rel_l2 94.17   tke 74.03   mvpe 92.84")
print("  bias:                             rel_l2 %+.2f   tke %+.2f   mvpe %+.2f"%(rl_m-94.168,tk_m-74.026,mv_m-92.836))
print("  (train_real is this model's OWN fine-tuning data -> expect a small optimistic bias)")

err=np.abs(P0[...,:C]-Y[...,:C])
def E_const(lam): 
    eu=err[...,0][SCM[...,0]]/lam; ev=err[...,1][SCM[...,1]]/lam
    return float((np.exp(-2*0.030/SIG)*(eu<=0.030)).sum()+(np.exp(-2*0.010/SIG)*(ev<=0.010)).sum())/(eu.size+ev.size)
lo,hi=0.02,20.0
for _ in range(60):
    mid=(lo+hi)/2
    if E_const(mid)>0.4399: hi=mid
    else: lo=mid
LAM=(lo+hi)/2
print("\n=== 2. SPS calibration (lambda fit to match THIS anchor -> partially circular for this config) ===")
print("  lambda=%.3f  ->  E_const=%.4f (target 0.4399)  sps=%.2f (target 29.84)"%(
    LAM,E_const(LAM),100*0.6784*E_const(LAM)))

def E_prop(lam):
    absP=np.abs(P0[...,:C]); hu=0.05*absP[...,0]; hv=0.05*absP[...,1]
    eu=err[...,0][SCM[...,0]]/lam; ev=err[...,1][SCM[...,1]]/lam
    hu=hu[SCM[...,0]]/lam*lam  # h itself not rescaled by lam (bound is fixed by us); only error is
    hu=(0.05*absP[...,0])[SCM[...,0]]; hv=(0.05*absP[...,1])[SCM[...,1]]
    return float((np.exp(-2*hu/SIG)*(eu<=hu)).sum()+(np.exp(-2*hv/SIG)*(ev<=hv)).sum())/(eu.size+ev.size)
Ep=E_prop(LAM)
print("\n=== 3. INDEPENDENT check: the SECOND real anchor (NOT used to fit lambda) ===")
print("  proportional bounds 0.05*|pred|:  E=%.4f  sps=%.2f   REAL: E=0.2076  sps=14.08"%(
    Ep,100*0.6784*Ep))
print("  residual: E %+.4f  (this is the true, non-circular error bar on our sps estimates)"%(Ep-0.2076))

# ============================================================
# 2. Full pipeline prediction for the KNOWN 77.20 config
# ============================================================
sps_pred=100*W_of(rl_m,tk_m,mv_m)*E_const(LAM)
final_pred=formula(rl_m,tk_m,mv_m,91.32,sps_pred)
print("\n=== 4. BACKTEST: full pipeline prediction vs the REAL known score ===")
print("  predicted subscores -> formula:  rel_l2 %.2f tke %.2f mvpe %.2f sps %.2f"%(rl_m,tk_m,mv_m,sps_pred))
print("  PREDICTED final:  %.2f"%final_pred)
print("  ACTUAL   final:   77.20")
BIAS = final_pred - 77.20
print("  BACKTEST RESIDUAL: %+.2f  <-- apply this as a correction to any other prediction today"%BIAS)

np.save(f"{B}/local_harness/backtest_bias.npy", np.array([BIAS]))
np.save(f"{B}/local_harness/backtest_lambda.npy", np.array([LAM]))
print("\nsaved backtest_bias.npy, backtest_lambda.npy",flush=True)
