"""Can a LOW-CAPACITY per-pixel bound predictor capture the FNO's SPS headroom
(constant 24 -> oracle 38.7 on proxy) and GENERALIZE across trajectories?
Ridge regression on interpretable local features -> predicted |error| -> bound.
Honest tune/test trajectory split. Bounds-only => zero accuracy risk.
"""
import importlib.util, os, json, numpy as np
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness";KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
P=np.load(f"{H}/real_eval_v3/fno_tta_cache.npz")["P1"].astype(np.float32)
Y=np.load(f"{H}/real_eval_v3/targets.npz")["target"];meta=json.load(open(f"{H}/real_eval_v3/meta.json"))
sims=np.array([m["sim_id"] for m in meta]);C=2
N,T,Hh,W,_=P.shape

def feats(pred):
    # per-pixel interpretable features predictive of |error|, per channel
    u,v=pred[...,0],pred[...,1]
    gux=np.abs(np.gradient(u,axis=3)); guy=np.abs(np.gradient(u,axis=2))
    gvx=np.abs(np.gradient(v,axis=3)); gvy=np.abs(np.gradient(v,axis=2))
    gmag=np.sqrt(gux**2+guy**2+gvx**2+gvy**2)          # local complexity (wake)
    vort=np.abs(np.gradient(v,axis=3)-np.gradient(u,axis=2))
    yy,xx=np.meshgrid(np.linspace(-1,1,Hh),np.linspace(0,1,W),indexing="ij")
    rad=np.sqrt(xx**2+yy**2)[None,None]*np.ones((N,T,1,1))
    tfrac=np.linspace(0,1,T)[None,:,None,None]*np.ones((N,1,Hh,W))
    ke=0.5*(u**2+v**2)
    F=np.stack([gmag,vort,np.abs(u),np.abs(v),ke,tfrac,np.broadcast_to(rad,u.shape),np.ones_like(u)],axis=-1)
    return F.reshape(-1,F.shape[-1])

X=feats(P)                                   # (Npix, nfeat)
err=np.abs(Y[...,:C]-P[...,:C]).reshape(-1,C)
mask_flat=np.repeat((sims[:,None]),T*Hh*W).reshape(-1)  # sim per pixel
uniq=sorted(set(sims.tolist())); tune_tr=set(uniq[::2])
tune=np.array([s in tune_tr for s in mask_flat]); test=~tune

def ridge_fit(Xt,yt,lam=10.0):
    A=Xt.T@Xt+lam*np.eye(Xt.shape[1]); return np.linalg.solve(A,Xt.T@yt)

# fit per channel: predict |error| from features on TUNE
W_ch=[ridge_fit(X[tune],err[tune,ch]) for ch in range(C)]
pred_err=np.stack([np.clip(X@W_ch[ch],1e-4,None) for ch in range(C)],axis=-1)  # (Npix,C)

def sps_from_halfflat(hu_flat,hv_flat,sel):
    # build per-pixel half array (N,T,H,W,3) then score on selected windows
    half=np.zeros((N*T*Hh*W,3),np.float32); half[:,0]=hu_flat; half[:,1]=hv_flat
    half=half.reshape(N,T,Hh,W,3)
    win_sel=np.array([s in sel for s in sims])
    s,cov=scoring.aggregate_sps(P[win_sel],Y[win_sel],C,lower=P[win_sel]-half[win_sel],upper=P[win_sel]+half[win_sel])
    return scoring.score_sps(s),cov*100

# baselines on TEST windows
test_tr=set(uniq[1::2])
hc=np.full(N*T*Hh*W,0.030,np.float32); hcv=np.full(N*T*Hh*W,0.010,np.float32)
print(f"constant [0.030,0.010]  TEST: SPS={sps_from_halfflat(hc,hcv,test_tr)[0]:.2f}")
# ridge per-pixel bound = k * predicted_error; tune k on TUNE, eval on TEST
best=None
for k in np.linspace(0.8,3.0,23):
    su,_=sps_from_halfflat(k*pred_err[:,0],k*pred_err[:,1],tune_tr)
    if best is None or su>best[0]: best=(su,k)
k=best[1]
st,ct=sps_from_halfflat(k*pred_err[:,0],k*pred_err[:,1],test_tr)
print(f"ridge per-pixel bound   TEST: SPS={st:.2f} cov={ct:.0f}%  (k={k:.2f}, tuned on tune)")
# oracle on TEST for reference
oer=np.abs(Y[...,:C]-P[...,:C]).reshape(-1,C)+1e-4
print(f"oracle (|err|)          TEST: SPS={sps_from_halfflat(oer[:,0],oer[:,1],test_tr)[0]:.2f}")
