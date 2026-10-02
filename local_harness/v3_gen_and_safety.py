import importlib.util, os, sys, json, numpy as np, torch
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
E=f"{H}/real_eval_v3"
sys.path.insert(0,KIT);sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
mi=torch.tensor([0.154960856,-0.000513992854,0.0]);si=torch.tensor([0.0968056545,0.015960684,1.0])
mt=torch.tensor([0.154962569,-0.000517793698,0.0]);st=torch.tensor([0.0968104079,0.0159636438,1.0])
X=np.load(f"{E}/inputs.npz")["input"];Y=np.load(f"{E}/targets.npz")["target"];meta=json.load(open(f"{E}/meta.json"));sims=np.array([m["sim_id"] for m in meta]);C=2
CACHE=f"{E}/fno_tta_cache.npz"
if not os.path.exists(CACHE):
    m=load_baseline(f"{H}/fno_model/sim_real_fno_fp16.pth",device="cpu")[0]
    P1,P2=[],[]
    with torch.no_grad():
        for i in range(0,X.shape[0],16):
            xb=torch.from_numpy(X[i:i+16]).float()
            y1=(m((xb-mi)/si)*st+mt).numpy()
            x2=torch.flip(xb,dims=[2]).clone();x2[...,1]=-x2[...,1]
            y2=m((x2-mi)/si)*st+mt;y2=torch.flip(y2,dims=[2]).clone();y2[...,1]=-y2[...,1]
            P1.append(y1);P2.append(y2.numpy())
    np.savez(CACHE,P1=np.concatenate(P1,0).astype(np.float32),P2=np.concatenate(P2,0).astype(np.float32));print("cached",flush=True)
d=np.load(CACHE);pred0=(0.5*(d["P1"]+d["P2"])).astype(np.float32)
def mov(a,w):
    if w<=1:return a
    k=w//2;ap=np.pad(a,((0,0),(k,k),(0,0),(0,0),(0,0)),mode="edge");o=np.zeros_like(a)
    for i in range(w):o+=ap[:,i:i+a.shape[1]]
    return (o/w).astype(np.float32)
# per-traj safety: does smoothing ever HURT L2 or TKE on the unsteady ones?
unst={s:float(np.mean(Y[sims==s][...,:2].std(axis=1))) for s in set(sims.tolist())}
print(f"\n{'trajectory (unsteady)':26s}| L2 @ w=1,3,5,7,9        | TKE @ w=1,3,5,7,9")
for s in sorted(set(sims.tolist()),key=lambda x:-unst[x]):
    mk=sims==s
    L=[scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(mov(pred0[mk],w),Y[mk],C)))) for w in (1,3,5,7,9)]
    T=[scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(mov(pred0[mk],w),Y[mk],C)))) for w in (1,3,5,7,9)]
    flagL="  L2-HURT" if L[2]<L[0]-0.05 else ""
    print(f"{s+' ('+format(unst[s],'.4f')+')':26s}| {L[0]:4.1f} {L[1]:4.1f} {L[2]:4.1f} {L[3]:4.1f} {L[4]:4.1f} | {T[0]:4.1f} {T[1]:4.1f} {T[2]:4.1f} {T[3]:4.1f} {T[4]:4.1f}{flagL}")
print("\n=> if L2 never drops at w=5 even on the top (most-unsteady) rows, w=5 denoising is safe.")
