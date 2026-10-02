"""DPOT-small on the FULL 891-window held-out set: robust accuracy + SPS bound
optimization (honest tune/test split) + projected final via the solved formula.
"""
import importlib.util, os, sys, time, json, yaml
import numpy as np, torch
REPO="/Users/aryamannsrivastava/Desktop/sem7/UGP/code/RealPDEBench"
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,REPO)
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
SIGMA=scoring.SIGMA_GLOBAL
from realpdebench.model.load_model import load_model
cfg=yaml.safe_load(open(f"{REPO}/realpdebench/configs/foil/dpot_s.yaml")); cfg["checkpoint_path"]=None
class Stub(torch.utils.data.Dataset):
    def __len__(self): return 1
    def __getitem__(self,i): return torch.zeros(20,32,64,3),torch.zeros(20,32,64,3)
model=load_model(Stub(),device="cpu",**cfg)
model.load_state_dict(torch.load(f"{H}/baseline_ckpts/foil/dpot_s/finetune.pth",map_location="cpu")["model_state_dict"],strict=False)
model.eval()
MI=torch.tensor([0.154960856,-0.000513992854,0.0]);SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]);ST=torch.tensor([0.0968104079,0.0159636438,1.0])
X=np.load(f"{H}/real_eval_v3/inputs.npz")["input"];Y=np.load(f"{H}/real_eval_v3/targets.npz")["target"]
meta=json.load(open(f"{H}/real_eval_v3/meta.json"));sims=np.array([m["sim_id"] for m in meta]);C=2
CACHE=f"{H}/real_eval_v3/dpot_cache.npz"
if not os.path.exists(CACHE):
    t0=time.time();preds=[]
    with torch.no_grad():
        for i in range(0,X.shape[0],8):
            xb=torch.from_numpy(X[i:i+8]).float()
            preds.append((model((xb-MI)/SI)*ST+MT).numpy())
    P=np.concatenate(preds,0).astype(np.float32);P[...,2]=0.0
    np.savez(CACHE,P=P);print(f"DPOT full forward {time.time()-t0:.0f}s ({(time.time()-t0)/X.shape[0]*1000:.0f}ms/sample)",flush=True)
P=np.load(CACHE)["P"]
rel=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(P,Y,C))))
tke=scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(P,Y,C))))
mvpe=scoring.score_error(scoring.mvpe_rel_l2(P,Y))
print(f"\n=== DPOT-small on full {X.shape[0]} held-out ===")
print(f"  rel_l2={rel:.2f}  tke={tke:.2f}  mvpe={mvpe:.2f}   (FNO real: 94.17/74.03/92.84)")
# optimize SPS bounds (honest tune/test split)
uniq=sorted(set(sims.tolist()));tune=np.isin(sims,uniq[::2]);test=~tune
ab=np.abs(Y[...,:C]-P[...,:C]);sc=(Y[...,:C]!=0.0)
Ws=(0.5*(1-scoring.rel_l2_per_sample(P,Y,C)/(0.5+scoring.rel_l2_per_sample(P,Y,C)))
   +0.3*(1-np.nan_to_num(scoring.tke_rel_l2_per_sample(P,Y,C)/(0.5+scoring.tke_rel_l2_per_sample(P,Y,C))))
   +0.2*(1-np.nan_to_num(scoring.mvpe_rel_l2_per_sample(P,Y)/(0.5+scoring.mvpe_rel_l2_per_sample(P,Y))))).astype(np.float32)
Wp=np.broadcast_to(Ws.reshape(-1,1,1,1,1),ab.shape)
def best(ch,mask):
    m=mask.reshape(-1,1,1,1);e=ab[...,ch][sc[...,ch]&m];w=Wp[...,ch][sc[...,ch]&m];o=np.argsort(e);G=np.cumsum(w[o]);e=e[o];return float(e[np.argmax(np.exp(-2*e/SIGMA)*G)])
def spsf(hu,hv,mask):
    ns=np.count_nonzero(sc&mask.reshape(-1,1,1,1,1));tot=0.0;cv=0
    for ch,h in ((0,hu),(1,hv)):
        ins=(ab[...,ch]<=h)&sc[...,ch]&mask.reshape(-1,1,1,1);tot+=np.exp(-2*h/SIGMA)*np.sum(Wp[...,ch]*ins,dtype=np.float64);cv+=np.count_nonzero(ins)
    return 100*min(max(tot/ns if ns else 0,0),1),100*cv/ns if ns else 0
hu,hv=best(0,tune),best(1,tune);st,ct=spsf(hu,hv,test)
hu_a,hv_a=best(0,np.ones_like(tune)),best(1,np.ones_like(tune));sa,ca=spsf(hu_a,hv_a,np.ones_like(tune))
print(f"  SPS opt: test-split SPS={st:.2f} cov={ct:.0f}% ; all-data SPS={sa:.2f} at [{hu_a:.4f},{hv_a:.4f}]")
# project final (6/7-pt formula), time for DPOT ~ high (fast). Use conservative real-calib: local SPS ~= real (fixed bounds).
def proj(rel,tke,mvpe,time_,sps): return 0.4814*rel+0.1182*tke+0.1256*mvpe+0.0946*time_-6.0507+0.2997*sps
for tm in (90,):
    for sps_real in (sa*0.9, sa, 40):
        print(f"  projected final: rel={rel:.1f} sps={sps_real:.1f} time~{tm} -> {proj(rel,tke,mvpe,tm,sps_real):.2f}")
