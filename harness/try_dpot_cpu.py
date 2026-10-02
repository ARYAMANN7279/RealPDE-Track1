"""Feasibility: can DPOT-small run on CPU? Build it, load the foil finetune
checkpoint, forward a few held-out windows, and (if fast enough) score accuracy
on a subset vs the FNO's rel_l2 94.17. Tells us if DPOT is the Monday priority.
"""
import importlib.util, os, sys, time, yaml
import numpy as np, torch

REPO="/Users/aryamannsrivastava/Desktop/sem7/UGP/code/RealPDEBench"
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, REPO)
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
from realpdebench.model.load_model import load_model

cfg=yaml.safe_load(open(f"{REPO}/realpdebench/configs/foil/dpot_s.yaml"))
cfg["checkpoint_path"]=None  # skip missing pretrain; we load finetune below

class Stub(torch.utils.data.Dataset):
    def __len__(self): return 1
    def __getitem__(self,i): return torch.zeros(20,32,64,3), torch.zeros(20,32,64,3)

print("building DPOT-small...", flush=True)
t0=time.time()
model=load_model(Stub(), device="cpu", **cfg)
print(f"  built in {time.time()-t0:.0f}s", flush=True)
ck=torch.load(f"{H}/baseline_ckpts/foil/dpot_s/finetune.pth", map_location="cpu")
res=model.load_state_dict(ck["model_state_dict"], strict=False)
print(f"  loaded finetune (missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})", flush=True)
model.eval()

MEAN_IN=torch.tensor([0.154960856,-0.000513992854,0.0]);STD_IN=torch.tensor([0.0968056545,0.015960684,1.0])
MEAN_TGT=torch.tensor([0.154962569,-0.000517793698,0.0]);STD_TGT=torch.tensor([0.0968104079,0.0159636438,1.0])
X=np.load(f"{H}/real_eval_v3/inputs.npz")["input"]; Y=np.load(f"{H}/real_eval_v3/targets.npz")["target"]
N=24  # small subset for feasibility
Xs, Ys = X[:N], Y[:N]

print(f"forward on {N} windows (CPU)...", flush=True)
t0=time.time()
with torch.no_grad():
    preds=[]
    for i in range(0, N, 4):
        xb=torch.from_numpy(Xs[i:i+4]).float()
        xn=(xb-MEAN_IN)/STD_IN
        yb=model(xn)*STD_TGT+MEAN_TGT
        preds.append(yb.numpy())
        print(f"  batch {i//4} done {time.time()-t0:.0f}s", flush=True)
pred=np.concatenate(preds,0).astype(np.float32); pred[...,2]=0.0
dt=time.time()-t0
print(f"forward done: {dt:.0f}s ({dt/N*1000:.0f}ms/sample CPU)", flush=True)
c=scoring.measured_channels(Ys)
rel=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(pred,Ys,c))))
mvpe=scoring.score_error(scoring.mvpe_rel_l2(pred,Ys))
print(f"\nDPOT-small on {N} held-out windows: rel_l2={rel:.2f} mvpe={mvpe:.2f}", flush=True)
# FNO on same subset for comparison
d=np.load(f"{H}/real_eval_v3/fno_tta_cache.npz");fno=d["P1"][:N]
frel=scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(fno,Ys,c))))
fmvpe=scoring.score_error(scoring.mvpe_rel_l2(fno,Ys))
print(f"FNO   on same {N} windows: rel_l2={frel:.2f} mvpe={fmvpe:.2f}")
print(f"\n=> DPOT {'BEATS' if rel>frel else 'does NOT beat'} FNO on rel_l2 ({rel:.2f} vs {frel:.2f}) on this subset")
