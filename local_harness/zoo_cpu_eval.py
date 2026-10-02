"""Which zoo models run on CPU, and how accurate? Test unet/transolver/mwt/gk_trans
on a held-out subset; compare to DPOT/FNO; then try DPOT-centric ensembles.
"""
import importlib.util, os, sys, time, yaml, numpy as np, torch
REPO="/Users/aryamannsrivastava/Desktop/sem7/UGP/code/RealPDEBench"
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
CK=f"{H}/baseline_ckpts/foil"
sys.path.insert(0,REPO)
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
from realpdebench.model.load_model import load_model
MI=torch.tensor([0.154960856,-0.000513992854,0.0]);SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]);ST=torch.tensor([0.0968104079,0.0159636438,1.0])
class Stub(torch.utils.data.Dataset):
    def __len__(self): return 1
    def __getitem__(self,i): return torch.zeros(20,32,64,3),torch.zeros(20,32,64,3)
X=np.load(f"{H}/real_eval_v3/inputs.npz")["input"];Y=np.load(f"{H}/real_eval_v3/targets.npz")["target"];C=2
N=int(sys.argv[1]) if len(sys.argv)>1 else 48
Xs,Ys=X[:N],Y[:N]
def acc(P):
    return (scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(P,Ys,C)))),
            scoring.score_error(scoring.mvpe_rel_l2(P,Ys)))
def infer(model,handle3=False):
    outs=[]
    with torch.no_grad():
        for i in range(0,N,4):
            xb=torch.from_numpy(Xs[i:i+4]).float();xn=(xb-MI)/SI
            try: yb=model(xn)
            except TypeError:
                dom=torch.zeros(xb.shape[0],dtype=torch.long); yb=model(xn,dom)
            if isinstance(yb,tuple): yb=yb[0]
            outs.append((yb*ST+MT).numpy())
    P=np.concatenate(outs,0).astype(np.float32);P[...,2]=0.0;return P
CFG={
 "unet":"unet","transolver":"transolver","mwt":"mwt","galerkin_transformer":"gk_trans",
}
preds={}
# DPOT + FNO from cache
preds["dpot"]=np.load(f"{H}/real_eval_v3/dpot_cache.npz")["P"][:N]
preds["fno"]=np.load(f"{H}/real_eval_v3/fno_tta_cache.npz")["P1"][:N]
print(f"{'model':14s}{'rel_l2':>8s}{'mvpe':>8s}{'ms/smp':>8s}  status")
for mname,ckdir in CFG.items():
    cfgp=f"{REPO}/realpdebench/configs/foil/{mname}.yaml"
    try:
        cfg=yaml.safe_load(open(cfgp));cfg["model_name"]=mname
        if "checkpoint_path" in cfg: cfg["checkpoint_path"]=None
        t0=time.time();m=load_model(Stub(),device="cpu",**cfg)
        ckf=f"{CK}/{ckdir}/finetune.pth"
        sd=torch.load(ckf,map_location="cpu");sd=sd.get("model_state_dict",sd)
        res=m.load_state_dict(sd,strict=False);m.eval()
        t1=time.time();P=infer(m);dt=time.time()-t1
        preds[ckdir]=P;r,v=acc(P)
        print(f"{mname:14s}{r:8.2f}{v:8.2f}{dt/N*1000:8.0f}  ok (miss={len(res.missing_keys)})")
    except Exception as e:
        print(f"{mname:14s}{'-':>8s}{'-':>8s}{'-':>8s}  FAIL {type(e).__name__}: {str(e)[:50]}")
# baselines
for k in ("dpot","fno"):
    r,v=acc(preds[k]);print(f"{k:14s}{r:8.2f}{v:8.2f}{'cache':>8s}")
# DPOT-centric ensembles (weighted toward DPOT)
print("\n-- DPOT-centric ensembles (rel_l2, mvpe) --")
import itertools
others=[k for k in preds if k!="dpot"]
for k in others:
    for wd in (0.7,0.85):
        P=wd*preds["dpot"]+(1-wd)*preds[k];r,v=acc(P)
        print(f"  dpot*{wd}+{k}*{1-wd:.2f}: rel_l2={r:.2f} mvpe={v:.2f}  (dpot alone {acc(preds['dpot'])[0]:.2f})")
np.savez(f"{H}/real_eval_v3/zoo_preds_subset.npz",**{k:v for k,v in preds.items()})
