"""MONDAY (GPU) — evaluate the full official foil baseline zoo on the REAL
held-out set and find the best single model + best ensemble to beat 72.78.

Runs on the VM (GPU). Needs, side by side:
  - cloned RealPDEBench repo importable as `realpdebench` (code/RealPDEBench)
  - foil baseline checkpoints from HF `AI4Science-WestlakeU/RealPDEBench-models`
    under CKPT_ROOT/foil/<arch>/finetune.pth  (download list at bottom)
  - the held-out eval npz (scp local_harness/real_eval_heldout/ to the VM, or
    regenerate there with prep_real.py against downloaded real shards)
  - official scoring.py (starting_kit_v9)

Loading recipe (verified offline, session 4):
  * checkpoints are {'model_state_dict': ...}; load that into the built model.
  * build each model via realpdebench.model.load_model.load_model(stub, device, **cfg)
    where cfg comes from realpdebench/configs/foil/<arch>.yaml and stub[0] returns
    (input(20,32,64,3), target(20,32,64,3)) so shapes are inferred.
  * DPOT loads its pretrained weights via the `checkpoint_path` constructor kwarg;
    point that at the dpot pretrain file, THEN load finetune model_state_dict on top.
  * normalize with GaussianNormalizer train_real stats (same as the FNO submission).
  * CNO is very slow on CPU (filtered convs) — GPU only; that's why this is Monday.
"""
import os, sys, glob, importlib.util, time, json
import numpy as np
import torch
import yaml

REPO = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/code/RealPDEBench"   # adjust on VM
CKPT_ROOT = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness/baseline_ckpts"
KIT = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
EVAL = "/SML_DISK_24TB/rajeshr/Aryamann/UGP/local_harness/real_eval_heldout"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, REPO)
from realpdebench.model.load_model import load_model  # noqa
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
scoring = importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)

MEAN_IN = torch.tensor([0.154960856,-0.000513992854,0.0]); STD_IN = torch.tensor([0.0968056545,0.015960684,1.0])
MEAN_TGT = torch.tensor([0.154962569,-0.000517793698,0.0]); STD_TGT = torch.tensor([0.0968104079,0.0159636438,1.0])

class Stub(torch.utils.data.Dataset):
    def __len__(self): return 1
    def __getitem__(self, i):
        return torch.zeros(20,32,64,3), torch.zeros(20,32,64,3)

def cfg_for(arch):
    p = os.path.join(REPO, "realpdebench/configs/foil", f"{arch}.yaml")
    with open(p) as f: c = yaml.safe_load(f)
    return c

def build(arch):
    cfg = cfg_for(arch); cfg["model_name"] = {"gk":"galerkin_transformer"}.get(arch, arch)
    model = load_model(Stub(), device=DEV, **cfg)
    ckpt = os.path.join(CKPT_ROOT, "foil", arch, "finetune.pth")
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        res = model.load_state_dict(sd, strict=False)
        print(f"  {arch}: loaded (missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")
    model.eval().to(DEV)
    return model

@torch.no_grad()
def infer(model, X, mirror=True, batch=32):
    mi,si,mt,st = (t.to(DEV) for t in (MEAN_IN,STD_IN,MEAN_TGT,STD_TGT))
    outs=[]
    for i in range(0,X.shape[0],batch):
        xb=torch.from_numpy(X[i:i+batch]).float().to(DEV)
        y=model((xb-mi)/si)*st+mt
        if mirror:
            x2=torch.flip(xb,dims=[2]).clone(); x2[...,1]=-x2[...,1]
            y2=model((x2-mi)/si)*st+mt; y2=torch.flip(y2,dims=[2]).clone(); y2[...,1]=-y2[...,1]
            y=0.5*(y+y2)
        outs.append(y.cpu().numpy())
    return np.concatenate(outs,0).astype(np.float32)

def score(pred,Y):
    c=scoring.measured_channels(Y)
    rel=float(np.mean(scoring.rel_l2_per_sample(pred,Y,c)))
    tke=float(np.mean(scoring.tke_rel_l2_per_sample(pred,Y,c)))
    mvpe=scoring.mvpe_rel_l2(pred,Y)
    sps,cov=scoring.aggregate_sps(pred,Y,c)
    return dict(L2=scoring.score_error(rel),TKE=scoring.score_error(tke),MVPE=scoring.score_error(mvpe),
                SPS=scoring.score_sps(sps),cov=cov*100,acc3=np.mean([scoring.score_error(rel),scoring.score_error(tke),scoring.score_error(mvpe)]))

def main():
    X=np.load(f"{EVAL}/inputs.npz")["input"]; Y=np.load(f"{EVAL}/targets.npz")["target"]
    print(f"held-out {X.shape}\n")
    ARCHS=["fno","cno","transolver","unet","mwt","gk","dpot_s"]  # dpot_s/gk are the ones to watch
    preds={}
    for a in ARCHS:
        try:
            t0=time.time(); m=build(a); p=infer(m,X); preds[a]=p
            s=score(p,Y); print(f"{a:12s} L2={s['L2']:.2f} TKE={s['TKE']:.2f} MVPE={s['MVPE']:.2f} SPS={s['SPS']:.2f} cov={s['cov']:.1f}% acc3={s['acc3']:.2f} ({time.time()-t0:.0f}s)")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{a:12s} FAIL {type(e).__name__}: {str(e)[:90]}")
    # ensembles of whatever loaded
    import itertools
    names=list(preds)
    print("\n-- ensembles --")
    for r in (2,3):
        for combo in itertools.combinations(names,r):
            p=sum(preds[n] for n in combo)/len(combo); s=score(p,Y)
            print(f"{'+'.join(combo):28s} acc3={s['acc3']:.2f} (L2={s['L2']:.1f} TKE={s['TKE']:.1f} MVPE={s['MVPE']:.1f})")
    np.savez(f"{EVAL}/zoo_preds.npz", **preds, Y=Y)

if __name__=="__main__":
    main()

# --- Monday checkpoint download (run on VM) ---
# from huggingface_hub import hf_hub_download
# for a in ["cno","transolver","unet","mwt","galerkin_transformer","dpot_s"]:
#     hf_hub_download("AI4Science-WestlakeU/RealPDEBench-models", f"foil/{a}/finetune.pth",
#                     repo_type="model", local_dir=CKPT_ROOT)
# DPOT also needs its pretrain file referenced by configs/foil/dpot_s.yaml checkpoint_path.
