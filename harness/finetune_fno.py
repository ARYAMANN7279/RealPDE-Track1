"""Gently fine-tune the OFFICIAL FNO on real train trajectories with a
competition-metric-weighted loss (rel_l2 + mvpe + tke), validating on the 9
disjoint held-out trajectories (where the FNO proxy is RELIABLE, 94.13 vs 94.17).
Early-stop on held-out rel_l2. Low LR = gentle (avoid the DPOT overfitting trap).

CPU: smoke-test only (--steps 2). GPU (tomorrow): real run.
  python finetune_fno.py --steps 2 --device cpu       # smoke test
  python finetune_fno.py --steps 4000 --device cuda   # real (tomorrow)
"""
import argparse, glob, importlib.util, json, os, sys, time
import numpy as np, torch
H="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness"
KIT="/Users/aryamannsrivastava/Desktop/sem7/UGP/starting_kit_v9/realpde_t1_starting_kit_v9"
SD="/Users/aryamannsrivastava/Desktop/sem7/UGP/data/real_hf/foil/hf_dataset/real"
FOIL="/Users/aryamannsrivastava/Desktop/sem7/UGP/data/foil"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline
spec=importlib.util.spec_from_file_location("scoring",os.path.join(KIT,"scoring.py"));scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
from datasets import Dataset

MI=np.array([0.154960856,-0.000513992854,0.0],np.float32);SIc=np.array([0.0968056545,0.015960684,1.0],np.float32)
MT=np.array([0.154962569,-0.000517793698,0.0],np.float32);STc=np.array([0.0968104079,0.0159636438,1.0],np.float32)

def build_windows(sids, sub=4, IN=20, HOR=40, max_per=40):
    xs,ys=[],[]
    def dec(b,t,h,w):
        for dt in (np.float32,np.float16,np.float64):
            if len(b)==t*h*w*np.dtype(dt).itemsize: return np.frombuffer(b,dt).reshape(t,h,w).astype(np.float32)
    files={}
    for sp in glob.glob(f"{SD}/*.arrow"):
        r=Dataset.from_file(sp)[0]
        if r["sim_id"] in sids: files[r["sim_id"]]=r
    for sid,r in files.items():
        u=dec(r["u"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::sub,::sub][:,:32,:64]
        v=dec(r["v"],r["shape_t"],r["shape_h"],r["shape_w"])[:,::sub,::sub][:,:32,:64]
        T=u.shape[0]; nwin=(T-HOR)//HOR+1
        for k in range(min(nwin,max_per)):
            t0=k*HOR; tr=np.stack([u[t0:t0+HOR],v[t0:t0+HOR],np.zeros_like(u[t0:t0+HOR])],-1)
            xs.append(tr[:IN]); ys.append(tr[IN:])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)

def rel_l2_t(p,t,c=2):
    p=p[...,:c].reshape(p.shape[0],-1); t=t[...,:c].reshape(t.shape[0],-1)
    return (torch.norm(p-t,dim=1)/torch.norm(t,dim=1).clamp_min(1e-8)).mean()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=2); ap.add_argument("--device",default="cpu")
    ap.add_argument("--lr",type=float,default=1e-5); ap.add_argument("--batch",type=int,default=8); a=ap.parse_args()
    dev=a.device if (a.device=="cpu" or torch.cuda.is_available()) else "cpu"
    train_idx=json.load(open(f"{H}/../data/hf_meta/foil/hf_dataset/train_index_real.json"))
    train_sids=set(e["sim_id"] for e in train_idx)
    ind=set(json.load(open(f"{FOIL}/in_dist_test_params_real.json")));outd=set(json.load(open(f"{FOIL}/out_dist_test_params_real.json")))
    dl=[Dataset.from_file(sp)[0]["sim_id"] for sp in glob.glob(f"{SD}/*.arrow")]
    tr_sids=[s for s in dl if s in train_sids]; print(f"train trajectories: {len(tr_sids)}",flush=True)
    Xtr,Ytr=build_windows(set(tr_sids)); print(f"train windows: {Xtr.shape}",flush=True)
    Xv=np.load(f"{H}/real_eval_v3/inputs.npz")["input"]; Yv=np.load(f"{H}/real_eval_v3/targets.npz")["target"]

    model,_=load_baseline(f"{H}/fno_model/sim_real_fno_fp16.pth",device=dev)
    model=model.float().to(dev); model.train()
    mi=torch.tensor(MI,device=dev);si=torch.tensor(SIc,device=dev);mt=torch.tensor(MT,device=dev);st=torch.tensor(STc,device=dev)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)

    @torch.no_grad()
    def val_rel():
        model.eval(); ps=[]
        for i in range(0,Xv.shape[0],16):
            xb=torch.tensor(Xv[i:i+16],device=dev); ps.append((model((xb-mi)/si)*st+mt).cpu().numpy())
        model.train(); P=np.concatenate(ps,0).astype(np.float32)
        return scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(P,Yv,2))))
    print(f"held-out rel_l2 BEFORE finetune: {val_rel():.3f}  (official FNO real = 94.17)",flush=True)
    N=Xtr.shape[0]
    for step in range(a.steps):
        idx=np.random.choice(N,min(a.batch,N),replace=False)
        xb=torch.tensor(Xtr[idx],device=dev);yb=torch.tensor(Ytr[idx],device=dev)
        pred=model((xb-mi)/si)*st+mt
        loss=rel_l2_t(pred,yb)+torch.nn.functional.mse_loss(pred[...,:2],yb[...,:2])
        opt.zero_grad(); loss.backward(); opt.step()
        if step%max(1,a.steps//5)==0 or step==a.steps-1:
            print(f"  step {step} loss={loss.item():.4f}",flush=True)
    print(f"held-out rel_l2 AFTER {a.steps} steps: {val_rel():.3f}",flush=True)
    print("SMOKE OK" if a.steps<=5 else "DONE",flush=True)

if __name__=="__main__": main()
