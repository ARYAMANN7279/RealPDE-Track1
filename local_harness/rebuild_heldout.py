"""Rebuild a larger held-out validation set from ALL downloaded real shards,
keeping only in_dist/out_dist (held-out condition) trajectories.
"""
import glob, json, os, numpy as np
from datasets import Dataset
SD="/Users/aryamannsrivastava/Desktop/sem7/UGP/data/real_hf/foil/hf_dataset/real"
FOIL="/Users/aryamannsrivastava/Desktop/sem7/UGP/data/foil"
OUT="/Users/aryamannsrivastava/Desktop/sem7/UGP/local_harness/real_eval_heldout_big"
os.makedirs(OUT,exist_ok=True)
ind=set(json.load(open(f"{FOIL}/in_dist_test_params_real.json")));outd=set(json.load(open(f"{FOIL}/out_dist_test_params_real.json")))
heldout=ind|outd
IN,HOR=20,40
def sub(a):return a[:,::4,::4][:,:32,:64]
def dec(b,t,h,w):
    for dt in (np.float32,np.float16,np.float64):
        if len(b)==t*h*w*np.dtype(dt).itemsize:return np.frombuffer(b,dtype=dt).reshape(t,h,w).astype(np.float32)
    raise ValueError("dtype?")
xs,ys,meta=[],[],[]
kept=[]
for sp in sorted(glob.glob(f"{SD}/*.arrow")):
    ds=Dataset.from_file(sp);row=ds[0];sid=row["sim_id"]
    if sid not in heldout: continue
    kept.append(sid)
    u=sub(dec(row["u"],row["shape_t"],row["shape_h"],row["shape_w"]));v=sub(dec(row["v"],row["shape_t"],row["shape_h"],row["shape_w"]))
    T=u.shape[0]
    for k in range((T-HOR)//HOR+1):
        t0=k*HOR;tr=np.stack([u[t0:t0+HOR],v[t0:t0+HOR],np.zeros_like(u[t0:t0+HOR])],axis=-1)
        xs.append(tr[:IN]);ys.append(tr[IN:]);meta.append({"sim_id":sid,"t0":int(t0)})
X=np.stack(xs).astype(np.float32);Yt=np.stack(ys).astype(np.float32)
print(f"held-out BIG: {X.shape[0]} windows from {len(kept)} trajectories: {sorted(kept)}")
np.savez(f"{OUT}/inputs.npz",input=X);np.savez(f"{OUT}/targets.npz",target=Yt);json.dump(meta,open(f"{OUT}/meta.json","w"))
print("wrote",OUT)
