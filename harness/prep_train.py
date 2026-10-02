"""Preprocess train_real H5 -> one float32 array at eval resolution for training.
(82, 868, 32, 64, 2) float32 ~= 2.3 GB. Excludes the known-corrupt 7575_0.h5.
"""
import glob, os, numpy as np, h5py, json
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; TR=f"{B}/data/comp_real/train_real"
OUT=f"{B}/local_harness/train_real_32x64.npy"; META=f"{B}/local_harness/train_real_meta.json"
files=sorted(glob.glob(f"{TR}/*.h5"))
files=[f for f in files if not f.endswith("7575_0.h5")]     # memory 7: corrupt duplicate
print("trajectories: %d"%len(files),flush=True)
Ts=[]; names=[]
arrs=[]
for fp in files:
    with h5py.File(fp,"r") as f:
        u=np.array(f["u"],dtype=np.float32)[:,::2,::2][:,:32,:64]
        v=np.array(f["v"],dtype=np.float32)[:,::2,::2][:,:32,:64]
    a=np.stack([u,v],-1); arrs.append(a); Ts.append(a.shape[0]); names.append(os.path.basename(fp))
    print("  %-14s T=%d"%(names[-1],a.shape[0]),flush=True)
T=min(Ts)
print("min T = %d (truncating all to it)"%T,flush=True)
X=np.stack([a[:T] for a in arrs]).astype(np.float32)
np.save(OUT,X); json.dump(dict(names=names,T=T,shape=list(X.shape)),open(META,"w"),indent=1)
print("saved %s  shape=%s  %.2f GB"%(OUT,X.shape,X.nbytes/1e9))
print("u mean=%.6f std=%.6f | v mean=%.6f std=%.6f | zero frac=%.4f"%(
    X[...,0].mean(),X[...,0].std(),X[...,1].mean(),X[...,1].std(),(X==0).mean()))
