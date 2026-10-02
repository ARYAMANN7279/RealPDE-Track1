"""Preprocess train_real keeping FULL trajectory lengths (ragged, concatenated
along time with offsets so windows never cross a trajectory boundary).
"""
import glob, os, numpy as np, h5py, json
B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; TR=f"{B}/data/comp_real/train_real"
files=sorted(glob.glob(f"{TR}/*.h5"))
files=[f for f in files if not f.endswith("7575_0.h5")]
parts=[]; names=[]; lens=[]
for fp in files:
    try:
        with h5py.File(fp,"r") as f:
            u=np.array(f["u"],dtype=np.float32)[:,::2,::2][:,:32,:64]
            v=np.array(f["v"],dtype=np.float32)[:,::2,::2][:,:32,:64]
    except Exception as ex:
        print("  SKIP %s (%s)"%(os.path.basename(fp),ex),flush=True); continue
    parts.append(np.stack([u,v],-1)); names.append(os.path.basename(fp)); lens.append(u.shape[0])
import collections
print("T distribution:",dict(collections.Counter(lens)),flush=True)
X=np.concatenate(parts,0)
off=np.cumsum([0]+lens)
np.save(f"{B}/local_harness/tr_frames.npy",X)
json.dump(dict(names=names,lens=lens,off=off.tolist()),open(f"{B}/local_harness/tr_meta.json","w"))
print("frames %s  %.2f GB  over %d trajectories"%(X.shape,X.nbytes/1e9,len(names)))
print("u mean=%.6f std=%.6f | v mean=%.6f std=%.6f | zeros=%.4f"%(
    X[...,0].mean(),X[...,0].std(),X[...,1].mean(),X[...,1].std(),(X==0).mean()))
tot=sum(max(0,l-40+1) for l in lens)
print("usable 20->20 windows (stride 1): %d"%tot)
