import argparse, json, os, sys, time
import numpy as np, torch

B="/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH=f"{B}/local_harness"
KIT=f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0,KIT); sys.path.insert(0,os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline

DEV="cuda:0"
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]

meta=json.load(open(f"{LH}/tr_meta.json")); off,lens,names=meta["off"],meta["lens"],meta["names"]
X=np.load(f"{LH}/tr_frames.npy",mmap_mode="r")

wins,wt=[],[]
for i in range(len(lens)):
    for t0 in range(off[i],off[i]+lens[i]-39,10): wins.append(t0); wt.append(i)
wins=np.array(wins); wt=np.array(wt)

model,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)

sd_packed = torch.load(f"{B}/train_es/fno_sv3_e2e_fp16_packed.pth", map_location='cpu')

sd_final = {}
base_sd = model.state_dict()
for k,v in sd_packed.items():
    if len(v.shape) > 0 and v.shape[-1] == 2 and torch.is_complex(base_sd[k]):
        sd_final[k] = torch.view_as_complex(v.float())
    else:
        sd_final[k] = v.float()

model.load_state_dict(sd_final)
model = model.to(DEV).eval()

N=len(wins)
XI=np.zeros((N,20,32,64,2),np.float32); PR=np.zeros((N,20,32,64,2),np.float32)
RS=np.zeros((N,20,32,64,2),np.float32); SC=np.zeros((N,20,32,64,2),bool)

t0=time.time()
with torch.no_grad():
    for i in range(0,N,32):
        w=np.stack([np.concatenate([np.asarray(X[s:s+40]),np.zeros((40,32,64,1),np.float32)],-1)
                    for s in wins[i:i+32]]).astype(np.float32)
        xb=torch.from_numpy(w[:,:20]).to(DEV)
        p=(model((xb-MI)/SI)*ST+MT).cpu().numpy()
        XI[i:i+32]=w[:,:20,...,:2]; PR[i:i+32]=p[...,:2]
        RS[i:i+32]=w[:,20:,...,:2]-p[...,:2]
        SC[i:i+32]=(w[:,20:,...,:2]!=0.0)
        if i%1280==0: print(f"  {i}/{N} {time.time()-t0:.0f}s",flush=True)

out=f"{B}/train_es/cache_e2e_sv3.npz"
np.savez(out,XI=XI,PR=PR,RS=RS,SC=SC,wt=wt)
print(f"[done] {out}")
