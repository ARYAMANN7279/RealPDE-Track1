"""Does a single fine-tune really beat the shipped soup, or is that selection bias?

ens_vs_soup.py found member long_w15lr3 beating the soup by +1.36 tke on re_lohi. But
that was the MAX over 11 members on one split, which is biased upward. Proper test:
score every member on THREE independent condition-disjoint splits and check whether the
same member wins on splits it was not selected on.
"""
import json, os, sys, glob
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT,"_vendor"))
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT,"scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
DEV="cuda:0"; C=2
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]
X=np.load(f"{LH}/tr_frames.npy",mmap_mode="r")
meta=json.load(open(f"{LH}/tr_meta.json")); off,lens,names=meta["off"],meta["lens"],meta["names"]
ntraj=len(lens)
re_of=lambda i:int(names[i].split("_")[0]); aoa_of=lambda i:int(names[i].split("_")[1].replace(".h5",""))
SPLITS={"re_lohi": lambda i: re_of(i) in {3750,5025,25425,26700},
        "aoa15":   lambda i: aoa_of(i)==15,
        "aoa0":    lambda i: aoa_of(i)==0,
        "every5":  lambda i: i%5==0}
DATA={}
for nm,f in SPLITS.items():
    st=[]
    for i in range(ntraj):
        if f(i): st += list(range(off[i],off[i]+lens[i]-39,20))
    st=np.array(st)[::2]
    Wd=np.stack([np.concatenate([np.asarray(X[s:s+40]),np.zeros((40,32,64,1),np.float32)],-1) for s in st]).astype(np.float32)
    DATA[nm]=(Wd[:,:20],Wd[:,20:]); print("%-8s %d windows"%(nm,len(st)),flush=True)
base,_=load_baseline(f"{B}/data/comp_real/sim_real_fno.pth",device=DEV)
def evalall(p):
    if p is not None: base.load_state_dict(torch.load(p,map_location=DEV))
    base.to(DEV).eval(); out={}
    with torch.no_grad():
        for nm,(Xi,Y) in DATA.items():
            P=[]
            for i in range(0,len(Xi),32):
                xb=torch.from_numpy(np.ascontiguousarray(Xi[i:i+32])).to(DEV)
                P.append((base((xb-MI)/SI)*ST+MT).cpu().numpy())
            P=np.concatenate(P,0).astype(np.float32)
            r=(S.score_error(float(S.rel_l2_per_sample(P,Y,C).mean())),
               S.score_error(float(S.tke_rel_l2_per_sample(P,Y,C).mean())),
               S.score_error(float(S.mvpe_rel_l2_per_sample(P,Y).mean())))
            out[nm]=0.490*r[0]+0.218*r[1]+0.292*r[2]
    return out
MEM=sorted(glob.glob(f"{LH}/ft_w0*_best.pth")+glob.glob(f"{LH}/ft_lr*_best.pth")
           +glob.glob(f"{LH}/ft_long_*_best.pth")+glob.glob(f"{LH}/ft_w15lr3_best.pth"))
rows=[("SOUP (shipped)", evalall(f"{LH}/soup_final_candidate.pth"))]
for m in MEM: rows.append((os.path.basename(m)[3:-9], evalall(m)))
ks=list(SPLITS.keys())
print("\n%-16s"%"model"+"".join("%10s"%k for k in ks)+"%10s"%"mean")
print("-"*(16+10*(len(ks)+1)))
for nm,d in rows:
    mu=float(np.mean([d[k] for k in ks]))
    print("%-16s"%nm+"".join("%10.4f"%d[k] for k in ks)+"%10.4f"%mu)
soup=dict(rows)["SOUP (shipped)"]
print("\nper-split winner:")
for k in ks:
    w=max(rows,key=lambda r:r[1][k])
    print("   %-8s %-16s %.4f   (soup %.4f, delta %+.4f)"%(k,w[0],w[1][k],soup[k],w[1][k]-soup[k]))
print("\nmembers beating the soup on ALL %d splits:"%len(ks))
any_=False
for nm,d in rows[1:]:
    if all(d[k]>soup[k] for k in ks):
        any_=True
        print("   %-16s mean %+.4f over soup"%(nm,float(np.mean([d[k]-soup[k] for k in ks]))))
if not any_: print("   NONE -- the soup is not beaten consistently; ens_vs_soup was selection bias")
