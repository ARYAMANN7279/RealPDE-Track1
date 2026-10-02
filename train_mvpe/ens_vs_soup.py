"""Is a PREDICTION ensemble better than the WEIGHT soup we ship?

The soup averages weights (one shippable checkpoint). Averaging predictions is normally
better but needs every member at inference -- 6 x 192MB blows the 256MB cap. If the
prediction ensemble is meaningfully better, DISTILLING it into a single student is a
shippable way to capture that, and it is entirely outside the recipe sec14 exhausted.

Evaluated on re_lohi, the condition-disjoint split shown to track the live board.
"""
import json, os, sys, glob
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
DEV = "cuda:0"; C = 2
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off,lens,names = meta["off"],meta["lens"],meta["names"]
ntraj = len(lens)
LOHI = {3750,5025,25425,26700}
starts = []
for i in range(ntraj):
    if int(names[i].split("_")[0]) in LOHI:
        starts += list(range(off[i], off[i]+lens[i]-39, 20))
starts = np.array(starts)[::2]
print("re_lohi eval windows: %d" % len(starts), flush=True)
Wd = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40,32,64,1),np.float32)],-1)
               for s in starts]).astype(np.float32)
Xin, Y = Wd[:,:20], Wd[:,20:]

# the real-data-init family: the members that built SOUP_v1
MEM = sorted(glob.glob(f"{LH}/ft_w0*_best.pth") + glob.glob(f"{LH}/ft_lr*_best.pth")
             + glob.glob(f"{LH}/ft_long_*_best.pth") + glob.glob(f"{LH}/ft_w15lr3_best.pth"))
print("ensemble members (%d):" % len(MEM), flush=True)
for m in MEM: print("   ", os.path.basename(m), flush=True)

base,_ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def predict(sd_path):
    if sd_path is not None:
        base.load_state_dict(torch.load(sd_path, map_location=DEV))
    base.to(DEV).eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(Xin),32):
            xb=torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
            out.append((base((xb-MI)/SI)*ST+MT).cpu().numpy())
    return np.concatenate(out,0).astype(np.float32)
def score(P):
    return (S.score_error(float(S.rel_l2_per_sample(P,Y,C).mean())),
            S.score_error(float(S.tke_rel_l2_per_sample(P,Y,C).mean())),
            S.score_error(float(S.mvpe_rel_l2_per_sample(P,Y).mean())))
def eff(r): return 0.490*r[0]+0.218*r[1]+0.292*r[2]

print("\n%-34s %7s %7s %7s %9s"%("model","rel_l2","tke","mvpe","eff"), flush=True)
print("-"*68)
P0 = predict(None); r0 = score(P0)
print("%-34s %7.2f %7.2f %7.2f %9.4f"%("kit baseline (no fine-tune)",*r0,eff(r0)), flush=True)
Psoup = predict(f"{LH}/soup_final_candidate.pth"); rs = score(Psoup)
print("%-34s %7.2f %7.2f %7.2f %9.4f"%("WEIGHT soup (shipped)",*rs,eff(rs)), flush=True)
acc = np.zeros_like(P0); indiv=[]
for m in MEM:
    Pm = predict(m); acc += Pm; rm = score(Pm); indiv.append(eff(rm))
    print("%-34s %7.2f %7.2f %7.2f %9.4f"%("  member "+os.path.basename(m)[3:-9],*rm,eff(rm)), flush=True)
Pens = acc/len(MEM); re_ = score(Pens)
print("-"*68, flush=True)
print("%-34s %7.2f %7.2f %7.2f %9.4f"%("PREDICTION ensemble (%d)"%len(MEM),*re_,eff(re_)), flush=True)
print("\nprediction ensemble vs weight soup: rel_l2 %+.3f  tke %+.3f  mvpe %+.3f  eff %+.4f"
      %(re_[0]-rs[0], re_[1]-rs[1], re_[2]-rs[2], eff(re_)-eff(rs)), flush=True)
print("best single member eff %.4f | soup %.4f | ensemble %.4f"%(max(indiv),eff(rs),eff(re_)), flush=True)
d = 0.467*(re_[0]-rs[0]) + 0.208*(re_[1]-rs[1]) + 0.290*(re_[2]-rs[2])
print("\nif it transferred 1:1 -> final %+.4f ; at the measured ~50%% rate -> %+.4f"%(d, d/2), flush=True)
np.save(f"{B}/train_mvpe/runs/ens_pred_lohi.npy", Pens)
