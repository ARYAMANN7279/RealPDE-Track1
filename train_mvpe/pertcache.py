"""Add FNO-derived disagreement channels the U-Net cannot compute for itself.

Five architectures/objectives all cap at 22-23% using (input window + prediction). Those
are things the U-Net can in principle derive itself. What it CANNOT derive is how the
FNO behaves -- it cannot simulate a 200MB spectral operator internally.

sec19A: ensemble disagreement was the strongest signal ever measured here (corr 0.558)
but 3 x 201MB will not ship. A SINGLE-checkpoint perturbation probe has no size problem;
it costs one extra forward pass (time subscore ~ -0.48 final, so it must buy >0.48).

Channels added, all from the SAME checkpoint:
  d_noise  |f(x + eps) - f(x)|   input-sensitivity  (sec19A standalone corr 0.420)
  d_trunc  |f_lowpass(x) - f(x)| sensitivity to high-wavenumber input content
"""
import json, os, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT,"_vendor"))
from load_baseline import load_baseline
DEV="cuda:0"
MI=torch.tensor([0.154960856,-0.000513992854,0.0]); SI=torch.tensor([0.0968056545,0.015960684,1.0])
MT=torch.tensor([0.154962569,-0.000517793698,0.0]); ST=torch.tensor([0.0968104079,0.0159636438,1.0])
MI,SI,MT,ST=[t.to(DEV) for t in (MI,SI,MT,ST)]
d = np.load(f"{B}/train_mvpe/runs/errnet_cache.npz")
XI = d["XI"]; N = len(XI)
model,_ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
model.load_state_dict(torch.load(f"{LH}/soup_final_candidate.pth", map_location=DEV))
model = model.to(DEV).eval()
def fwd(x): return model((x-MI)/SI)*ST+MT
DN = np.zeros((N,20,32,64,2), np.float32)
DT = np.zeros((N,20,32,64,2), np.float32)
g = torch.Generator(device=DEV); g.manual_seed(1234)
print("[pert] %d windows"%N, flush=True)
with torch.no_grad():
    for i in range(0, N, 32):
        xb = torch.from_numpy(np.concatenate([XI[i:i+32], np.zeros(XI[i:i+32].shape[:-1]+(1,),np.float32)],-1)).to(DEV)
        p0 = fwd(xb)
        # 1) input noise at 2% of the per-channel std
        sd = xb[...,:2].reshape(-1,2).std(0)
        nz = torch.zeros_like(xb); nz[...,:2] = torch.randn(xb[...,:2].shape, generator=g, device=DEV)*sd*0.02
        DN[i:i+32] = (fwd(xb+nz)-p0)[...,:2].abs().cpu().numpy()
        # 2) low-pass the input along both spatial axes (drop the top third of modes)
        F_ = torch.fft.rfft2(xb[...,:2], dim=(2,3))
        ky, kx = F_.shape[2], F_.shape[3]
        M = torch.ones_like(F_); M[:,:,ky//3:2*ky//3,:] = 0; M[:,:,:,int(kx*2/3):] = 0
        lp = xb.clone(); lp[...,:2] = torch.fft.irfft2(F_*M, s=(32,64), dim=(2,3))
        DT[i:i+32] = (fwd(lp)-p0)[...,:2].abs().cpu().numpy()
        if i % 1600 == 0: print("   %d/%d"%(i,N), flush=True)
np.savez(f"{B}/train_mvpe/runs/pert_cache.npz", DN=DN, DT=DT)
print("[pert] saved", flush=True)
ER = d["ER"]; SC = d["SC"]
for nm, A in (("d_noise",DN), ("d_trunc",DT)):
    for ci,c in ((0,"u"),(1,"v")):
        m = SC[...,ci]
        r = np.corrcoef(np.log(ER[...,ci][m]+1e-9), np.log(A[...,ci][m]+1e-12))[0,1]
        print("  %s %s : log-space corr with |err| = %.3f" % (nm,c,r), flush=True)
print("\n(reference: our U-Net reaches 0.66/0.57; sec19A ensemble disagreement 0.558)", flush=True)
