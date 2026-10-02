"""Does the SHIPPED inference path use the head the way the LUT was fitted?

04_train_head.py:60 and 05_build_lut.py:50 both run the head PER FRAME:
    f = FT[j].permute(0,1,4,2,3).reshape(-1,NF,32,64)   # (B*20, NF, H, W)
The shipped submission.py instead does:
    fb = ft.mean(dim=1)        # time-average the FEATURES
    mu = np.repeat(m1[:,None], 20, axis=1)   # one map broadcast to 20 frames

If that changes E materially, the head's ~10% realised transfer is an
inference-path mismatch, not distribution shift -- which is consistent with the
edge surviving BOTH Re-extrapolation (88.5%) and AoA-extrapolation (89.2%).

Everything here is the shipped artifact: SOUP_v1's own checkpoint, head weights,
LUT and bin edges, unzipped from submission_SOUP_v1.zip.
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
SUB = f"{B}/_tmp/soupv1"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sys.path.insert(0, f"{B}/train_soup")
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
from head_common import feats, global_scale
SIG = S.SIGMA_GLOBAL; DEV = "cuda:0"; C = 2
np.random.seed(1234); torch.manual_seed(1234)

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
SI = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(DEV) for x in (MI, SI, MT, ST)]

X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off, lens, names = meta["off"], meta["lens"], meta["names"]
ntraj = len(lens)
wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
W = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40,32,64,1), np.float32)], -1)
              for s in wins]).astype(np.float32)
Xin, Y = W[:, :20], W[:, 20:]
SCM = (Y[..., :C] != 0.0)
HELD = np.isin(wt, sorted(set(range(0, ntraj, 5))))
print("windows %d | held-out (every5) %d" % (len(wins), HELD.sum()), flush=True)

# ---- shipped checkpoint ----
model, _ = load_baseline(f"{SUB}/sim_real_fno_fp16.pth", device=DEV)
model = model.to(DEV).eval()
P = []
with torch.no_grad():
    for i in range(0, len(Xin), 32):
        xb = torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
        P.append((model((xb - mi)/si)*st + mt).cpu().numpy())
P = np.concatenate(P, 0).astype(np.float32)
P[..., 2] = 0.0
ERR = np.abs(P[..., :C] - Y[..., :C]).astype(np.float32)
SCALE = global_scale(ERR, SCM)
for ci in range(C): ERR[..., ci] *= SCALE[ci]
print("global_scale:", np.round(SCALE, 4), flush=True)

# ---- shipped head + LUT ----
z = np.load(f"{SUB}/head_assets.npz")
NF = int(z["nf"]); LUT = z["LUT"]; ED = z["ED"]; fmu = z["fmu"]; fsd = z["fsd"]
class Head(nn.Module):
    def __init__(s, nf, w=128):
        super().__init__()
        s.n = nn.Sequential(nn.Conv2d(nf,w,3,padding=1), nn.GELU(),
            nn.Conv2d(w,w,3,padding=2,dilation=2), nn.GELU(),
            nn.Conv2d(w,w,3,padding=4,dilation=4), nn.GELU(),
            nn.Conv2d(w,w,3,padding=8,dilation=8), nn.GELU(),
            nn.Conv2d(w,w,3,padding=1), nn.GELU(), nn.Conv2d(w,2,1))
    def forward(s, x): return s.n(x)
head = Head(NF).to(DEV).eval()
head.load_state_dict({k[2:]: torch.from_numpy(z["w_"+k[2:]]) for k in z.files if k.startswith("w_")})

FT = (feats(P) - fmu) / fsd          # (N,20,32,64,NF), identical for both paths

@torch.no_grad()
def mu_perframe():
    out = []
    for i in range(0, len(P), 8):
        f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        out.append(head(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
    return np.concatenate(out, 0)

@torch.no_grad()
def mu_timeavg():
    out = []
    for i in range(0, len(P), 8):
        f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(DEV)      # (b,20,H,W,NF)
        fb = f.mean(dim=1).permute(0,3,1,2)                                # (b,NF,H,W)
        m1 = head(fb).permute(0,2,3,1).cpu().numpy()                       # (b,H,W,2)
        out.append(np.repeat(m1[:, None], 20, axis=1))
    return np.concatenate(out, 0)

def E_of(h, mask):
    num = 0.0; tot = 0
    for ci in range(C):
        m = SCM[..., ci] & mask[:, None, None, None]
        hh = h[..., ci][m]; ee = ERR[..., ci][m]
        num += float((np.exp(-2*hh/SIG) * (ee <= hh)).sum()); tot += ee.size
    return num / tot

def h_from_mu(MU, mult=1.0):
    h = np.empty_like(MU)
    for ci in range(C):
        h[..., ci] = LUT[np.digitize(MU[..., ci], ED[ci]), ci] * mult
    return h

def E_const(mask, hu, hv):
    num = 0.0; tot = 0
    for ci, hc in ((0, hu), (1, hv)):
        m = SCM[..., ci] & mask[:, None, None, None]
        ee = ERR[..., ci][m]
        num += float(np.exp(-2*hc/SIG) * (ee <= hc).sum()); tot += ee.size
    return num / tot

MUf, MUt = mu_perframe(), mu_timeavg()
print("\nmu spread (scored elements, u channel):")
for nm, M in (("per-frame", MUf), ("time-avg ", MUt)):
    v = M[..., 0][SCM[..., 0]]
    print("  %s  sd %.4f   p5 %.3f  p50 %.3f  p95 %.3f" % (nm, v.std(), *np.percentile(v, [5, 50, 95])))

for nm, mask in (("ALL windows", np.ones(len(P), bool)), ("held-out (every5)", HELD)):
    print("\n=== %s ===" % nm)
    print("  constants [0.0129,0.0098]   E %.4f      (real anchor 0.4876)" % E_const(mask, 0.0129, 0.0098))
    print("  constants [0.030 ,0.010 ]   E %.4f      (real anchor 0.4399)" % E_const(mask, 0.030, 0.010))
    ef = E_of(h_from_mu(MUf), mask); et = E_of(h_from_mu(MUt), mask)
    print("  head PER-FRAME  (as fitted) E %.4f" % ef)
    print("  head TIME-AVG   (as SHIPPED)E %.4f      (real measured 0.5020)" % et)
    print("  --> per-frame minus shipped: %+.4f" % (ef - et))
    for mult in (1.25,):
        print("  shipped path x%.2f width     E %.4f  dE %+.4f  (real dE -0.0081)"
              % (mult, E_of(h_from_mu(MUt, mult), mask), E_of(h_from_mu(MUt, mult), mask) - et))
