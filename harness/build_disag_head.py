"""DISAGREEMENT-DISTILLED HEAD -- the shippable form of today's finding.

Chain: feats(P) -> [small CNN head] -> predicted log-disagreement -> LUT -> half-width.

Why this and not the shipped head:
  * shipped head regresses log|err|. Locally +0.0588 E, live +0.0056 (section 6A):
    a learned error ranking does not survive the shift to unseen Reynolds numbers.
  * the ensemble disagreement field is the only uncertainty signal measured today
    whose advantage GROWS under Reynolds shift (retention 130%, both directions).
  * an ensemble cannot ship (1 fp16 ckpt = 201.4 MB vs a 256 MB cap), but the
    disagreement FIELD can be distilled into the existing ~2 MB head: it is far
    more learnable than error (ridge screen r=0.778 vs 0.501) and transfers fully.
  * identical architecture and inference cost to what SOUP_v1 already ships, so
    the time subscore is unchanged -- only the regression target differs.

The head's output is used ONLY to RANK/bin elements; every half-width still comes
from CLEAN errors via best_h() inside each bin, with a GLOBAL per-channel scale
(never a per-element transform -- that is the section 6A / GATE 4B failure).
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
WORK = f"{B}/train_work_maxsoup"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sys.path.insert(0, f"{B}/train_soup")
from head_common import feats, Net, global_scale, SIGMA_GLOBAL as SIG
from load_baseline import load_baseline
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
dev = "cuda:0"; C = 2; IN_STEP = OUT_STEP = 20; HOR = 40
SEED = 1234
np.random.seed(SEED); torch.manual_seed(SEED)

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI_ = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(dev) for x in (MI, SI_, MT, ST)]

F = np.load(os.path.join(WORK, "frames.npy"), mmap_mode="r")
m = json.load(open(os.path.join(WORK, "meta.json")))
wins, wtraj = [], []
for i, (o, L) in enumerate(zip(m["off"], m["lens"])):
    for t0 in range(o, o + L - HOR + 1, HOR):
        wins.append(t0); wtraj.append(i)
W = np.stack([np.concatenate([np.asarray(F[s:s+HOR]), np.zeros((HOR, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = W[:, :IN_STEP], W[:, IN_STEP:]
wtraj = np.array(wtraj)
SCM = (Y[..., :C] != 0.0)
print("windows %d over %d trajectories (pipeline-aligned, stride %d)" % (len(W), len(m["names"]), HOR))

base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=dev)
def run_sd(sd):
    mm = base_model; mm.load_state_dict(sd); mm = mm.to(dev).eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            o.append((mm((torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(dev)-mi)/si)*st+mt).cpu().numpy()[..., :C])
    return np.concatenate(o, 0).astype(np.float32)

# the SHIPPED model's own predictions (single soup checkpoint)
P = run_sd(torch.load(f"{LH}/soup_final_candidate.pth", map_location=dev))
print("soup predictions ready")

MEM = ["ft_w005_best.pth","ft_w010_best.pth","ft_w015_best.pth","ft_lr1e5_best.pth",
       "ft_w060_best.pth","ft_lr3e5_best.pth","ft_long_w15lr3_best.pth",
       "ft_long_w20lr2_best.pth","ft_long_w10lr1_best.pth"]
MEM = [f"{LH}/{x}" for x in MEM if os.path.exists(f"{LH}/{x}")]
acc = None; acc2 = None
for p in MEM:
    q = run_sd(torch.load(p, map_location=dev))
    acc = q if acc is None else acc + q
    acc2 = q*q if acc2 is None else acc2 + q*q
n = len(MEM)
SG = np.sqrt(np.maximum(acc2/n - (acc/n)**2, 0)).astype(np.float32)
del acc, acc2
print("disagreement field ready (K=%d), teacher only -- not shipped" % n)

P3 = np.concatenate([P, np.zeros(P.shape[:-1]+(1,), np.float32)], -1)
FT = feats(P3); NF = FT.shape[-1]
uniq = sorted(set(wtraj.tolist()))
tr = np.isin(wtraj, uniq[0::2]); te = ~tr
mu_ = FT[tr].reshape(-1, NF).mean(0); sd_ = FT[tr].reshape(-1, NF).std(0) + 1e-6
FT = (FT - mu_) / sd_
TGT = np.log(SG + 1e-12).astype(np.float32)   # <-- disagreement, not error
print("features %s  target=log(disagreement)" % (FT.shape,))

net = Net(NF).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
EP = 26
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP, eta_min=5e-5)
best = 1e9
for ep in range(EP):
    idx = np.where(tr)[0]; np.random.default_rng(SEED+ep).shuffle(idx)
    for i in range(0, len(idx), 8):
        j = idx[i:i+8]
        f = torch.from_numpy(np.ascontiguousarray(FT[j])).to(dev).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        pr = net(f).reshape(len(j),20,2,32,64).permute(0,1,3,4,2)
        t = torch.from_numpy(np.ascontiguousarray(TGT[j])).to(dev)
        mk = torch.from_numpy(SCM[j].astype(np.float32)).to(dev)
        loss = ((pr - t).abs() * mk).sum() / mk.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    sch.step()
    if float(loss) < best:
        best = float(loss)
        torch.save({"sd": net.state_dict(), "mu": mu_, "std": sd_, "nf": NF}, os.path.join(WORK, "head_disag.pth"))
    if ep % 5 == 4: print("  ep %2d L1 %.4f" % (ep, float(loss)), flush=True)
print("trained head_disag.pth (best L1 %.4f)" % best)

ck = torch.load(os.path.join(WORK, "head_disag.pth"), map_location=dev, weights_only=False)
net.load_state_dict(ck["sd"]); net.eval()
MU = []
with torch.no_grad():
    for i in range(0, len(P), 8):
        f = torch.from_numpy(np.ascontiguousarray(FT[i:i+8])).to(dev).permute(0,1,4,2,3).reshape(-1,NF,32,64)
        MU.append(net(f).reshape(-1,20,2,32,64).permute(0,1,3,4,2).cpu().numpy())
MU = np.concatenate(MU, 0)

ERR = np.abs(P - Y[..., :C]).astype(np.float32)
SCALE = global_scale(ERR, SCM)
for ci in range(C): ERR[..., ci] *= SCALE[ci]
print("global per-channel scale: u x%.3f v x%.3f" % (SCALE[0], SCALE[1]))

def best_h(v):
    v = np.sort(v)
    if v.size == 0: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

NB = 24; LUT = np.zeros((NB,2), np.float32); ED = np.zeros((2,NB-1), np.float32)
for ci in range(C):
    mk = SCM[..., ci]; mu = MU[..., ci][mk]; er = ERR[..., ci][mk]
    q = np.quantile(mu, np.linspace(0,1,NB+1)[1:-1]); ED[ci] = q
    b = np.digitize(mu, q)
    for k in range(NB):
        s = b == k
        LUT[k, ci] = best_h(er[s][::3]) if s.sum() > 200 else 0.012
assert LUT[:,0].min() > 0.005, "LUT u near-zero (%.5f) -- corrupted ranking" % LUT[:,0].min()
assert LUT[:,1].min() > 0.0015, "LUT v near-zero (%.5f)" % LUT[:,1].min()
print("LUT h_u %.4f..%.4f (med %.4f)" % (LUT[:,0].min(), LUT[:,0].max(), np.median(LUT[:,0])))
print("LUT h_v %.4f..%.4f (med %.4f)" % (LUT[:,1].min(), LUT[:,1].max(), np.median(LUT[:,1])))

# held-out comparison vs constants, on the trajectory-disjoint split
def E_eval(hmap_fn, mask):
    num = 0.0; tot = 0
    m4 = mask[:, None, None, None]
    for ci in range(C):
        me = SCM[..., ci] & m4
        h = hmap_fn(ci, me); ee = ERR[..., ci][me]
        num += float((np.exp(-2*h/SIG)*(ee <= h)).sum()); tot += ee.size
    return num/tot
hu_c = best_h(ERR[..., 0][tr[:,None,None,None] & SCM[...,0]])
hv_c = best_h(ERR[..., 1][tr[:,None,None,None] & SCM[...,1]])
Ec = E_eval(lambda ci, me: (hu_c if ci==0 else hv_c), te)
Ed = E_eval(lambda ci, me: LUT[np.digitize(MU[..., ci][me], ED[ci]), ci], te)
print("\nheld-out: E_constants=%.4f  E_disag-head=%.4f  gain %+.4f" % (Ec, Ed, Ed-Ec))

sd_out = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
g = np.ones((11,2), np.complex64)   # identity spectral gain: correction is dropped
np.savez_compressed(os.path.join(WORK, "head_assets_disag.npz"), gt=g, LUT=LUT, ED=ED,
                    fmu=ck["mu"], fsd=ck["std"], nf=np.int32(NF),
                    **{"w_"+k: v for k, v in sd_out.items()})
print("wrote head_assets_disag.npz")
json.dump({"E_const": Ec, "E_disag_head": Ed, "gain": Ed-Ec,
           "scale": list(map(float, SCALE))}, open(os.path.join(WORK, "disag_head_result.json"), "w"), indent=1)
