"""Is ENSEMBLE DISAGREEMENT more learnable-and-transferable than RAW ERROR?

Context: disagreement is the only uncertainty signal found that IMPROVES under
Reynolds shift (130% retention vs the shipped head's ~10%), but an ensemble
cannot ship -- one fp16 checkpoint is 201.4 MB against a 256 MB cap.

The only way to exploit it within the cap is to DISTILL it: train the small
(~2 MB) head to predict disagreement rather than error, then map head output ->
half-width via a LUT on clean errors, exactly as 05_build_lut.py already does.

That only helps if disagreement is genuinely a more transferable regression
target than error. Raw per-element error carries large aleatoric noise; the
disagreement field is smoother and more systematic, so it plausibly generalises
better -- but that is a hypothesis, and this project has been burned by exactly
this kind of plausible-but-untested reasoning three times.

Fast screen (ridge on the head's own features, subsampled) comparing, under
Reynolds shift:
    target = log|err|        (what the shipped head learns)
    target = log disagreement (the proposal)
If disagreement is not clearly more transferable, distillation is not worth
training and this whole avenue closes.
"""
import json, os, re, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
sys.path.insert(0, f"{B}/train_soup")
from head_common import feats
from load_baseline import load_baseline
SIG = 0.0563870259; DEV = "cuda:0"; C = 2; IN = 20

MI = np.array([0.154960856, -0.000513992854, 0.0], np.float32); SI_ = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MT = np.array([0.154962569, -0.000517793698, 0.0], np.float32); ST = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
mi, si, mt, st = [torch.tensor(x).to(DEV) for x in (MI, SI_, MT, ST)]
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off = meta["off"]; lens = meta["lens"]; names = meta["names"]
ntraj = len(lens)
RE = np.array([int(re.match(r"(\d+)", n).group(1)) for n in names])
uniq_re = sorted(set(RE.tolist())); mid = uniq_re[len(uniq_re)//2]
wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i] + lens[i] - 39, 20):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
# subsample windows for the screen
rng = np.random.default_rng(0)
keep = rng.choice(len(wins), size=min(700, len(wins)), replace=False)
keep.sort()
wins = wins[keep]; wt = wt[keep]
Wall = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = Wall[:, :IN], Wall[:, IN:]; SCM = (Y[..., :C] != 0.0)
w_re = RE[wt]
print("screen on %d windows" % len(wins))

base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def run_sd(sd):
    m = base_model; m.load_state_dict(sd); m = m.to(DEV).eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy()[..., :C])
    return np.concatenate(o, 0).astype(np.float32)

MEM = ["ft_w005_best.pth","ft_w010_best.pth","ft_w015_best.pth","ft_lr1e5_best.pth",
       "ft_w060_best.pth","ft_lr3e5_best.pth","ft_long_w15lr3_best.pth",
       "ft_long_w20lr2_best.pth","ft_long_w10lr1_best.pth"]
MEM = [f"{LH}/{m}" for m in MEM if os.path.exists(f"{LH}/{m}")]
A = np.stack([run_sd(torch.load(p, map_location=DEV)) for p in MEM], 0)
Pm = A.mean(0); SG = A.std(0)
ERR = np.abs(Pm - Y[..., :C])
print("ensemble ready (K=%d)" % len(MEM))

P3 = np.concatenate([Pm, np.zeros(Pm.shape[:-1]+(1,), np.float32)], -1)
FT = feats(P3)                      # (n,20,32,64,NF)
NF = FT.shape[-1]
print("features", FT.shape)

lo_w = w_re < mid; hi_w = w_re >= mid

def ridge_transfer(target, tag, ci=0):
    """fit ridge on LOW Re, measure corr on HIGH Re (and vice versa)."""
    m_all = SCM[..., ci]
    out = []
    for fit_sel, ev_sel, lab in [(lo_w, hi_w, "low->high"), (hi_w, lo_w, "high->low")]:
        f4 = fit_sel[:, None, None, None] & m_all
        e4 = ev_sel[:, None, None, None] & m_all
        Xf = FT[f4][::13]; yf = target[f4][::13]
        Xe = FT[e4][::13]; ye = target[e4][::13]
        mu = Xf.mean(0); sd = Xf.std(0) + 1e-6
        Xf = (Xf - mu)/sd; Xe = (Xe - mu)/sd
        Xf = np.concatenate([Xf, np.ones((len(Xf),1),np.float32)],1)
        Xe = np.concatenate([Xe, np.ones((len(Xe),1),np.float32)],1)
        lam = 1.0
        Aa = Xf.T@Xf + lam*np.eye(Xf.shape[1],dtype=np.float32)
        w = np.linalg.solve(Aa, Xf.T@yf)
        pe = Xe@w
        c = float(np.corrcoef(pe, ye)[0,1])
        # in-fit corr for reference
        pf = Xf@w; cf = float(np.corrcoef(pf, yf)[0,1])
        out.append((lab, cf, c))
    print("  %-24s %-11s in-fit r=%.3f  shifted r=%.3f  retention %.0f%%" %
          (tag, out[0][0], out[0][1], out[0][2], 100*out[0][2]/out[0][1] if out[0][1] else 0))
    print("  %-24s %-11s in-fit r=%.3f  shifted r=%.3f  retention %.0f%%" %
          ("", out[1][0], out[1][1], out[1][2], 100*out[1][2]/out[1][1] if out[1][1] else 0))
    return np.mean([o[2]/o[1] if o[1] else 0 for o in out]), np.mean([o[2] for o in out])

print("\n" + "="*78)
print("Which target is more learnable AND transferable? (ridge screen, u channel)")
print("="*78)
t_err = np.log(ERR[..., 0] + 1e-6).astype(np.float32)
t_dis = np.log(SG[..., 0] + 1e-12).astype(np.float32)
ret_e, r_e = ridge_transfer(t_err, "target = log|err|")
ret_d, r_d = ridge_transfer(t_dis, "target = log disagreement")

print("\n" + "="*78)
print("VERDICT")
print("="*78)
print("  log|err|         : mean shifted r = %.3f  (retention %.0f%%)" % (r_e, 100*ret_e))
print("  log disagreement : mean shifted r = %.3f  (retention %.0f%%)" % (r_d, 100*ret_d))
if r_d > r_e + 0.05:
    print("\n  --> Disagreement IS the more transferable target. Distillation is worth training.")
elif r_d > r_e:
    print("\n  --> Disagreement is marginally better; gain likely too small to matter.")
else:
    print("\n  --> Disagreement is NOT more transferable as a regression target.")
    print("      Distillation would inherit the same failure; this avenue closes.")
print("\n  NOTE: even a perfectly distilled head only reproduces the DISAGREEMENT FIELD,")
print("  not the ensemble's accuracy benefit -- and the measured disagreement gain")
print("  (+0.037 E under shift) is an upper bound on what distillation could deliver.")
json.dump({"r_err": float(r_e), "r_dis": float(r_d),
           "ret_err": float(ret_e), "ret_dis": float(ret_d)},
          open(f"{LH}/distill_test_result.json","w"), indent=1)
