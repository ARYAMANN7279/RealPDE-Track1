"""THE SHIPPABLE CONSEQUENCE of the Reynolds-transfer finding.

An ensemble cannot ship: one fp16 checkpoint is 201.4 MB against a 256 MB cap.
But the transfer test showed something that CAN ship for free.

Under Reynolds shift, E_const fell 0.6285 -> 0.5632: errors on unseen flow
conditions are LARGER. Bounds fitted in-distribution are therefore too TIGHT for
the live set, and an element whose true error exceeds its half-width scores
EXACTLY ZERO -- the asymmetry that makes too-tight far worse than too-wide.

This is a precise, mechanical candidate explanation for why every local bounds
estimate has overshot live (v7: 78.80 -> 77.96; SOUP_v1: 79.70 -> 78.45).

Question this answers: fit bounds on LOW Re, then score on HIGH Re. What
multiplier applied to the in-distribution-optimal half-width maximises the
SHIFTED score? If the answer is > 1.0, our shipped bounds have been too tight and
widening them is a free, single-model, zero-inference-cost gain.

v1 already ships a x1.15 safety multiplier chosen by judgement (section 5); this
measures what it should actually be under a real distribution shift.
"""
import json, os, re, sys
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util
spec = importlib.util.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from load_baseline import load_baseline
SIG = S.SIGMA_GLOBAL; DEV = "cuda:0"; C = 2; IN = 20

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
Wall = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40, 32, 64, 1), np.float32)], -1) for s in wins]).astype(np.float32)
Xin, Y = Wall[:, :IN], Wall[:, IN:]; SCM = (Y[..., :C] != 0.0)
w_re = RE[wt]

base_model, _ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV)
def run_sd(sd):
    m = base_model; m.load_state_dict(sd); m = m.to(DEV).eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            o.append((m((torch.from_numpy(Xin[i:i+32]).to(DEV)-mi)/si)*st+mt).cpu().numpy()[..., :C])
    return np.concatenate(o, 0).astype(np.float32)

# the SHIPPABLE model: the single 9-member weight-averaged soup (1 forward pass)
P = run_sd(torch.load(f"{LH}/soup_final_candidate.pth", map_location=DEV))
ERR = np.abs(P - Y[..., :C])
print("using the single shippable soup checkpoint (1 forward pass)")

def best_h(v):
    v = np.sort(v)
    if v.size < 50: return 0.012
    k = np.arange(1, v.size+1)/v.size
    return float(v[np.argmax(np.exp(-2*v/SIG)*k)])

def E_at(m, hu, hv):
    eu = ERR[..., 0][m & SCM[..., 0]]; evv = ERR[..., 1][m & SCM[..., 1]]
    return float((np.exp(-2*hu/SIG)*(eu <= hu)).sum() + (np.exp(-2*hv/SIG)*(evv <= hv)).sum())/(eu.size+evv.size)

def cov_at(m, hu, hv):
    eu = ERR[..., 0][m & SCM[..., 0]]; evv = ERR[..., 1][m & SCM[..., 1]]
    return float((eu <= hu).sum() + (evv <= hv).sum())/(eu.size+evv.size)

lo = (w_re < mid)[:, None, None, None]; hi = (w_re >= mid)[:, None, None, None]
uniq_t = sorted(set(wt.tolist()))
ind_f = np.isin(wt, uniq_t[0::2])[:, None, None, None]; ind_e = ~ind_f

MULTS = [0.8, 0.9, 1.0, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]

def sweep(fitm, evm, label):
    hu0 = best_h(ERR[..., 0][fitm & SCM[..., 0]])
    hv0 = best_h(ERR[..., 1][fitm & SCM[..., 1]])
    print("\n%s" % label)
    print("  in-fit optimum: hu=%.5f hv=%.5f" % (hu0, hv0))
    print("  %-6s %9s %10s" % ("mult", "E_eval", "coverage"))
    best = None
    for mlt in MULTS:
        e = E_at(evm, hu0*mlt, hv0*mlt); c = cov_at(evm, hu0*mlt, hv0*mlt)
        star = ""
        if best is None or e > best[1]: best = (mlt, e); star = ""
        print("  x%-5.2f %9.4f %10.3f" % (mlt, e, c))
    # recompute best cleanly
    vals = [(mlt, E_at(evm, hu0*mlt, hv0*mlt)) for mlt in MULTS]
    bm, be = max(vals, key=lambda t: t[1])
    print("  --> BEST multiplier %.2f  (E %.4f, vs x1.00 %.4f, gain %+.4f)"
          % (bm, be, E_at(evm, hu0, hv0), be - E_at(evm, hu0, hv0)))
    return bm, be, hu0, hv0

print("="*78)
print("CONTROL: in-distribution (what we normally fit/score on)")
print("="*78)
bm0, be0, _, _ = sweep(ind_f, ind_e, "fit interleaved -> score interleaved")

print("\n" + "="*78)
print("SHIFTED: fit LOW Re -> score HIGH Re   <-- closest analogue to live")
print("="*78)
bm1, be1, hu1, hv1 = sweep(lo, hi, "fit Re<%d -> score Re>=%d" % (mid, mid))

print("\n" + "="*78)
print("SHIFTED (reverse): fit HIGH Re -> score LOW Re")
print("="*78)
bm2, be2, hu2, hv2 = sweep(hi, lo, "fit Re>=%d -> score Re<%d" % (mid, mid))

print("\n" + "="*78)
print("VERDICT")
print("="*78)
print("  best multiplier in-distribution : x%.2f" % bm0)
print("  best multiplier under shift     : x%.2f (fwd)  x%.2f (rev)" % (bm1, bm2))
print("  v1 shipped x1.15 (chosen by judgement, section 5)")
if (bm1 + bm2)/2 > 1.05:
    print("\n  --> Bounds fitted in-distribution ARE too tight for shifted data.")
    print("      Widening is a free, single-model, zero-inference-cost gain.")
else:
    print("\n  --> No consistent widening benefit; current multiplier is already reasonable.")
json.dump({"mult_indist": bm0, "mult_fwd": bm1, "mult_rev": bm2},
          open(f"{LH}/shift_multiplier_result.json", "w"), indent=1)
print("\nwrote shift_multiplier_result.json")
