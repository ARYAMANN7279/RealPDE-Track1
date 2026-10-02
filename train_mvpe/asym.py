"""Are symmetric bounds leaving value on the table?

Every submission we have made uses lower = pred - h, upper = pred + h. But SPS scores
exp(-(upper-lower)/sigma) * 1[target in [lower,upper]]; nothing requires the interval to
be centred on the prediction. If the signed error has bias or skew, the optimal interval
is offset. This measures the gain from optimal (offset, width) over optimal (width) alone.

Calibration-free: both policies are evaluated on the same raw signed errors.
"""
import numpy as np
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; SIG = 0.0563870259
d = np.load(f"{B}/train_mvpe/runs/errnet_cache.npz")
PR, SC = d["PR"], d["SC"]
# signed error = pred - target; recover target from |err| is impossible, so recompute
import json, sys, torch
LH = f"{B}/local_harness"
X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r")
meta = json.load(open(f"{LH}/tr_meta.json")); off,lens,names = meta["off"],meta["lens"],meta["names"]
ntraj = len(lens); wins, wt = [], []
for i in range(ntraj):
    for t0 in range(off[i], off[i]+lens[i]-39, 10):
        wins.append(t0); wt.append(i)
wins = np.array(wins); wt = np.array(wt)
RE = np.array([int(names[i].split("_")[0]) for i in wt])
VAL = np.isin(RE,[3750,5025,25425,26700])
sel = np.where(VAL)[0][::3]
Y = np.stack([np.asarray(X[wins[i]+20:wins[i]+40])[...,:2] for i in sel]).astype(np.float32)
Pv, Sv = PR[sel], SC[sel]
SGN = (Pv - Y)                      # signed error; target = pred - SGN
print("windows used: %d  elements: %d" % (len(sel), int(Sv.sum())))
for ci, nm in ((0,"u"),(1,"v")):
    s = SGN[...,ci][Sv[...,ci]].astype(np.float64)
    print("  %s signed error: mean %+.6f  median %+.6f  skew %+.3f  sd %.6f"
          % (nm, s.mean(), np.median(s), float(((s-s.mean())**3).mean()/s.std()**3), s.std()))

def best_sym(s):
    g = np.linspace(0.0005,0.05,900); a = np.abs(s)
    k = np.array([(a<=h).sum() for h in g])
    j = int(np.argmax(np.exp(-2*g/SIG)*k)); return g[j], float(np.exp(-2*g[j]/SIG)*k[j])/s.size
def best_asym(s):
    """optimal (lo,hi) offsets: interval is [pred-hi, pred+lo] in error coords"""
    srt = np.sort(s); n = srt.size
    best = (None,None,-1)
    for w in np.linspace(0.001,0.08,160):                 # total width
        # slide the window: choose start quantile maximising coverage for this width
        lo = srt[0]; hi_idx = np.searchsorted(srt, srt + w, side="right")
        cov = hi_idx - np.arange(n)
        j = int(np.argmax(cov))
        val = np.exp(-w/SIG)*cov[j]/n
        if val > best[2]: best = (srt[j], srt[j]+w, val)
    return best
print()
tot_sym = tot_asym = 0.0; N = 0
for ci, nm in ((0,"u"),(1,"v")):
    s = SGN[...,ci][Sv[...,ci]].astype(np.float64)
    sub = s[::7]
    h, Es = best_sym(sub); lo, hi, Ea = best_asym(sub)
    print("  %s  symmetric  h %.5f            E %.4f" % (nm, h, Es))
    print("  %s  asymmetric [%+.5f,%+.5f] w %.5f  E %.4f   gain %+.4f"
          % (nm, lo, hi, hi-lo, Ea, Ea-Es))
    tot_sym += Es*sub.size; tot_asym += Ea*sub.size; N += sub.size
print("\n  combined  symmetric E %.4f -> asymmetric E %.4f   gain %+.5f"
      % (tot_sym/N, tot_asym/N, (tot_asym-tot_sym)/N))
W = 0.684593
print("  -> final impact %+.4f" % (0.217*100*W*((tot_asym-tot_sym)/N)))
