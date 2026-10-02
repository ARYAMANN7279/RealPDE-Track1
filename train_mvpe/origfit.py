"""Validate the whole fitting method on the ORIGINAL checkpoint, which has FOUR
real anchors instead of the soup's four-but-two-are-LUT-based.

All four submissions below scored identical accuracy subscores (94.17 / 74.03 /
92.84), so they share one W and E = sps / (100*W). They span a very wide range of
bound configurations, including one that is per-element (0.05*|pred|). If a
2-parameter scalar map can fit these and LOO-predict a held-out anchor, the method
is sound and the soup's problem is anchor scarcity. If it fails here too, the
method is fundamentally limited and the soup projection must not be trusted.
"""
import os, sys, json
import numpy as np, torch
B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"; LH = f"{B}/local_harness"
KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, "_vendor"))
import importlib.util as iu
sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
S = iu.module_from_spec(sp); sp.loader.exec_module(S)
from load_baseline import load_baseline
SIG = S.SIGMA_GLOBAL; DEV = "cuda:0"; C = 2

# ---- W for the original checkpoint, from its real subscores ----
from scipy.optimize import brentq
inv = lambda t: brentq(lambda e: S.score_error(e) - t, 1e-9, 50.0)
n_ = lambda x: x/(0.5+x)
e_dm, e_tke, e_mv = inv(94.17), inv(74.03), inv(92.84)
W = 0.5*(1-n_(e_dm)) + 0.3*(1-n_(e_tke)) + 0.2*(1-n_(e_mv))
print("original checkpoint W = %.6f" % W)

ANCHORS = [("default 0.05*|pred|", "prop", 0.05,      14.08),
           ("calibrated",          "const", (0.107537, 0.010307), 18.30),
           ("plain_sps",           "const", (0.030, 0.010),       29.84),
           ("ROBUST",              "const", (0.0129, 0.0098),     33.08)]
for nm, kind, cfg, sps in ANCHORS:
    print("  %-20s sps %6.2f  ->  real E %.4f" % (nm, sps, sps/(100*W)))

CACHE = f"{B}/train_mvpe/runs/orig_arrays.npz"
if not os.path.exists(CACHE):
    MI = np.array([0.154960856,-0.000513992854,0.0],np.float32); SI = np.array([0.0968056545,0.015960684,1.0],np.float32)
    MT = np.array([0.154962569,-0.000517793698,0.0],np.float32); ST = np.array([0.0968104079,0.0159636438,1.0],np.float32)
    mi,si,mt,st = [torch.tensor(x).to(DEV) for x in (MI,SI,MT,ST)]
    X = np.load(f"{LH}/tr_frames.npy", mmap_mode="r"); meta = json.load(open(f"{LH}/tr_meta.json"))
    off, lens = meta["off"], meta["lens"]; ntraj = len(lens)
    wins = []
    for i in range(ntraj):
        for t0 in range(off[i], off[i]+lens[i]-39, 20): wins.append(t0)
    Wd = np.stack([np.concatenate([np.asarray(X[s:s+40]), np.zeros((40,32,64,1),np.float32)],-1) for s in wins]).astype(np.float32)
    Xin, Y = Wd[:,:20], Wd[:,20:]; SCM = (Y[...,:C] != 0.0)
    model,_ = load_baseline(f"{B}/data/comp_real/sim_real_fno.pth", device=DEV); model = model.to(DEV).eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(Xin), 32):
            xb = torch.from_numpy(np.ascontiguousarray(Xin[i:i+32])).to(DEV)
            P.append((model((xb-mi)/si)*st+mt).cpu().numpy())
    P = np.concatenate(P,0).astype(np.float32); P[...,2] = 0.0
    ERR = np.abs(P[...,:C]-Y[...,:C]).astype(np.float32); AP = np.abs(P[...,:C]).astype(np.float32)
    out = {}
    for ci, nm in ((0,"u"),(1,"v")):
        m = SCM[...,ci]
        out[f"err_{nm}"] = ERR[...,ci][m][::4]; out[f"ap_{nm}"] = AP[...,ci][m][::4]
    np.savez_compressed(CACHE, **out); print("cached", CACHE)
z = np.load(CACHE)
d = {nm: dict(err=z[f"err_{nm}"].astype(np.float64), ap=z[f"ap_{nm}"].astype(np.float64)) for nm in ("u","v")}
for nm in ("u","v"):
    o = np.argsort(d[nm]["err"]); d[nm]["errs"] = d[nm]["err"][o]
N = d["u"]["err"].size + d["v"]["err"].size
print("elements: u %d  v %d" % (d["u"]["err"].size, d["v"]["err"].size))

def E_model(p, kind, cfg):
    au, av = p; tot = 0.0
    for nm, a in (("u", au), ("v", av)):
        e = d[nm]["err"] * a
        if kind == "const":
            h = cfg[0] if nm == "u" else cfg[1]
            tot += float((np.exp(-2*h/SIG) * (e <= h)).sum())
        else:
            h = cfg * d[nm]["ap"]
            tot += float((np.exp(-2*h/SIG) * (e <= h)).sum())
    return tot / N

TG = [sps/(100*W) for _,_,_,sps in ANCHORS]
AU = np.linspace(0.8, 4.5, 149); AV = np.linspace(0.8, 4.5, 149)
def fit(idxs):
    best = None
    for au in AU:
        for av in AV:
            p = (au, av)
            c = max(abs(E_model(p, ANCHORS[i][1], ANCHORS[i][2]) - TG[i]) for i in idxs)
            if best is None or c < best[0]: best = (c, p)
    return best[1], best[0]

p_all, c_all = fit(range(4))
print("\nFULL FIT  a_u %.3f  a_v %.3f   max|resid| %.4f" % (*p_all, c_all))
for i, (nm,k,cfg,sps) in enumerate(ANCHORS):
    print("   %-20s target %.4f  model %.4f  resid %+.4f" % (nm, TG[i], E_model(p_all,k,cfg), E_model(p_all,k,cfg)-TG[i]))

print("\nLEAVE-ONE-ANCHOR-OUT (fit on 3, predict the 4th)")
errs = []
for i in range(4):
    p, c = fit([j for j in range(4) if j != i])
    pred = E_model(p, ANCHORS[i][1], ANCHORS[i][2])
    errs.append(abs(pred - TG[i]))
    print("   %-20s predicted %.4f  actual %.4f  error %+.4f   (a_u %.3f a_v %.3f)"
          % (ANCHORS[i][0], pred, TG[i], pred-TG[i], *p))
print("\n   max LOO error %.4f   mean %.4f" % (max(errs), float(np.mean(errs))))
print("   for comparison, global_scale's own u/v were 2.320 / 1.198")
