"""Coordinator task, 14 Sep.
(1) On the SCREEN / SPEED_SAFE bounds (6e7a6290), predict E_live and sps_live for per-channel global width scales
    k_u x k_v and for a per-bin re-optimised 24-level LUT, under every live family fitted in fit_live.json. Each
    family is re-anchored so it reproduces SCREEN's own live E exactly; every prediction carries that family's
    leave-one-group-out error on the three shipped width-change DIFFERENCES (LUTCAL-FP16, WIDE125-SOUPv1,
    LUTFIX-SOUPv1).
(2) LUTCAL vs TMEAN (identical centres, LUT only): split into global scaling / u-part / v-part, locally and under
    every family; compare with the live difference.
(3) The proposal gate: dE_live > +0.005 with the LOO band entirely above it.
Local numbers use each zip's own predict() on the 734-window set (cache734); kit aggregate_sps for the exact check.
"""
import io
import json
import math
import os
import zipfile

import numpy as np
import torch
from scipy.optimize import brentq

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
D = f"{B}/agents/bounds/cache734"
SIG = 0.0563870259
SQ2 = math.sqrt(2.0)
dev = "cuda"
FL = json.load(open(f"{B}/agents/bounds/fit_live.json"))
E_LIVE = FL["E_live"]
MEDt = torch.tensor(FL["med_ref"], device=dev)
W_LIVE = 0.68908                 # SCREEN live W_agg (SPEED_SAFE: same backbone and bounds)
KFIN = 0.24737 * 100 * W_LIVE    # final points per unit E
FAMS = ["S1", "S2", "SL", "SR", "SP", "SN"]
PAIRS = [("LUTCAL", "FP16"), ("WIDE125", "SOUPv1"), ("LUTFIX", "SOUPv1")]


def kit():
    import importlib.util as iu
    KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
    sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
    S_ = iu.module_from_spec(sp)
    sp.loader.exec_module(S_)
    return S_


S = kit()
T = np.load(f"{D}/TGT.npy")
per_win = int(np.prod(T.shape[1:]))
idx_all = np.nonzero((T != 0).reshape(-1))[0]
rng = np.random.default_rng(0)                                   # identical to fit_live.py
idx = np.sort(rng.choice(idx_all, size=min(12_000_000, len(idx_all)), replace=False))
win = idx // per_win
cht = torch.from_numpy((idx % 2).astype(np.int64)).to(dev)
fold = torch.from_numpy((win % 2).astype(np.int64)).to(dev)
t_sub = T.reshape(-1)[idx]
g = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
with zipfile.ZipFile(f"{B}/submissions/submission_SCREEN.zip") as zf:
    LUT = np.load(io.BytesIO(zf.read("bounds_assets.npz")))["LUT"].astype(np.float64)   # (24,2)


def Wwin_of(P):
    n_ = lambda x: x / (0.5 + x)
    return (0.5 * (1 - n_(S.rel_l2_per_sample(P, T, 2))) + 0.3 * (1 - n_(S.tke_rel_l2_per_sample(P, T, 2)))
            + 0.2 * (1 - n_(S.mvpe_rel_l2_per_sample(P, T))))


def load(tag, keep_full=False):
    P = np.load(f"{D}/P_{tag}.npy")
    C = np.load(f"{D}/C_{tag}.npy")
    HD = np.load(f"{D}/HD_{tag}.npy")
    HU = np.load(f"{D}/HU_{tag}.npy")
    Ww = Wwin_of(P)
    p, c = P.reshape(-1)[idx], C.reshape(-1)[idx]
    hd, hu = HD.reshape(-1)[idx], HU.reshape(-1)[idx]
    X = dict(d=g(t_sub - (p + c)), r=g(t_sub - p), c=g(c), h=g(0.5 * (hd + hu)), W=g(Ww[win]),
             asym=float(np.max(np.abs(hd - hu))))
    return X, (dict(P=P, C=C, HD=HD, HU=HU) if keep_full else None)


def Phi(z):
    return 0.5 * (1.0 + torch.erf(z / SQ2))


def xval(fam, th, X):
    d = X["d"]
    if fam in ("S1", "SL", "SR"):
        return th[0] * d
    if fam == "S2":
        return torch.where(cht == 0, th[0] * d, th[1] * d)
    if fam == "SP":
        m = MEDt[cht]
        return torch.sign(d) * th[0] * m * (d.abs() / m) ** th[1]
    if fam == "SN":
        return th[0] * (X["r"] - th[1] * X["c"])
    return d                                                        # LOCAL


def Ehat(fam, th, X, h, mask=None):
    x = xval(fam, th, X)
    if fam == "SR":
        P = Phi(torch.log(h / x.abs().clamp_min(1e-12)) / th[1])
    else:
        P = (x.abs() <= h).float()
        if fam == "SL":
            P = P * th[1]
    w = X["W"] if mask is None else X["W"] * mask
    return float((w * torch.exp(-2 * h / SIG) * P).sum() / w.sum())


def scale_th(fam, th0, s):
    th = list(th0)
    th[0] *= s
    if fam == "S2":
        th[1] *= s
    return th


def anchor(fam, X, target):
    th0 = FL["fams"][fam]["theta"]
    s = brentq(lambda s_: Ehat(fam, scale_th(fam, th0, s_), X, X["h"]) - target, 0.2, 4.0, xtol=1e-7)
    return scale_th(fam, th0, s), s


def bands():
    out = {}
    for f in FAMS:
        loo, rin = FL["fams"][f]["loo"], FL["fams"][f]["res_in"]
        el = [loo[a] - loo[b] for a, b in PAIRS]
        ei = [rin[a] - rin[b] for a, b in PAIRS]
        out[f] = dict(loo=el, ins=ei, lo=min(el), hi=max(el))
    return out


BND = bands()
print("## LOO error of each family on the shipped width-change DIFFERENCES (model - live), E units")
print("| family | LUTCAL-FP16 | WIDE125-SOUPv1 | LUTFIX-SOUPv1 | in-sample same three |")
print("|---|---:|---:|---:|---|")
for f in FAMS:
    b = BND[f]
    print(f"| {f} | {b['loo'][0]:+.4f} | {b['loo'][1]:+.4f} | {b['loo'][2]:+.4f} | "
          + " ".join(f"{v:+.4f}" for v in b["ins"]) + " |")


def row(name, X, h, anch, base_local, base_loc_sps=None):
    loc = Ehat("LOCAL", None, X, h) - base_local
    cells, lo_all, hi_all = [], [], []
    for f in FAMS:
        dE = Ehat(f, anch[f], X, h) - E_BASE
        b = BND[f]
        lo, hi = dE - b["hi"], dE - b["lo"]          # LOO-calibrated range for this family
        lo_all.append(lo); hi_all.append(hi)
        cells.append((f, dE, lo, hi))
    return loc, cells, min(lo_all), max(hi_all)


# ============================== (1) SCREEN bounds ==============================================
XS, FS = load("SCREEN", keep_full=True)
E_BASE = E_LIVE["SCREEN"]
mids = [(LUT[1:, c] + LUT[:-1, c]) / 2 for c in (0, 1)]
hnp = XS["h"].cpu().numpy()
chn = cht.cpu().numpy()
binn = np.where(chn == 0, np.searchsorted(mids[0], hnp), np.searchsorted(mids[1], hnp))
snap = np.abs(hnp - LUT[binn, chn]).max()
bin_t = torch.from_numpy(binn.astype(np.int64)).to(dev)
print(f"\n[SCREEN] hd/hu max asym {XS['asym']:.2e}; LUT snap max err {snap:.2e}; E_live {E_BASE:.5f}")
ANCH = {}
for f in FAMS:
    th, s = anchor(f, XS, E_BASE)
    ANCH[f] = th
    print(f"  anchor {f}: theta {np.round(FL['fams'][f]['theta'], 4).tolist()} x {s:.4f} -> {np.round(th, 4).tolist()}")
E0_loc = Ehat("LOCAL", None, XS, XS["h"])
print(f"  local W-weighted E (subsample) {E0_loc:.5f}")

res = dict(grid=[], lut={}, lutcal={}, bands=BND)
print("\n## (1a) per-channel global scales on SCREEN bounds: predicted dE_live per family [LOO-calibrated range]")
print("| k_u | k_v | local dE | " + " | ".join(FAMS) + " | range over families (LOO-cal) | E_live range | sps_live range | dfinal range |")
print("|---:|---:|---:|" + "---|" * len(FAMS) + "---|---|---|---|")
for ku in (0.7, 0.8, 0.9, 1.0):
    for kv in (1.0, 1.1, 1.2, 1.35):
        h2 = XS["h"] * torch.where(cht == 0, torch.tensor(ku, device=dev), torch.tensor(kv, device=dev))
        loc, cells, lo, hi = row(f"{ku}/{kv}", XS, h2, ANCH, E0_loc)
        print(f"| {ku} | {kv} | {loc:+.4f} | " + " | ".join(f"{dE:+.4f} [{l:+.4f},{u:+.4f}]" for _, dE, l, u in cells)
              + f" | [{lo:+.4f}, {hi:+.4f}] | [{E_BASE+lo:.4f}, {E_BASE+hi:.4f}] | "
              f"[{100*W_LIVE*(E_BASE+lo):.2f}, {100*W_LIVE*(E_BASE+hi):.2f}] | [{KFIN*lo:+.3f}, {KFIN*hi:+.3f}] |")
        res["grid"].append(dict(ku=ku, kv=kv, local=loc, fams={f: dict(dE=dE, lo=l, hi=u) for f, dE, l, u in cells},
                                lo=lo, hi=hi))

# ---- (1b) per-bin re-optimised LUT, 2-fold by window parity -------------------------------------
HG = torch.tensor(np.geomspace(0.4, 2.5, 161), device=dev, dtype=torch.float32)


def fit_lut(fam, th, X, trainmask):
    new = LUT.copy()
    x_all = xval(fam, th, X) if fam != "LOCAL" else X["d"]
    for c in (0, 1):
        for b in range(24):
            m = (cht == c) & (bin_t == b) & trainmask
            if int(m.sum()) < 2000:
                continue
            hs = float(LUT[b, c]) * HG
            ax = x_all[m].abs()
            w = X["W"][m]
            if fam == "SR":
                P = Phi(torch.log(hs[None, :] / ax.clamp_min(1e-12)[:, None]) / th[1])
                val = (w[:, None] * P).sum(0)
            else:
                o = torch.argsort(ax)
                cw = torch.cumsum(w[o], 0)
                k = torch.searchsorted(ax[o].contiguous(), hs, right=True)
                val = torch.where(k > 0, cw[(k - 1).clamp_min(0)], torch.zeros_like(hs))
            R = torch.exp(-2 * hs / SIG) * val
            new[b, c] = float(hs[int(torch.argmax(R))])
    return new


def apply_lut(lut):
    lt = torch.tensor(lut, device=dev, dtype=torch.float32)
    return lt[bin_t, cht]


print("\n## (1b) per-bin re-optimised LUT (fit on one window-parity fold, scored on the other)")
print("| LUT fitted under | median h ratio u / v | local dE (CV) | " + " | ".join(f"dE under {f}" for f in FAMS)
      + " | range (LOO-cal) | dfinal range |")
print("|---|---|---:|" + "---:|" * len(FAMS) + "---|---|")
for fitfam in ["LOCAL"] + FAMS:
    thf = ANCH.get(fitfam)
    h_cv = torch.empty_like(XS["h"])
    for k in (0, 1):
        lut_k = fit_lut(fitfam, thf, XS, fold == k)
        hk = apply_lut(lut_k)
        h_cv = torch.where(fold == 1 - k, hk, h_cv)
    lut_all = fit_lut(fitfam, thf, XS, torch.ones_like(fold, dtype=torch.bool))
    ru = np.median(lut_all[:, 0] / LUT[:, 0]); rv = np.median(lut_all[:, 1] / LUT[:, 1])
    loc, cells, lo, hi = row(fitfam, XS, h_cv, ANCH, E0_loc)
    print(f"| {fitfam} | {ru:.3f} / {rv:.3f} | {loc:+.4f} | " + " | ".join(f"{dE:+.4f} [{l:+.4f},{u:+.4f}]" for _, dE, l, u in cells)
          + f" | [{lo:+.4f}, {hi:+.4f}] | [{KFIN*lo:+.3f}, {KFIN*hi:+.3f}] |")
    res["lut"][fitfam] = dict(lut=lut_all.tolist(), ratio_u=ru, ratio_v=rv, local=loc,
                              fams={f: dict(dE=dE, lo=l, hi=u) for f, dE, l, u in cells}, lo=lo, hi=hi)

# ---- exact kit check of the base and two grid points (full 734 windows) -------------------------
P, C, HD, HU = FS["P"], FS["C"], FS["HD"], FS["HU"]
ctr = P + C
for ku, kv in ((1.0, 1.0), (0.8, 1.0), (1.0, 1.35)):
    k = np.array([ku, kv], np.float32)
    w_, cov = S.aggregate_sps(P, T, 2, lower=ctr - HD * k, upper=ctr + HU * k)
    print(f"[kit] k_u {ku} k_v {kv}: local sps {100*w_:.4f} coverage {cov:.4f}")
del FS, P, C, HD, HU, ctr

# ============================== (2) LUTCAL decomposition ======================================
XT, _ = load("TMEAN")
XL, _ = load("LUTCAL")
dc = float((XT["d"] - XL["d"]).abs().max())
ru = float(torch.median((XL["h"] / XT["h"])[cht == 0])); rv = float(torch.median((XL["h"] / XT["h"])[cht == 1]))
print(f"\n## (2) LUTCAL vs TMEAN: max|centre diff| {dc:.2e}; median width ratio u {ru:.4f} v {rv:.4f}")
AT = {f: anchor(f, XT, E_LIVE["TMEAN"])[0] for f in FAMS}
E_T = E_LIVE["TMEAN"]
obs = E_LIVE["LUTCAL"] - E_T
variants = {
    "global scale (u x%.3f, v x%.3f)" % (ru, rv): XT["h"] * torch.where(cht == 0, torch.tensor(ru, device=dev), torch.tensor(rv, device=dev)),
    "u-part only (LUTCAL u widths)": torch.where(cht == 0, XL["h"], XT["h"]),
    "v-part only (LUTCAL v widths)": torch.where(cht == 1, XL["h"], XT["h"]),
    "full LUTCAL widths": XL["h"],
}
E_Tloc = Ehat("LOCAL", None, XT, XT["h"])
print(f"observed live dE (LUTCAL - TMEAN) = {obs:+.5f}  ({KFIN*obs:+.3f} final);  vs FP16 {E_LIVE['LUTCAL']-E_LIVE['FP16']:+.5f}")
print("| variant (TMEAN centres + W) | local dE | " + " | ".join(FAMS) + " |")
print("|---|---:|" + "---:|" * len(FAMS))
for name, h in variants.items():
    loc = Ehat("LOCAL", None, XT, h) - E_Tloc
    vals = [Ehat(f, AT[f], XT, h) - E_T for f in FAMS]
    print(f"| {name} | {loc:+.4f} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    res["lutcal"][name] = dict(local=loc, fams=dict(zip(FAMS, vals)))
res["lutcal_observed"] = obs
json.dump(res, open(f"{B}/agents/bounds/grid_predict.json", "w"), indent=1, default=float)
print("[saved] grid_predict.json")
