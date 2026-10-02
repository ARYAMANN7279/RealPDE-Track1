"""Step 4a -- fit live residual-inflation models to EVERY bounds anchor we own, with
leave-one-group-out validation, then test the scale mechanism across backbones.

Data: agents/bounds/cache734/{P,C,HD,HU}_<tag>.npy + TGT.npy -- each live zip's OWN predict() on
the §32 734-window set.  d = t - centre (centre = (lower+upper)/2), r = t - pred, c = centre - pred.
Model:  E_hat(anchor) = sum_i W_s(i) exp(-(hd+hu)/sig) P_i(theta) / sum_i W_s(i)   (scored elements,
W_s from the anchor's own prediction exactly as scoring.py), compared with E_live = sps/(100 W_agg).

Families (theta):
  S1  (lam)            d' = lam d
  S2  (lam_u, lam_v)   d' = lam_ch d
  SL  (lam, rho)       d' = lam d, and a fraction 1-rho of live elements is uncoverable
  SR  (lam, tau)       d' = lam d exp(eta), eta ~ N(0,tau^2) per element  (live width RANKING degrades)
  SP  (a, b)           |d'| = a m_ch (|d|/m_ch)^b  (§53.3 heavy-tail map; m_ch fixed = TMEAN medians)
  SN  (lam, nu)        d' = lam (r - nu c)   (only a fraction nu of the centre skill transfers)
  SRL (lam, tau, rho)
Stage 1: the 7 distinct policies on backbone dc935158 (groups below), LOO by group.
Stage 2: the SAME bounds (6e7a6290) on 6 backbones: with each family's shape fixed, solve the scale
         per backbone and compare it with that backbone's live/local rel_l2 error ratio.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from scipy.optimize import brentq, minimize

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
D = f"{B}/agents/bounds/cache734"
SIG = 0.0563870259
SQ2 = math.sqrt(2.0)

LIVE = {  # tag: (rel_l2, tke, mvpe, sps)  -- live subscores (see live_E_table.py for sources)
    "SOUPv1": (94.045224, 75.998641, 92.873514, 34.3656),
    "LUTFIX": (94.045224, 75.998641, 92.873514, 33.916631),
    "WIDE125": (94.053, 75.948, 92.8725, 33.8038),
    "ARCSINH": (94.045224, 75.998641, 92.873514, 27.858126),
    "SHIFT85": (94.045224, 75.998641, 92.873514, 37.480269),
    "ENSFASTv3": (94.045224, 75.998641, 92.873514, 37.721250),
    "TMEAN": (94.072631, 75.998641, 93.141584, 37.803117),
    "FP16": (94.074271, 75.998930, 93.124176, 37.815686),
    "LUTCAL": (94.074298, 75.998641, 93.124054, 37.424168),
    "SV2": (94.050465, 76.900655, 93.130900, 37.887566),
    "SCREEN": (94.131466, 76.425823, 93.158157, 37.923268),
    "BLIND": (94.133981, 76.501839, 93.186427, 37.851591),
    "W73": (94.129749, 76.437468, 93.163238, 37.885209),
    "RECIPE2": (94.114566, 76.125814, 93.109835, 37.849259),
}
GROUPS = [["SOUPv1"], ["LUTFIX"], ["WIDE125"], ["ARCSINH"], ["SHIFT85"],
          ["ENSFASTv3", "TMEAN", "FP16"], ["LUTCAL"]]
BACKBONES = ["FP16", "SV2", "SCREEN", "BLIND", "W73", "RECIPE2"]


def W_agg(rel, tke, mvpe):
    def one(sc):
        e = 2.0 * (100.0 / sc - 1.0)
        return 1.0 - e / (0.5 + e)
    return 0.5 * one(rel) + 0.3 * one(tke) + 0.2 * one(mvpe)


E_LIVE = {k: v[3] / (100.0 * W_agg(*v[:3])) for k, v in LIVE.items()}

ap = argparse.ArgumentParser()
ap.add_argument("--sub", type=int, default=12_000_000)
ap.add_argument("--out", default="fit_live.json")
ap.add_argument("--maxiter", type=int, default=600)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def kit():
    import importlib.util as iu
    KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
    sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
    S = iu.module_from_spec(sp)
    sp.loader.exec_module(S)
    return S


S = kit()
T = np.load(f"{D}/TGT.npy")
N = T.shape[0]
sc_flat = (T != 0).reshape(-1)
idx_all = np.nonzero(sc_flat)[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(idx_all, size=min(a.sub, len(idx_all)), replace=False))
per_win = int(np.prod(T.shape[1:]))
win = idx // per_win
ch = (idx % 2).astype(np.int64)
t_sub = T.reshape(-1)[idx]
print(f"[data] windows {N}, scored {len(idx_all)}, subsample {len(idx)}", flush=True)

tags = [t for t in LIVE if os.path.exists(f"{D}/HD_{t}.npy")]
missing = [t for t in LIVE if t not in tags]
print("[anchors] present:", tags, " missing:", missing, flush=True)

A = {}
n_ = lambda x: x / (0.5 + x)
for t in tags:
    P = np.load(f"{D}/P_{t}.npy")
    Wwin = (0.5 * (1 - n_(S.rel_l2_per_sample(P, T, 2))) + 0.3 * (1 - n_(S.tke_rel_l2_per_sample(P, T, 2)))
            + 0.2 * (1 - n_(S.mvpe_rel_l2_per_sample(P, T))))
    err_loc = float(np.mean(S.rel_l2_per_sample(P, T, 2)))
    p = P.reshape(-1)[idx]
    del P
    c = np.load(f"{D}/C_{t}.npy").reshape(-1)[idx]
    hd = np.load(f"{D}/HD_{t}.npy").reshape(-1)[idx]
    hu = np.load(f"{D}/HU_{t}.npy").reshape(-1)[idx]
    d = t_sub - (p + c)
    r = t_sub - p
    wel = Wwin[win].astype(np.float32)
    g = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
    A[t] = dict(d=g(d), r=g(r), c=g(c), hd=g(hd), hu=g(hu), wp=g(wel * np.exp(-(hd + hu) / SIG)),
                den=float(wel.sum()), err_loc=err_loc, Wmean=float(Wwin.mean()))
    ins = ((d >= -hd) & (d <= hu))
    A[t]["E_loc"] = float((wel * np.exp(-(hd + hu) / SIG) * ins).sum() / wel.sum())
    print(f"  {t:10s} E_loc {A[t]['E_loc']:.5f}  E_live {E_LIVE[t]:.5f}  ratio {E_LIVE[t]/A[t]['E_loc']:.4f}"
          f"  med h_u {np.median(0.5*(hd+hu)[ch == 0]):.5f} h_v {np.median(0.5*(hd+hu)[ch == 1]):.5f}"
          f"  err_loc {err_loc:.5f}", flush=True)
cht = torch.from_numpy(ch).to(dev)
ref = "TMEAN" if "TMEAN" in A else tags[0]
MED = [float(A[ref]["d"][cht == k].abs().median()) for k in (0, 1)]
MEDt = torch.tensor(MED, device=dev)


def Phi(z):
    return 0.5 * (1.0 + torch.erf(z / SQ2))


def pin(fam, th, X):
    d, hd, hu = X["d"], X["hd"], X["hu"]
    if fam == "S1":
        x = th[0] * d
        return ((x >= -hd) & (x <= hu)).float()
    if fam == "S2":
        lam = torch.where(cht == 0, torch.tensor(th[0], device=dev), torch.tensor(th[1], device=dev))
        x = lam * d
        return ((x >= -hd) & (x <= hu)).float()
    if fam == "SL":
        x = th[0] * d
        return th[1] * ((x >= -hd) & (x <= hu)).float()
    if fam in ("SR", "SRL"):
        ad = d.abs().clamp_min(1e-12)
        hh = torch.where(d >= 0, hu, hd)
        P = Phi(torch.log(hh / (th[0] * ad)) / th[1])
        return P * (th[2] if fam == "SRL" else 1.0)
    if fam == "SP":
        m = MEDt[cht]
        x = torch.sign(d) * th[0] * m * (d.abs() / m) ** th[1]
        return ((x >= -hd) & (x <= hu)).float()
    if fam == "SN":
        x = th[0] * (X["r"] - th[1] * X["c"])
        return ((x >= -hd) & (x <= hu)).float()
    raise ValueError(fam)


def Ehat(fam, th, t):
    X = A[t]
    return float((X["wp"] * pin(fam, th, X)).sum()) / X["den"]


FAMS = {  # name: (x0 (raw), to_theta)
    "S1": ([math.log(2.0)], lambda x: [math.exp(x[0])]),
    "S2": ([math.log(2.0)] * 2, lambda x: [math.exp(x[0]), math.exp(x[1])]),
    "SL": ([math.log(1.8), 2.0], lambda x: [math.exp(x[0]), 1 / (1 + math.exp(-x[1]))]),
    "SR": ([math.log(1.8), math.log(0.5)], lambda x: [math.exp(x[0]), math.exp(x[1])]),
    "SP": ([math.log(1.2), math.log(1.5)], lambda x: [math.exp(x[0]), math.exp(x[1])]),
    "SN": ([math.log(2.0), 1.0], lambda x: [math.exp(x[0]), x[1]]),
    "SRL": ([math.log(1.8), math.log(0.5), 2.0],
            lambda x: [math.exp(x[0]), math.exp(x[1]), 1 / (1 + math.exp(-x[2]))]),
}


def fit(fam, fit_tags, tries=3):
    x0, tt = FAMS[fam]

    def obj(x):
        th = tt(x)
        if any((not np.isfinite(v)) for v in th) or th[0] > 20:
            return 1e3
        return sum((Ehat(fam, th, t) - E_LIVE[t]) ** 2 for t in fit_tags)
    best = None
    r_ = np.random.default_rng(11)
    for k in range(tries):
        xs = np.array(x0) + (0 if k == 0 else r_.normal(0, 0.4, len(x0)))
        res = minimize(obj, xs, method="Nelder-Mead",
                       options=dict(maxiter=a.maxiter, xatol=1e-5, fatol=1e-12))
        if best is None or res.fun < best.fun:
            best = res
    return tt(best.x), best.fun


stage1 = [t for g in GROUPS for t in g if t in A]
out = dict(E_live=E_LIVE, E_loc={t: A[t]["E_loc"] for t in A}, med_ref=MED, fams={})
K = 0.24737 * 100 * 0.6854       # final points per unit E on the dc935158 backbone
print(f"\n=== STAGE 1: {len(stage1)} anchors on dc935158, {len(GROUPS)} policy groups ===", flush=True)
for fam in FAMS:
    th, f = fit(fam, stage1)
    res_in = {t: Ehat(fam, th, t) - E_LIVE[t] for t in stage1}
    loo = {}
    for g in GROUPS:
        gg = [t for t in g if t in A]
        if not gg:
            continue
        th_g, _ = fit(fam, [t for t in stage1 if t not in gg], tries=2)
        for t in gg:
            loo[t] = Ehat(fam, th_g, t) - E_LIVE[t]
    mi = max(abs(v) for v in res_in.values())
    ml = max(abs(v) for v in loo.values())
    print(f"\n[{fam}] theta {np.round(th, 4).tolist()}  in-sample max|res| {mi:.5f} ({mi*K:.3f} final)"
          f"  LOO max|res| {ml:.5f} ({ml*K:.3f} final)", flush=True)
    print("   " + "  ".join(f"{t}:{res_in[t]:+.4f}/{loo.get(t, float('nan')):+.4f}" for t in stage1), flush=True)
    out["fams"][fam] = dict(theta=list(map(float, th)), res_in=res_in, loo=loo, max_in=mi, max_loo=ml)

print("\n=== STAGE 2: same bounds (6e7a6290), six backbones: solve the scale, shape fixed ===", flush=True)
bb = [t for t in BACKBONES if t in A]
for fam in ("S1", "SL", "SR", "SP", "SRL"):
    th0 = out["fams"][fam]["theta"]
    rows = []
    for t in bb:
        def g_(s):
            th = list(th0)
            th[0] = s
            return Ehat(fam, th, t) - E_LIVE[t]
        try:
            s = brentq(g_, 0.05, 15.0, xtol=1e-5)
        except ValueError:
            s = float("nan")
        e_live = 2.0 * (100.0 / LIVE[t][0] - 1.0)
        rows.append((t, s, e_live / A[t]["err_loc"], A[t]["err_loc"], e_live))
    print(f"[{fam}] shape {np.round(th0[1:], 4).tolist()}: backbone, fitted scale, live/local rel_l2 ratio")
    for r_ in rows:
        print(f"    {r_[0]:8s} scale {r_[1]:.4f}   rel_l2 ratio {r_[2]:.4f}  (local {r_[3]:.5f} live {r_[4]:.5f})")
    sv = np.array([r_[1] for r_ in rows]); rv = np.array([r_[2] for r_ in rows])
    ok = np.isfinite(sv)
    if ok.sum() >= 3:
        print(f"    corr(scale, ratio) = {np.corrcoef(sv[ok], rv[ok])[0,1]:+.3f};  scale/ratio "
              f"mean {np.mean(sv[ok]/rv[ok]):.3f} sd {np.std(sv[ok]/rv[ok]):.3f}")
    out.setdefault("stage2", {})[fam] = rows
json.dump(out, open(a.out, "w"), indent=1, default=float)
print("[saved]", a.out)
