"""Step 2b -- calibrate the heteroscedastic scale model of analytic_E.py to OUR residuals.

Inputs (one of):
  --r20 PATH     train_es/r20_bounds_cache.npz (SV2 stack, 900 re_lohi windows): P, C, H, T, pm_*
  --c734 TAG     agents/bounds/cache734/{P,C,HD,HU}_TAG.npy + TGT.npy  (C = centre - pred)
Per channel it reports
  * residual about the bound centre d = t - centre: med|d|, tail ratios, Var(log|d|), kurtosis
  * E of the cached policy, coverage, the local kit sps (self-test vs the recorded value)
  * E_const (best single half-width), E_feat (best half-width per LUT level = in-sample ceiling of
    ANY 1-D LUT on our feature), E_realised-oracle (h = |d|)
  * within-LUT-bin tail heaviness (what conditioning on our feature leaves)
  * the scale model: sl^2 = Var(log|d|) - Var(log|Z|) per family, its tail-ratio check, the implied
    scale-oracle E (rho = 1), constant E (rho = 0), and the rho_eff that reproduces E_feat
  * live inflation: E of the cached policy scaled by k when residuals are inflated by lam
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_E import FAMS, SIG, cover_sym, logabs_var  # noqa: E402

B = "/SML_DISK_24TB/rajeshr/Aryamann/UGP"
ap = argparse.ArgumentParser()
ap.add_argument("--r20", default="")
ap.add_argument("--c734", default="")
ap.add_argument("--setid", type=int, default=-1, help="c734 only: 0 = re_lohi windows")
ap.add_argument("--sub", type=int, default=16_000_000)
ap.add_argument("--out", default="")
a = ap.parse_args()
rng = np.random.default_rng(0)


def kit():
    import importlib.util as iu
    KIT = f"{B}/starting_kit_v9/realpde_t1_starting_kit_v9"
    sp = iu.spec_from_file_location("scoring", os.path.join(KIT, "scoring.py"))
    S = iu.module_from_spec(sp)
    sp.loader.exec_module(S)
    return S


def load():
    if a.r20:
        z = np.load(a.r20)
        P, C, H, T = z["P"], z["C"], z["H"], z["T"]
        # r20's pm_* fields are the RAW per-window errors (e), not pm = e/(0.5+e): using them as pm
        # gave W 0.80 and local sps 53.89 vs the recorded 50.8059 (coverage matched exactly).
        e = np.stack([z["pm_dm"], z["pm_tke"], z["pm_mvpe"]], 1).astype(np.float64)
        pm = e / (0.5 + e)
        s1 = float(np.median(np.abs(C[:4]))); s2 = float(np.median(np.abs(C[:4] - P[:4])))
        Cabs = C if s2 < s1 else P + C
        print(f"[load] r20: median|C| {s1:.5f} median|C-P| {s2:.5f} -> C is "
              f"{'ABSOLUTE centre' if s2 < s1 else 'offset'}")
        return P, Cabs, H, H, T, pm, None
    D = f"{B}/agents/bounds/cache734"
    P = np.load(f"{D}/P_{a.c734}.npy")
    Cabs = P + np.load(f"{D}/C_{a.c734}.npy")
    HD = np.load(f"{D}/HD_{a.c734}.npy")
    HU = np.load(f"{D}/HU_{a.c734}.npy")
    T = np.load(f"{D}/TGT.npy")
    S = kit()
    n = lambda x: x / (0.5 + x)
    pm = np.stack([n(S.rel_l2_per_sample(P, T, 2)), n(S.tke_rel_l2_per_sample(P, T, 2)),
                   n(S.mvpe_rel_l2_per_sample(P, T))], 1).astype(np.float64)
    ws = np.load(f"{B}/_agent2/winset.npz")
    return P, Cabs, HD, HU, T, pm, ws["setid"]


P, Cabs, HD, HU, T, pm, setid = load()
import io  # noqa: E402
import zipfile  # noqa: E402
with zipfile.ZipFile(f"{B}/submissions/submission_SCREEN.zip") as _zf:
    LUT = np.load(io.BytesIO(_zf.read("bounds_assets.npz")))["LUT"]   # (24,2), bounds 6e7a6290
N = P.shape[0]
Wwin = 0.5 * (1 - pm[:, 0]) + 0.3 * (1 - pm[:, 1]) + 0.2 * (1 - pm[:, 2])
winsel = np.ones(N, bool) if (setid is None or a.setid < 0) else (setid == a.setid)
print(f"[data] windows {N}, used {int(winsel.sum())}; W_s mean {Wwin[winsel].mean():.5f}")

# ---- exact local kit sps of the cached policy (both channels, element-weighted) ----------
lo = Cabs - HD
up = Cabs + HU
sc = (T != 0) & winsel[:, None, None, None, None]
ins = (T >= lo) & (T <= up)
pen = np.exp(-(up - lo) / SIG)
Wel = np.broadcast_to(Wwin[:, None, None, None, None], T.shape)
n_sc = int(sc.sum())
sps_loc = 100.0 * float(np.sum(Wel * pen * ins, where=sc, dtype=np.float64)) / n_sc
cov_loc = float(np.count_nonzero(ins & sc)) / n_sc
E_W = float(np.sum(Wel * pen * ins, where=sc, dtype=np.float64) / np.sum(Wel, where=sc,
                                                                             dtype=np.float64))
print(f"[kit] local sps {sps_loc:.4f}  coverage {cov_loc:.4f}  W-weighted E {E_W:.5f}  "
      f"W_elem-mean {np.mean(Wel[sc]):.5f}")
res = dict(sps_local=sps_loc, coverage=cov_loc, E_W=E_W, ch={})
del lo, up, ins, pen, Wel

HG = np.geomspace(1e-5, 0.08, 2400)


def best_h(absd_sorted):
    F = np.searchsorted(absd_sorted, HG, side="right") / max(len(absd_sorted), 1)
    R = np.exp(-2 * HG / SIG) * F
    j = int(np.argmax(R))
    return float(HG[j]), float(R[j])


VZ = {k: logabs_var(f) for k, f in FAMS.items()}
for ci, cn in ((0, "u"), (1, "v")):
    m = sc[..., ci]
    d = (T[..., ci] - Cabs[..., ci])[m].astype(np.float64)
    hd = HD[..., ci][m].astype(np.float64)
    hu = HU[..., ci][m].astype(np.float64)
    h = 0.5 * (hd + hu)
    asym = float(np.max(np.abs(hd - hu)))
    ad = np.abs(d)
    n = len(d)
    E_pol = float(np.mean(np.exp(-2 * h / SIG) * ((d >= -hd) & (d <= hu))))
    cov = float(np.mean((d >= -hd) & (d <= hu)))
    ads = np.sort(ad)
    hc, E_const = best_h(ads)
    E_real = float(np.mean(np.exp(-2 * ad / SIG)))
    med = float(np.median(ad))
    q = np.quantile(ad, [0.5, 0.9, 0.99, 0.999])
    kurt = float(np.mean(d ** 4) / np.mean(d ** 2) ** 2)
    nz = ad > 0
    lad = np.log(ad[nz])
    v_lad = float(np.var(lad))
    # ---- per LUT level: in-sample optimal half-width on our feature ----------------------
    # snap every half-width to its LUT level (h from lower/upper carries ~1e-8 float32 noise that
    # correlates with |centre| and would otherwise split each level into spurious sub-bins)
    lut = LUT[:, ci].astype(np.float64)
    li = np.abs(h[:, None] - lut[None, :]).argmin(1) if len(h) < 2 else \
        np.clip(np.searchsorted((lut[1:] + lut[:-1]) / 2, h), 0, len(lut) - 1)
    snap_err = float(np.max(np.abs(h - lut[li])))
    print(f"  [snap] max|h - LUT level| = {snap_err:.2e} (must be ~1e-7; otherwise the policy is not the 24-bin LUT)")
    lev, inv = lut, li
    E_feat = 0.0
    rows = []
    for b in range(len(lev)):
        sel = inv == b
        nb = int(sel.sum())
        if nb < 200:
            E_feat += float(np.sum(np.exp(-2 * h[sel] / SIG) * (ad[sel] <= h[sel]))) / n
            continue
        adb = np.sort(ad[sel])
        hb, Rb = best_h(adb)
        db = d[sel]
        Eb_pol = float(np.mean(np.exp(-2 * lev[b] / SIG) * (adb <= lev[b])))
        madb = float(np.median(np.abs(db - np.median(db))))
        rows.append((lev[b], nb / n, float(np.median(adb)), float(np.std(db)) / (1.4826 * madb + 1e-30),
                     float(np.mean((db - db.mean()) ** 4) / np.var(db) ** 2), float(np.mean(adb <= lev[b])),
                     Eb_pol, hb, Rb, hb / lev[b]))
        E_feat += Rb * nb / n
    # ---- correlations (subsample) ---------------------------------------------------------
    k = rng.choice(n, size=min(a.sub, n), replace=False)
    from scipy import stats
    rs = float(stats.spearmanr(h[k], ad[k]).correlation)
    lh = np.log(h[k]); la = np.log(np.maximum(ad[k], 1e-12))
    rp = float(np.corrcoef(lh, la)[0, 1])
    print(f"\n=== channel {cn}: n_scored {n}  (hd/hu max asym {asym:.2e})")
    print(f"  med|d| {med:.6f}  q90/q50 {q[1]/q[0]:.2f}  q99/q50 {q[2]/q[0]:.2f}  q999/q50 {q[3]/q[0]:.2f}"
          f"  kurtosis {kurt:.1f}  Var(log|d|) {v_lad:.4f}")
    print(f"  E_policy {E_pol:.5f}  coverage {cov:.4f}  | E_const {E_const:.5f} (h={hc:.5f})"
          f"  E_feat(best per LUT level) {E_feat:.5f}  E_realised_oracle {E_real:.5f}")
    print(f"  spearman(h,|d|) {rs:.4f}  pearson(log h, log|d|) {rp:.4f}")
    print("  LUT level | share | med|d| | sd/(1.48MAD) | kurt | cover | E_pol_bin | h_opt | E_opt_bin | h_opt/h_LUT")
    for r in rows:
        print("  %.5f | %.4f | %.5f | %.2f | %.1f | %.3f | %.4f | %.5f | %.4f | %.2f" % r)
    # ---- scale model per family ------------------------------------------------------------
    fam_res = {}
    for fn, fam in FAMS.items():
        sl2 = v_lad - VZ[fn]
        if sl2 <= 0.01:
            print(f"  [{fn}] Var(log|d|) {v_lad:.3f} <= Var(log|Z|) {VZ[fn]:.3f}: family inconsistent")
            fam_res[fn] = None
            continue
        sl = float(np.sqrt(sl2))
        zz = fam.rvs(size=2_000_000, random_state=np.random.default_rng(5))
        ss = np.exp(sl * np.random.default_rng(6).standard_normal(2_000_000))
        sim = np.abs(ss * zz)
        mq = np.quantile(sim, [0.5, 0.9, 0.99, 0.999])
        k_med = med / mq[0] / SIG           # s_med / sig so the model median matches
        c_or = sl / np.sqrt(sl2 + VZ[fn])
        rho_corr = rp / c_or if c_or > 0 else float("nan")
        from analytic_E import mixture_E, realized_oracle
        rhos = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
        Es = [mixture_E(fam, k_med, sl, r, n_gh=40, n_h=1400)[0] for r in rhos]
        Ero = realized_oracle(fam, k_med, sl)
        # rho that reproduces E_feat
        if E_feat <= Es[0]:
            rho_E = 0.0
        elif E_feat >= Es[-1]:
            rho_E = float("inf")
        else:
            rho_E = float(np.interp(E_feat, Es, rhos))
        print(f"  [{fn}] sl {sl:.3f}  tail model q90/50 {mq[1]/mq[0]:.2f} q99/50 {mq[2]/mq[0]:.2f}"
              f" q999/50 {mq[3]/mq[0]:.2f}  k_med {k_med:.4f}  corr_oracle {c_or:.3f}"
              f"  rho_from_corr {rho_corr:.3f}  rho_from_E {rho_E:.3f}")
        print("     E(rho): " + "  ".join(f"{r:g}:{e:.4f}" for r, e in zip(rhos, Es))
              + f"   model realised-oracle {Ero:.4f}")
        fam_res[fn] = dict(sl=sl, k_med=k_med, tail=[float(x / mq[0]) for x in mq],
                           corr_oracle=c_or, rho_corr=rho_corr, rho_E=rho_E,
                           E_rho=dict(zip(map(str, rhos), Es)), E_realised_oracle=Ero)
    # ---- live inflation surface for the cached policy (subsample) ---------------------------
    ks = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75]
    lams = [1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]
    hk, dk = h[k], ad[k]
    print("  E(k*h ; residual x lam)   rows lam, cols k;  last col = argmax k")
    surf = {}
    for lam in lams:
        row = [float(np.mean(np.exp(-2 * kk * hk / SIG) * (lam * dk <= kk * hk))) for kk in ks]
        surf[str(lam)] = row
        print(f"   lam {lam:>4}: " + " ".join(f"{v:.4f}" for v in row) + f"   k*={ks[int(np.argmax(row))]}")
    res["ch"][cn] = dict(n=n, med=med, tail=[float(x / q[0]) for x in q], kurt=kurt, var_logabs=v_lad,
                         E_policy=E_pol, coverage=cov, E_const=E_const, h_const=hc, E_feat=E_feat,
                         E_realised_oracle=E_real, spearman=rs, pearson_log=rp, bins=rows,
                         families=fam_res, inflation_surface=dict(k=ks, lam=surf))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("[saved]", a.out)
