"""Step 3 -- back out live E = sps/(100 W) for every live submission we own, and for the board.

W = 0.5(1-pm_dm) + 0.3(1-pm_tke) + 0.2(1-pm_mvpe), err = 2(100/score - 1), pm = err/(0.5+err).
(W is computed from the AGGREGATE subscores, as the brief specifies; the scorer applies pm per
window, so this W is a slight under-estimate of the element-weighted mean W and E_live a slight
over-estimate -- the bias is common to every row with a similar accuracy.)
Sources: project_memory.md §1 table, §19, §26, §45, §53, §60, §65, §112, §127, §136, §32.2.
"""
import numpy as np


def W_of(rel, tke, mvpe):
    def one(sc):
        e = 2.0 * (100.0 / sc - 1.0)
        return 1.0 - e / (0.5 + e)
    return 0.5 * one(rel) + 0.3 * one(tke) + 0.2 * one(mvpe)


# name, rel_l2, tke, mvpe, sps, backbone md5 (8), bounds policy
LIVE = [
    ("cno (default band)",            88.58, 65.72, 81.68,  5.19, "cno",      "default 0.05|pred|"),
    ("fno (default band)",            94.17, 74.03, 92.84, 14.08, "kit",      "default 0.05|pred|"),
    ("super_surrogate",               94.40, 72.25, 92.75, 13.06, "other",    "default-like"),
    ("fno_calibrated",                93.02, 72.11, 90.37, 18.30, "other",    "calibrated const"),
    ("fno_xgboost",                   75.18, 69.24, 68.37,  1.57, "other",    "xgboost"),
    ("fno_plain_sps",                 94.17, 74.03, 92.84, 29.84, "kit",      "const [.030,.010] centred"),
    ("fno_biasbounds",                94.17, 74.03, 92.84, 23.65, "kit",      "bias-shifted const"),
    ("fno_divfree_gpu",               91.38, 71.75, 86.11, 16.01, "other",    "?"),
    ("hf_fno",                        85.95, 69.37, 73.15,  9.33, "other",    "?"),
    ("ROBUST",                        94.17, 74.03, 92.84, 33.08, "kit",      "per-location LUT x1.15"),
    ("FULLSTACK_v6",                  93.85, 74.97, 92.84, 28.38, "ft",       "13-feat head, bad LUT"),
    ("FULLSTACK_v7",                  93.85, 74.97, 92.84, 33.41, "ft",       "13-feat head, LUT fixed"),
    ("SOUP_v1",                  94.045224, 75.998641, 92.873514, 34.3656,   "dc935158", "13-feat head + 24-bin LUT, centred"),
    ("LUTFIX",                   94.045224, 75.998641, 92.873514, 33.916631, "dc935158", "13-feat head, re-fit LUT (shape)"),
    ("WIDE125",                  94.053, 75.948, 92.8725, 33.8038,           "40c46470", "13-feat head, LUT x1.25"),
    ("ASYM_arcsinh",             94.045224, 75.998641, 92.873514, 27.858126, "dc935158", "asym W96, arcsinh, tight (h_u x0.39)"),
    ("SHIFT_v2_W96_a85",         94.045224, 75.998641, 92.873514, 37.480269, "dc935158", "1 W96 net, centre a=0.85, 24-bin LUT"),
    ("ENSEMBLE_FAST_v3",         94.045224, 75.998641, 92.873514, 37.721250, "dc935158", "3-net centre ens a=0.95, LUT24"),
    ("TMEAN",                    94.072631, 75.998641, 93.141584, 37.803117, "dc935158", "= ENS_FAST_v3 bounds, +tmean pred"),
    ("LUTCAL",                   94.074298, 75.998641, 93.124054, 37.424168, "dc935158", "TIER2D + 64-bin LUT (u x0.87, v x1.43)"),
    ("FP16",                     94.074271, 75.998930, 93.124176, 37.815686, "dc935158", "TIER2D bounds (6e7a6290), fp16 FNO"),
    ("SV2",                      94.050465, 76.900655, 93.130900, 37.887566, "571062ce", "6e7a6290 bounds"),
    ("SCREEN (banked)",          94.131466, 76.425823, 93.158157, 37.923268, "9f5d8261", "6e7a6290 bounds"),
    ("BLIND",                    94.133981, 76.501839, 93.186427, 37.851591, "4238304d", "6e7a6290 bounds"),
    ("W73",                      94.129749, 76.437468, 93.163238, 37.885209, "8d112b78", "6e7a6290 bounds"),
    ("RECIPE2",                  94.114566, 76.125814, 93.109835, 37.849259, "49f855d3", "6e7a6290 bounds"),
]

BOARD = """skabob 94.79 79.19 93.86 44.00
roysegal 94.77 80.01 94.24 44.29
np-user 94.76 79.66 94.13 44.07
iapetos1918 94.85 78.91 94.09 43.72
doomduke2 94.85 78.66 94.14 43.53
zhoubojian 94.68 78.81 94.00 42.84
andychang 94.60 78.97 93.88 43.30
hituc 94.75 79.52 93.28 43.57
seantang 94.55 78.68 94.07 42.78
csasaa 94.64 79.55 93.81 43.34
tangenter 94.67 79.45 93.81 43.30
lsycode 94.71 78.98 94.18 42.86
hihihihi 94.63 79.60 93.78 43.38
xtgg233 94.85 78.68 94.30 41.76
pwm_r2s 94.53 79.30 93.94 42.14
simon-zhou 94.63 77.63 93.51 41.99
julius1 94.44 78.79 93.25 42.86
drozdja 94.74 78.87 93.68 42.09
alvar0 94.75 78.64 93.97 41.01
lghule 94.61 78.54 94.04 41.63
pone7 94.41 78.83 93.99 41.27
phgelado 94.47 77.90 93.07 41.93
necolsy 94.35 78.72 93.20 41.74
redouanelg 94.80 78.54 94.01 40.39
physicsoracle 94.67 77.64 92.96 41.93
dictxxx 94.56 78.60 93.82 42.20
xie233 94.60 79.25 94.14 43.18
rand_politeness 94.63 78.39 93.97 41.51
junlong 94.89 78.07 94.06 40.18
juntao_wang 94.59 79.57 93.83 42.49
xiaomin 94.84 76.82 93.99 41.78
bobajeshjohnson 94.62 77.22 93.58 41.04
anaelle_haomiao 94.79 78.04 93.84 42.07
modu-lemon 94.25 78.18 93.83 41.91
g2404426g 94.24 78.39 93.65 40.00
02loi 94.69 78.07 93.71 40.51
agent33 94.42 77.67 93.67 40.12
zhousheng 94.65 78.27 92.64 39.92
syouya_tobita 94.12 77.69 93.47 40.49
smartparticles 94.24 78.89 93.27 40.64
deleted_user_7805 94.42 79.22 93.84 39.39
zyangastar 94.15 77.19 93.13 40.35
reg0x00 94.36 78.32 93.11 39.73
lzy12301 94.56 77.44 93.23 39.35
ahsdjkahd 94.28 77.55 93.64 39.07
eric_code_2026 94.22 77.11 92.12 39.16
amalss 94.38 79.37 93.60 39.48
kongchun 94.47 77.75 92.61 38.44
orange000 94.46 74.47 93.19 39.16
haidilao 94.46 76.92 93.23 38.07
huao1105 94.44 76.92 93.19 37.92
deleted_user_7875 94.30 78.74 92.98 35.96
rwibawa 94.67 77.83 93.14 35.98
aryamannsr 94.13 76.43 93.16 37.92
zzou1 94.11 77.59 93.73 40.47
kangrui 94.40 78.41 93.62 37.40
sakibs 94.82 75.19 93.90 38.11
ouah7 94.18 74.72 92.92 37.57
changliming 94.26 77.35 93.11 36.96
guoswpeni 94.23 77.19 93.14 37.57"""


def main():
    print("## Live E for every submission we own (E_live = sps / (100 W_agg))\n")
    print("| # | submission | backbone | bounds policy | W_agg | sps | **E_live** |")
    print("|---:|---|---|---|---:|---:|---:|")
    rows = []
    for i, (n, r, t, m, s, bb, pol) in enumerate(LIVE, 1):
        W = W_of(r, t, m)
        E = s / (100 * W)
        rows.append((n, W, E))
        print(f"| {i} | {n} | `{bb}` | {pol} | {W:.5f} | {s:.4f} | **{E:.5f}** |")
    best = max(rows[12:], key=lambda z: z[2])
    print(f"\nBest E_live among fine-tuned-backbone artifacts: {best[0]} {best[2]:.5f}")
    print(f"Best E_live ever: {max(rows, key=lambda z: z[2])}")

    print("\n## Board (top 60) -- E = sps/(100 W)\n")
    recs = []
    for line in BOARD.strip().splitlines():
        p = line.split()
        n, r, t, m, s = p[0], *map(float, p[1:])
        W = W_of(r, t, m)
        recs.append((n, r, t, m, s, W, s / (100 * W)))
    recs_sorted = sorted(recs, key=lambda z: -z[6])
    print("| rank(E) | team | rel_l2 | tke | mvpe | sps | W | E |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for k, z in enumerate(recs_sorted, 1):
        mark = " **(us)**" if z[0] == "aryamannsr" else ""
        print(f"| {k} | {z[0]}{mark} | {z[1]:.2f} | {z[2]:.2f} | {z[3]:.2f} | {z[4]:.2f} | "
              f"{z[5]:.4f} | {z[6]:.4f} |")
    Wv = np.array([z[5] for z in recs]); Ev = np.array([z[6] for z in recs])
    rel = np.array([z[1] for z in recs])
    A = np.c_[np.ones_like(Wv), Wv]
    coef, *_ = np.linalg.lstsq(A, Ev, rcond=None)
    pred = A @ coef
    us = [z for z in recs if z[0] == "aryamannsr"][0]
    print(f"\nOLS  E = {coef[0]:.4f} + {coef[1]:.4f} W   (n={len(Ev)}, r = {np.corrcoef(Wv, Ev)[0,1]:.3f},"
          f" resid sd {np.std(Ev-pred):.4f})")
    print(f"  predicted E at our W {us[5]:.4f}: {coef[0]+coef[1]*us[5]:.4f}; ours {us[6]:.4f};"
          f" our residual {us[6]-(coef[0]+coef[1]*us[5]):+.4f}"
          f" = {(us[6]-(coef[0]+coef[1]*us[5]))/np.std(Ev-pred):+.2f} sd")
    print(f"  corr(E, rel_l2 score) = {np.corrcoef(rel, Ev)[0,1]:.3f}")
    near = [z for z in recs if abs(z[5] - us[5]) <= 0.006]
    print(f"\nTeams with W within +-0.006 of ours ({us[5]:.4f}):")
    for z in sorted(near, key=lambda z: -z[6]):
        print(f"  {z[0]:<18} W {z[5]:.4f}  E {z[6]:.4f}  sps {z[4]:.2f}  (rel {z[1]:.2f} tke {z[2]:.2f} mvpe {z[3]:.2f})")
    print("\nValue of E at our W: d(final)/dE = 0.24737*100*W =", round(0.24737 * 100 * us[5], 3))


if __name__ == "__main__":
    main()
