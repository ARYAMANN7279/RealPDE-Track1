"""Figures for the revised talk: the time-score curve, and the anatomy of a submission."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
DARK, MID, ACC = "#1A2E4A", "#1F4E79", "#2E86C1"
GREEN, ORANGE, GREY = "#1E8B4C", "#CA6F1E", "#8A94A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#C7CEDB", "axes.labelcolor": "#2C3E50",
                     "xtick.color": "#2C3E50", "ytick.color": "#2C3E50",
                     "axes.spines.top": False, "axes.spines.right": False})

# ── fig8: why runtime is scored the way it is ────────────────────────────────
t = np.linspace(0.5, 300, 2000)
score = 100.0 / (1.0 + np.sqrt(t / 728.96))
OURS = 7.80
ours_s = 100.0 / (1.0 + np.sqrt(OURS / 728.96))
fig, ax = plt.subplots(figsize=(11.0, 4.3), dpi=200)
ax.plot(t, score, "-", color=ACC, lw=2.6, zorder=3)
ax.axvline(300, color=ORANGE, lw=2, ls="--", zorder=2)
ax.text(292, 62, "hard limit\n300 s", color=ORANGE, fontsize=11, fontweight="bold", ha="right")
ax.plot(OURS, ours_s, "o", color=GREEN, ms=14, zorder=5, markeredgecolor="white", markeredgewidth=1.6)
ax.annotate(f"us:  {OURS:.1f} s  →  {ours_s:.1f} / 100", (OURS, ours_s),
            textcoords="offset points", xytext=(34, -6), fontsize=12.5, fontweight="bold",
            color=GREEN, va="center")
for tt, lab in ((60, "1 min"), (150, "2.5 min")):
    ss = 100.0 / (1.0 + np.sqrt(tt / 728.96))
    ax.plot(tt, ss, "o", color=GREY, ms=7, zorder=4)
    ax.annotate(f"{lab} → {ss:.0f}", (tt, ss), textcoords="offset points", xytext=(8, 10),
                fontsize=10.5, color=GREY)
ax.set_xscale("log")
ax.set_xlabel("runtime  (seconds, log scale)"); ax.set_ylabel("time score  (out of 100)")
ax.set_ylim(55, 102); ax.grid(color="#E6EAF2", lw=1, zorder=0)
ax.set_title("The curve is flat where we sit — further speed-ups buy almost nothing",
             color=DARK, fontweight="bold", fontsize=13, pad=12)
fig.tight_layout(); fig.savefig(f"{HERE}/fig8_time.png"); plt.close(fig)

# ── fig9: what is actually inside the submitted package ──────────────────────
fig, ax = plt.subplots(figsize=(11.2, 4.4), dpi=200)
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
parts = [("FNO backbone\n(the forecaster)", 201.4, MID),
         ("3 bounds U-Nets\n(uncertainty + re-centring)", 42.5, ACC),
         ("TKE head", 6.5, GREEN),
         ("submission.py\n+ vendored libraries", 0.5, GREY)]
total = 268.4
x = 0.35
WIDTH = 9.3
for lab, mb, c in parts:
    w = WIDTH * mb / total
    ax.add_patch(Rectangle((x, 3.15), w, 1.15, facecolor=c, alpha=0.85, edgecolor="white", lw=2))
    if mb > 20:
        ax.text(x + w / 2, 3.72, f"{mb:.0f} MB", ha="center", va="center", color="white",
                fontsize=13, fontweight="bold")
        ax.text(x + w / 2, 2.85, lab, ha="center", va="top", fontsize=11.5, color=DARK)
    x += w
free = total - sum(p[1] for p in parts)
wf = WIDTH * free / total
ax.add_patch(Rectangle((x, 3.15), wf, 1.15, facecolor="none", edgecolor=ORANGE, lw=2.2, ls="--"))
ax.text(x + wf / 2, 3.72, f"{free:.0f}\nMB", ha="center", va="center", color=ORANGE,
        fontsize=10.5, fontweight="bold")
ax.text(x + wf / 2, 2.85, "free", ha="center", va="top", fontsize=11, color=ORANGE)
ax.annotate("", xy=(0.35, 4.75), xytext=(9.65, 4.75),
            arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.8))
ax.text(5.0, 4.95, "256 MB hard cap  —  we use 93.5% of it", ha="center", fontsize=13,
        fontweight="bold", color=DARK)
ax.text(5.0, 1.95, "The package must expose one function:", ha="center", fontsize=12.5, color=DARK)
ax.text(5.0, 1.40, "predict(input)  →  { prediction,  lower,  upper }", ha="center",
        fontsize=15, fontweight="bold", color=MID, family="DejaVu Sans Mono")
ax.text(5.0, 0.72, "Graders call it in an isolated container: PyTorch and NumPy only, no internet, 5-minute limit.",
        ha="center", fontsize=11.5, color=GREY)
ax.set_title("Anatomy of a submission", color=DARK, fontweight="bold", fontsize=14, pad=6)
fig.tight_layout(); fig.savefig(f"{HERE}/fig9_zip.png"); plt.close(fig)
print("wrote fig8_time.png fig9_zip.png")
