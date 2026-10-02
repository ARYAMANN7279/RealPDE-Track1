"""Extra figures for the ACAL talk: competitive positioning, and an intuition
picture for the interval score (for listeners without an ML background)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DARK, MID, ACC = "#1A2E4A", "#1F4E79", "#2E86C1"
GREEN, ORANGE, GREY = "#1E8B4C", "#CA6F1E", "#8A94A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#C7CEDB", "axes.labelcolor": "#2C3E50",
                     "xtick.color": "#2C3E50", "ytick.color": "#2C3E50",
                     "axes.spines.top": False, "axes.spines.right": False})

# ── fig4: where we sit in the field ───────────────────────────────────────────
TOP = [("roysegal", 82.10103), ("skabob", 81.98770), ("np-user", 81.96840),
       ("modu-lemon", 81.91526), ("iapetos1918", 81.90253), ("JLShen", 81.86335),
       ("PhysicsOracle", 81.83962), ("doomduke2", 81.76869), ("andychang", 81.75346),
       ("pone7", 81.74229), ("seantang", 81.69702), ("zhoubojian", 81.66706)]
US = 80.081266
LAST50 = 80.43545

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200,
                               gridspec_kw={"width_ratios": [1, 2.4]})

# left: the whole 0-100 scale -- the field is one tight cluster
allv = [v for _, v in TOP] + [LAST50, US]
axL.scatter([0.03] * len(allv), allv, s=42, color=MID, alpha=0.55, zorder=3,
            edgecolor="white", linewidth=0.8)
axL.scatter([0.03], [US], s=120, color=GREEN, zorder=4, edgecolor="white", linewidth=1.4)
axL.set_ylim(0, 100); axL.set_xlim(-0.02, 0.12); axL.set_xticks([])
axL.set_ylabel("final score  (0 - 100)")
axL.grid(axis="y", color="#E6EAF2", lw=1, zorder=0)
axL.set_title("on the full scale", color=DARK, fontweight="bold", fontsize=12)
axL.annotate("every team\nis in here", xy=(0.03, 81), xytext=(0.075, 55),
             fontsize=10.5, color=DARK, fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="->", color=DARK, lw=1.4))

# right: the zoom -- dots only, no stems from a false baseline
names = [n for n, _ in TOP] + ["rank 50", "us"]
vals = [v for _, v in TOP] + [LAST50, US]
xs = np.arange(len(names))
cols = [MID] * 12 + [GREY, GREEN]
axR.axhline(81.74229, color=ORANGE, lw=1.6, ls="--", zorder=1)
axR.text(len(names) - 0.3, 81.755, "top-10 cutoff  81.742", color=ORANGE, fontsize=10,
         fontweight="bold", ha="right", va="bottom")
for x_, v, c in zip(xs, vals, cols):
    axR.plot(x_, v, "o", color=c, ms=13 if c == GREEN else 9, zorder=3,
             markeredgecolor="white", markeredgewidth=1.3)
axR.annotate("82.101", (0, TOP[0][1]), textcoords="offset points", xytext=(0, 13),
             ha="center", fontsize=10.5, fontweight="bold", color=DARK)
axR.annotate("80.061", (xs[-1], US), textcoords="offset points", xytext=(0, 14),
             ha="center", fontsize=11.5, fontweight="bold", color=GREEN)
axR.set_xticks(xs); axR.set_xticklabels(names, rotation=38, ha="right", fontsize=9.5)
axR.set_ylim(79.75, 82.45); axR.set_xlim(-0.8, len(names) - 0.2)
axR.grid(axis="y", color="#E6EAF2", lw=1, zorder=0)
axR.set_title("zoomed in  (note: the axis spans only 2.7 points)",
              color=DARK, fontweight="bold", fontsize=12)
fig.suptitle("All 50 published teams lie within 1.7 points of each other on a 100-point scale",
             color=DARK, fontweight="bold", fontsize=13.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.tight_layout(); fig.savefig(f"{HERE}/fig4_field.png"); plt.close(fig)

# ── fig5: what the interval score rewards ─────────────────────────────────────
SIG = 0.0564
cases = [("narrow, and correct", 0.010, True), ("medium, and correct", 0.030, True),
         ("wide, and correct", 0.100, True), ("narrow, but misses", 0.010, False)]
fig, ax = plt.subplots(figsize=(11.0, 4.3), dpi=200)
truth = 0.0
for i, (lab, wid, ok) in enumerate(cases):
    y = len(cases) - 1 - i
    centre = truth if ok else truth + 0.055
    sc = np.exp(-wid / SIG) * (1.0 if ok else 0.0)
    col = GREEN if (ok and sc > 0.5) else (MID if ok else ORANGE)
    ax.add_patch(Rectangle((centre - wid / 2, y - 0.22), wid, 0.44,
                           facecolor=col, alpha=0.30, edgecolor=col, lw=2, zorder=2))
    ax.hlines(y, centre - wid / 2, centre + wid / 2, color=col, lw=2.5, zorder=3)
    ax.text(-0.115, y, lab, ha="left", va="center", fontsize=12.5, color=DARK, fontweight="bold")
    ax.text(0.125, y, f"score {sc:.3f}", ha="right", va="center", fontsize=12.5,
            color=col, fontweight="bold")
ax.plot([truth] * len(cases), range(len(cases)), "|", color=DARK, ms=26, mew=2.6, zorder=5)
ax.text(truth, len(cases) - 0.52, "true measured value", ha="center", fontsize=11.5,
        color=DARK, fontweight="bold")
ax.set_xlim(-0.118, 0.128); ax.set_ylim(-0.75, len(cases) - 0.25)
ax.set_yticks([]); ax.set_xlabel("velocity  (interval drawn around the prediction)")
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color="#E6EAF2", lw=1, zorder=0)
ax.set_title("The uncertainty score:  exp(−width / σ) if the truth is inside, else 0    (σ = 0.056)",
             color=DARK, fontweight="bold", fontsize=13, pad=14)
fig.tight_layout(); fig.savefig(f"{HERE}/fig5_sps.png"); plt.close(fig)
print("wrote fig4_field.png fig5_sps.png")
