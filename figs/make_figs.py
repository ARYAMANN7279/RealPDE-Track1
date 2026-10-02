"""Figures for the UGP lab presentation. Palette matches the AE646 deck exactly."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DARK, MID, ACC = "#1A2E4A", "#1F4E79", "#2E86C1"
GREEN, ORANGE, GREY = "#1E8B4C", "#CA6F1E", "#8A94A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#C7CEDB", "axes.labelcolor": "#2C3E50",
                     "xtick.color": "#2C3E50", "ytick.color": "#2C3E50",
                     "axes.spines.top": False, "axes.spines.right": False})

# ── fig1: the climb ───────────────────────────────────────────────────────────
runs = [("SOUP_v1", 78.450), ("SHIFT_v2", 79.2464), ("ENSEMBLE_v3", 79.3418),
        ("TMEAN", 79.3695), ("FP16", 79.3789), ("SV2", 79.4626), ("SCREEN", 79.4844),
        ("SPEED_SAFE", 79.4870), ("FITA1B", 79.4939), ("FA1BMT", 79.5907),
        ("FF10MLS48", 79.6026), ("FA1BMLS48", 79.6567), ("FA1BMLOT7", 79.7535),
        ("LUT150", 79.8574), ("NARROW080", 80.0613), ("STACK", 80.0813)]
fig, ax = plt.subplots(figsize=(11.2, 4.5), dpi=200)
x = np.arange(len(runs)); y = [v for _, v in runs]
ax.axhline(80.0, color=GREEN, lw=2, ls="--", zorder=1)
ax.text(0.15, 80.012, "target: 80.0", color=GREEN, fontsize=11, fontweight="bold", va="bottom")
ax.plot(x, y, "-", color=ACC, lw=2, zorder=2)
ax.plot(x, y, "o", color=MID, ms=7, zorder=3)
ax.plot(x[-1], y[-1], "o", color=GREEN, ms=12, zorder=4)
for i, (n, v) in enumerate(runs):
    if n in ("SOUP_v1", "SHIFT_v2", "SV2", "FA1BMT", "LUT150", "STACK"):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=10, fontweight="bold", color=DARK)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in runs], rotation=38, ha="right", fontsize=9)
ax.set_ylabel("live final score"); ax.set_ylim(78.2, 80.35)
ax.grid(axis="y", color="#E6EAF2", lw=1)
ax.set_title("Live score, 23 Aug → 23 Sep  (only submissions that improved)",
             color=DARK, fontweight="bold", fontsize=13, pad=12)
fig.tight_layout(); fig.savefig(f"{HERE}/fig1_climb.png"); plt.close(fig)

# ── fig2: where the gap to the leader actually is ─────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
labels = ["us\n80.061", "+ their model\n(residuals −11.9%)", "+ their bounds\n(84%→91% of oracle)"]
vals = [80.0613, 81.2470, 82.1010]
cols = [MID, ACC, ORANGE]
b = ax.bar(labels, vals, color=cols, width=0.62, zorder=3)
ax.axhline(81.74229, color=GREEN, lw=2, ls="--", zorder=4)
ax.text(2.42, 81.76, "top-10 cutoff\n81.742", color=GREEN, fontsize=10,
        fontweight="bold", va="bottom", ha="right")
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center",
            fontsize=11, fontweight="bold", color=DARK)
ax.annotate("", xy=(1, 81.20), xytext=(0, 80.11),
            arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))
ax.text(0.5, 80.86, "+1.180\nmodel", ha="center", fontsize=10, color=DARK, fontweight="bold")
ax.annotate("", xy=(2, 82.05), xytext=(1, 81.30),
            arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))
ax.text(1.5, 81.76, "+0.854\nbounds", ha="center", fontsize=10, color=DARK, fontweight="bold")
ax.set_ylim(79.6, 82.55); ax.set_ylabel("final score")
ax.grid(axis="y", color="#E6EAF2", lw=1, zorder=0)
ax.set_title("Decomposing the 2.04 gap to rank 1: ~58% model, ~42% bounds",
             color=DARK, fontweight="bold", fontsize=12.5, pad=12)
fig.tight_layout(); fig.savefig(f"{HERE}/fig2_gap.png"); plt.close(fig)

# ── fig3: the calibrated proxy — rel_l2 has no resolution ─────────────────────
DATA = {
    "rel_l2": ([95.5076, 95.5241, 95.5096, 95.5239, 95.5351, 95.5351, 95.5351],
               [94.0505, 94.0743, 94.0726, 94.0743, 93.9770, 93.9767, 93.9767]),
    "tke":    ([81.4454, 80.5756, 80.5755, 80.5755, 82.5418, 82.5418, 82.5418],
               [76.9007, 75.9989, 75.9986, 75.9986, 78.6230, 78.6227, 78.6227]),
    "sps":    ([50.8437, 50.6697, 50.6268, 50.0150, 52.1458, 49.8136, 53.7349],
               [37.8876, 37.8157, 37.8031, 37.4242, 38.7200, 36.9184, 39.5581]),
}
VERD = {"rel_l2": ("slope −3.22  →  UNIDENTIFIABLE", ORANGE),
        "tke": ("slope +1.34  →  usable", MID),
        "sps": ("slope +0.62  →  well calibrated", GREEN)}
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), dpi=200)
for ax, ch in zip(axes, ("rel_l2", "tke", "sps")):
    lo, lv = np.array(DATA[ch][0]), np.array(DATA[ch][1])
    m, c = np.polyfit(lo, lv, 1)
    xs = np.linspace(lo.min() - 0.02 * np.ptp(lo) - 1e-3, lo.max() + 0.02 * np.ptp(lo) + 1e-3, 50)
    txt, col = VERD[ch]
    ax.plot(xs, m * xs + c, "-", color=GREY, lw=2, zorder=2)
    ax.plot(lo, lv, "o", color=col, ms=9, zorder=3, markeredgecolor="white", markeredgewidth=1.2)
    ax.set_title(ch, color=DARK, fontweight="bold", fontsize=13)
    ax.set_xlabel("local (val900)"); ax.grid(color="#E6EAF2", lw=1, zorder=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(axis="x", labelsize=10)
    ax.text(0.5, -0.30, txt, transform=ax.transAxes, ha="center",
            fontsize=11, fontweight="bold", color=col)
axes[0].set_ylabel("live leaderboard")
fig.suptitle("Does the local score predict the live score?  One panel per channel, 7 anchors",
             color=DARK, fontweight="bold", fontsize=13.5, y=1.00)
fig.tight_layout(rect=[0, 0.06, 1, 0.97])
fig.savefig(f"{HERE}/fig3_proxy.png"); plt.close(fig)
print("wrote fig1_climb.png fig2_gap.png fig3_proxy.png")
