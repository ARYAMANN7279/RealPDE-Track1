"""Slide 17: where we stand. 181 teams, the published top 50, and our position."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DARK, MID, ACC = "#1A2E4A", "#1F4E79", "#2E86C1"
GREEN, ORANGE, GREY = "#1E8B4C", "#CA6F1E", "#8A94A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#C7CEDB", "axes.labelcolor": "#2C3E50",
                     "xtick.color": "#2C3E50", "ytick.color": "#2C3E50",
                     "axes.spines.top": False, "axes.spines.right": False})

d = json.load(open(os.path.join(HERE, "lb.json")))
rows = sorted([(r["owner"], float(r["scores"][0]["score"])) for r in d["submissions"]],
              key=lambda x: -x[1])
TOTAL = int(d["count"])
vals = np.array([v for _, v in rows])
US = 80.081266
CUT10, CUT50, TOP = vals[9], vals[49], vals[0]

fig, ax = plt.subplots(figsize=(12.2, 4.9), dpi=200)
xs = np.arange(1, 51)

# the published field as a gently sloping line of dots
ax.plot(xs, vals, "-", color="#C7CEDB", lw=1.6, zorder=2)
ax.plot(xs, vals, "o", color=MID, ms=7, zorder=3, markeredgecolor="white", markeredgewidth=1.0)

# reference lines
ax.axhline(CUT10, color=ORANGE, lw=1.7, ls="--", zorder=2)
ax.text(27, CUT10 + 0.035, "top-10 cutoff   %.2f" % CUT10, color=ORANGE, fontsize=11.5,
        fontweight="bold", va="bottom", ha="center",
        bbox=dict(fc="white", ec="none", pad=1.6))
ax.axhline(CUT50, color=GREY, lw=1.5, ls=":", zorder=2)
ax.text(27, CUT50 + 0.035, "top-50 cutoff   %.2f" % CUT50, color=GREY, fontsize=11.5,
        fontweight="bold", va="bottom", ha="center",
        bbox=dict(fc="white", ec="none", pad=1.6))

# us
UX = 56.5
ax.plot([UX], [US], "o", color=GREEN, ms=19, zorder=6,
        markeredgecolor="white", markeredgewidth=2.0)
ax.annotate("US   %.3f" % US, (UX, US), textcoords="offset points", xytext=(16, -2),
            ha="left", va="center", fontsize=14, fontweight="bold", color=GREEN)

# the gap that matters, drawn as a short bracket
ax.annotate("", xy=(UX, CUT50), xytext=(UX, US),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.2))
ax.text(UX - 1.4, (US + CUT50) / 2, "%.2f from\nthe top 50" % (CUT50 - US), color=GREEN,
        fontsize=12, fontweight="bold", va="center", ha="right")

# rank-1 label
ax.annotate("%.2f   rank 1" % TOP, (1, TOP), textcoords="offset points", xytext=(10, 6),
            fontsize=11.5, fontweight="bold", color=DARK)

ax.set_xlim(-2, 72); ax.set_ylim(79.88, 82.32)
ax.set_xticks([1, 10, 20, 30, 40, 50])
ax.set_xlabel("rank on the published leaderboard")
ax.set_ylabel("final score")
ax.grid(axis="y", color="#EDF1F7", lw=1, zorder=0)
ax.set_title("%d teams entered.  The entire published top 50 spans %.2f points — and we sit %.2f "
             "below its cutoff." % (TOTAL, TOP - CUT50, CUT50 - US),
             color=DARK, fontweight="bold", fontsize=13.5, pad=14)
ax.text(0, 79.95, "%d further teams are outside the published top 50." % (TOTAL - 50),
        fontsize=11, color=GREY, style="italic")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig4_field.png"))
print("wrote fig4_field.png   total=%d  top1=%.5f cut10=%.5f cut50=%.5f us=%.5f"
      % (TOTAL, TOP, CUT10, CUT50, US))
