"""Figures for the HiPC SRS paper: system pipeline, and score progression."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
W = 3.4
NAVY, BLUE, PALE, RUST = "#23405F", "#5E86B5", "#CBDAEA", "#B0452B"

# ---------------- Fig 1: system pipeline with real contours -------------------
z = np.load("figure_data.npz")
inp, fc, half = z["inp"], z["fc"], z["half"]
vmin, vmax = np.percentile(inp, [2, 98])

fig = plt.figure(figsize=(W, 1.88))
gs = fig.add_gridspec(2, 5, width_ratios=[1.35, 0.55, 0.95, 0.55, 1.35],
                      height_ratios=[1, 1], wspace=0.06, hspace=0.42,
                      left=0.01, right=0.99, top=0.88, bottom=0.10)

def panel(gpos, field, title, cmap="RdBu_r", vlim=None):
    a = fig.add_subplot(gs[gpos])
    vv = vlim if vlim else (vmin, vmax)
    a.imshow(field, cmap=cmap, vmin=vv[0], vmax=vv[1], origin="lower", aspect="auto")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.7)
    a.set_title(title, fontsize=6.4, color=NAVY, pad=2.0)
    return a

def blockbox(gpos, label, fc_):
    a = fig.add_subplot(gs[gpos]); a.axis("off")
    a.add_patch(FancyBboxPatch((0.04, 0.24), 0.92, 0.52, transform=a.transAxes,
                               boxstyle="round,pad=0.02,rounding_size=0.08",
                               facecolor=fc_, edgecolor="none", clip_on=False))
    a.text(0.5, 0.50, label, transform=a.transAxes, ha="center", va="center",
           color="white", fontsize=6.8)
    return a

def conn(a, b, y=0.5):
    fig.canvas.draw()
    xa = a.get_position().x1; xb = b.get_position().x0
    ya = a.get_position().y0 + y * a.get_position().height
    fig.add_artist(FancyArrowPatch((xa + 0.004, ya), (xb - 0.004, ya),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=6, lw=0.75, color=NAVY))

a_in  = panel((0, 0), inp[-1], "input window, $u$")
a_fno = blockbox((0, 2), "FNO", NAVY)
a_fc  = panel((0, 4), fc[-1], "forecast $\\hat y$")
a_un  = blockbox((1, 0), "U-Net", BLUE)
hm = half[-1]
a_h   = panel((1, 4), hm, "half width $h$", cmap="viridis",
              vlim=(float(hm.min()), float(np.percentile(hm, 99))))
a_lut = blockbox((1, 2), "24-bin LUT", BLUE)

conn(a_in, a_fno); conn(a_fno, a_fc); conn(a_un, a_lut); conn(a_lut, a_h)
fig.canvas.draw()
p_in, p_un, p_fc, p_lut = [x.get_position() for x in (a_in, a_un, a_fc, a_lut)]
# input and forecast both feed the U-Net
fig.add_artist(FancyArrowPatch((p_in.x0 + 0.10, p_in.y0 - 0.005),
                               (p_un.x0 + 0.10, p_un.y1 + 0.02),
                               transform=fig.transFigure, arrowstyle="-|>",
                               mutation_scale=6, lw=0.75, color=NAVY))
fig.add_artist(FancyArrowPatch((p_fc.x0 + 0.10, p_fc.y0 - 0.005),
                               (p_un.x1 - 0.02, p_un.y1 + 0.02),
                               transform=fig.transFigure, arrowstyle="-|>",
                               mutation_scale=6, lw=0.75, color=NAVY,
                               connectionstyle="arc3,rad=-0.18"))
fig.savefig("figure1_pipeline.pdf"); plt.close(fig)

# ---------------- Fig 2: score progression -----------------------------------
steps = [("SOUP", 78.45), ("SHIFT", 79.25), ("TMEAN", 79.37), ("SV2", 79.46),
         ("SPEED", 79.49), ("FIT", 79.49), ("TKE", 79.59), ("MASK", 79.75),
         ("LUT", 79.86), ("NARROW", 80.06), ("STACK", 80.08)]
v = [s[1] for s in steps]; x = np.arange(len(v))
fig, ax = plt.subplots(figsize=(W, 1.85))
ax.plot(x, v, "-", color=BLUE, lw=1.3, zorder=2)
ax.plot(x, v, "o", color=NAVY, ms=3.4, zorder=3)
# highlight the interval-driven steps
for i in (1, 8, 9):
    ax.plot(x[i], v[i], "o", color=RUST, ms=5.0, zorder=4)
ax.annotate("interval head added, $+0.80$", (x[1], v[1]), textcoords="offset points",
            xytext=(8, -1), fontsize=6.5, color=RUST, va="top")
ax.annotate("width calibration, $+0.20$", (x[9], v[9]), textcoords="offset points",
            xytext=(-2, 9), ha="right", fontsize=6.5, color=RUST)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], rotation=55, ha="right", fontsize=6.2)
ax.set_ylabel("final score")
ax.set_ylim(78.2, 80.35)
ax.grid(axis="y", color="#E9EEF5", lw=0.6)
fig.tight_layout(pad=0.12)
fig.savefig("figure_progress.pdf"); plt.close(fig)
print("wrote figure1_pipeline.pdf figure_progress.pdf")
