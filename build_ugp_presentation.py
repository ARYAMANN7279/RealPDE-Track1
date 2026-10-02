"""
Builds the UGP lab presentation: UGP_RealPDE_Presentation.pptx

Style follows ae646/stage2/build_stage2_presentation.py exactly (same palette,
same header/footer/card/bullets helpers, same 13.33x7.5 canvas).
Figures come from figs/ (regenerate with figs/make_figs.py).

Run:  python3 build_ugp_presentation.py      (needs python-pptx, Pillow)
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")

DARK_BLUE   = RGBColor(0x1A, 0x2E, 0x4A)
MID_BLUE    = RGBColor(0x1F, 0x4E, 0x79)
ACCENT_BLUE = RGBColor(0x2E, 0x86, 0xC1)
LIGHT_BLUE  = RGBColor(0xD6, 0xE4, 0xF0)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE   = RGBColor(0xF4, 0xF6, 0xF9)
DARK_GRAY   = RGBColor(0x2C, 0x3E, 0x50)
GREEN       = RGBColor(0x1E, 0x8B, 0x4C)
ORANGE      = RGBColor(0xCA, 0x6F, 0x1E)

W, H = Inches(13.33), Inches(7.5)
CL, CR = Inches(0.45), Inches(12.88)
CW = CR - CL
CT = Inches(1.5)
CB = Inches(7.05)
CAH = CB - CT
TOTAL = 20

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def box(sl, l, t, w, h, fill=None, border=None, bw=Pt(1.2)):
    s = sl.shapes.add_shape(1, l, t, w, h)
    s.line.width = bw
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    else:
        s.fill.background()
    if border:
        s.line.color.rgb = border
    else:
        s.line.fill.background()
    return s


def tx(sl, text, l, t, w, h, sz=Pt(14), bold=False, color=DARK_GRAY, align=PP_ALIGN.LEFT):
    tb = sl.shapes.add_textbox(l, t, w, h)
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = "Arial"; p.font.size = sz; p.font.bold = bold; p.font.color.rgb = color
    return tb


def bullets(sl, items, l, t, w, h, sz=Pt(16), color=DARK_GRAY, gap=Pt(8)):
    tb = sl.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + item
        p.font.name = "Arial"; p.font.size = sz; p.font.color.rgb = color
        p.space_after = gap
    return tb


def pic(sl, name, l, t, maxw, maxh):
    path = os.path.join(FIG, name)
    iw, ih = Image.open(path).size
    scale = min(maxw / iw, maxh / ih)
    w, h = int(iw * scale), int(ih * scale)
    sl.shapes.add_picture(path, l + (maxw - w) // 2, t + (maxh - h) // 2, width=w, height=h)


def header(sl, title, sub):
    box(sl, 0, 0, W, H, fill=OFF_WHITE)
    box(sl, 0, 0, W, Inches(1.15), fill=DARK_BLUE)
    box(sl, 0, Inches(1.15), W, Inches(0.08), fill=ACCENT_BLUE)
    tx(sl, title, Inches(0.5), Inches(0.18), W - Inches(1.0), Inches(0.55), sz=Pt(30), bold=True, color=WHITE)
    tx(sl, sub, Inches(0.52), Inches(0.72), W - Inches(1.0), Inches(0.35), sz=Pt(16), color=LIGHT_BLUE)


def footer(sl, num):
    box(sl, 0, H - Inches(0.35), W, Inches(0.35), fill=DARK_BLUE)
    tx(sl, "UGP  |  NeurIPS 2026 RealPDE Competition, Track 1 (Sim2Real)", CL, H - Inches(0.32), CW, Inches(0.3),
       sz=Pt(12), color=LIGHT_BLUE)
    tx(sl, f"{num} / {TOTAL}", W - Inches(1.5), H - Inches(0.32), Inches(1.0), Inches(0.3),
       sz=Pt(12), bold=True, color=WHITE, align=PP_ALIGN.RIGHT)


def card(sl, l, t, w, h, title=None, title_color=DARK_BLUE):
    box(sl, l, t, w, h, fill=WHITE, border=LIGHT_BLUE, bw=Pt(1.0))
    if title:
        tx(sl, title, l + Inches(0.25), t + Inches(0.15), w - Inches(0.5), Inches(0.4),
           sz=Pt(19), bold=True, color=title_color)


def metric(sl, l, t, w, label, value, color=MID_BLUE, h=Inches(1.15)):
    box(sl, l, t, w, h, fill=WHITE, border=LIGHT_BLUE, bw=Pt(1.0))
    tx(sl, value, l, t + Inches(0.14), w, Inches(0.5), sz=Pt(26), bold=True, color=color, align=PP_ALIGN.CENTER)
    tx(sl, label, l, t + Inches(0.66), w, Inches(0.35), sz=Pt(12), color=DARK_GRAY, align=PP_ALIGN.CENTER)




def notes(sl, text):
    """Detail lives here, not on the slide."""
    sl.notes_slide.notes_text_frame.text = text


def kv(sl, l, t, w, pairs, sz=Pt(16), gap=Inches(0.62), lw=Inches(1.5), col=MID_BLUE):
    """Left label / right value rows — replaces a wall of bullets."""
    y = t
    for k, v in pairs:
        tx(sl, k, l, y, lw, Inches(0.4), sz=sz, bold=True, color=col)
        tx(sl, v, l + lw, y + Inches(0.02), w - lw, Inches(0.5), sz=Pt(sz.pt - 1.5))
        y += gap
    return y


def big(sl, l, t, w, h, num, lab, col=ACCENT_BLUE):
    box(sl, l, t, w, h, fill=WHITE, border=LIGHT_BLUE, bw=Pt(1.0))
    tx(sl, num, l, t + Inches(0.22), w, Inches(0.7), sz=Pt(34), bold=True, color=col, align=PP_ALIGN.CENTER)
    tx(sl, lab, l + Inches(0.2), t + Inches(1.0), w - Inches(0.4), Inches(0.9), sz=Pt(14),
       align=PP_ALIGN.CENTER)


# ── 1 Title ───────────────────────────────────────────────────────────────────
sl = slide()
box(sl, 0, 0, W, H, fill=DARK_BLUE)
box(sl, 0, H - Inches(0.8), W, Inches(0.8), fill=MID_BLUE)
tx(sl, "UGP Presentation  ·  ACAL Lab", Inches(1.0), Inches(1.75), W - Inches(2.0), Inches(0.5),
   sz=Pt(20), color=LIGHT_BLUE)
tx(sl, "Forecasting Real Airfoil Wakes", Inches(1.0), Inches(2.3), W - Inches(2.0),
   Inches(0.95), sz=Pt(46), bold=True, color=WHITE)
tx(sl, "Neural Operators on PIV Measurements", Inches(1.0), Inches(3.3), W - Inches(2.0),
   Inches(0.7), sz=Pt(30), color=ACCENT_BLUE)
tx(sl, "Aryamann Srivastava", Inches(1.0), Inches(4.5), W - Inches(2.0), Inches(0.4), sz=Pt(17), color=WHITE)
tx(sl, "NeurIPS 2026 RealPDE Competition  ·  Track 1, Sim-to-Real",
   Inches(1.0), Inches(4.95), W - Inches(2.0), Inches(0.4), sz=Pt(15), color=LIGHT_BLUE)
tx(sl, "80.061 / 100   ·   50 teams   ·   closes 27 September 2026",
   Inches(1.0), H - Inches(0.58), W - Inches(2.0), Inches(0.4), sz=Pt(14), color=LIGHT_BLUE)
notes(sl, "Title slide. One line of context: this is an international competition run alongside "
          "NeurIPS, and the data is real water-tunnel PIV, not simulation.")

# ── 2 Roadmap ─────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "What This Talk Covers", "No machine-learning background assumed")
footer(sl, 2)
items = [("1", "The problem", "A real wake, measured"), ("2", "The method", "What the model does"),
         ("3", "The scoring", "Five channels"), ("4", "What we tried", "Including the failures"),
         ("5", "Where we stand", "Against 50 teams")]
for i, (n, t_, d_) in enumerate(items):
    l = CL + i * (CW / 5)
    big(sl, l + Inches(0.06), CT + Inches(0.6), CW / 5 - Inches(0.12), Inches(2.1), n, t_)
    tx(sl, d_, l + Inches(0.06), CT + Inches(2.8), CW / 5 - Inches(0.12), Inches(0.5), sz=Pt(13),
       color=DARK_GRAY, align=PP_ALIGN.CENTER)
notes(sl, "Roadmap. Flag that the first half is deliberately slow for people without an ML "
          "background, and the second half is results and positioning.")

# ── 3 The data ────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "This Is the Data", "Streamwise velocity, measured by PIV in a water tunnel")
footer(sl, 3)
card(sl, CL, CT, CW, Inches(4.35))
pic(sl, "fig6_data.png", CL + Inches(0.15), CT + Inches(0.12), CW - Inches(0.3), Inches(4.1))
tx(sl, "Given the top row  →  predict the bottom row.",
   CL, CT + Inches(4.6), CW, Inches(0.5), sz=Pt(22), bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
notes(sl, "Walk them through the picture. NACA4418 airfoil, 20 degrees angle of attack, Re 13,950. "
          "Blue is slow fluid, red is fast. The dark blue lobe is the separated wake. Note the "
          "speckle — that is genuine measurement noise from the PIV cross-correlation, not a "
          "rendering artefact. The task is: see the top three frames (and 17 others), produce the "
          "bottom three (and 17 others). Twenty frames in, twenty frames out. Mention that in the "
          "real competition the test runs are at Reynolds numbers we never trained on.")

# ── 4 The task ────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "The Task, Precisely", "One sentence, then the numbers")
footer(sl, 4)
box(sl, CL, CT, CW, Inches(1.1), fill=WHITE, border=ACCENT_BLUE, bw=Pt(2.0))
tx(sl, "Given 20 measured frames,  predict the next 20.", CL, CT + Inches(0.26), CW, Inches(0.6),
   sz=Pt(26), bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
for i, (num, lab) in enumerate([("32 × 64", "grid points per frame"), ("2", "velocity components  u, v"),
                                ("81", "training runs"), ("69,000", "real frames total")]):
    big(sl, CL + i * (CW / 4) + Inches(0.06), CT + Inches(1.45), CW / 4 - Inches(0.12), Inches(1.9), num, lab)
card(sl, CL, CT + Inches(3.55), CW, CAH - Inches(3.55), "Conditions")
kv(sl, CL + Inches(0.35), CT + Inches(4.15), CW - Inches(0.7), [
    ("Training", "Reynolds 3,750 – 26,700   ·   angle of attack 0° – 20°"),
    ("Testing", "unseen conditions, reaching Re 27,975 — ABOVE our training maximum"),
], sz=Pt(17), gap=Inches(0.58), lw=Inches(1.6))
notes(sl, "Keep this short. The one number that matters is the last line: the test set extrapolates "
          "beyond the training range in Reynolds number, so this is not interpolation. Pressure is "
          "in the tensor but identically zero — PIV cannot measure it.")

# ── 5 Why it is hard ──────────────────────────────────────────────────────────
sl = slide()
header(sl, "Why This Is Hard", "Three independent difficulties")
footer(sl, 5)
for i, (t_, d_) in enumerate([
    ("Turbulence", "Small errors amplify.\n20 frames is a long horizon."),
    ("Extrapolation", "Test conditions were\nnever seen in training."),
    ("Irreducible noise", "The targets themselves\nare noisy measurements."),
]):
    l = CL + i * (CW / 3)
    card(sl, l + Inches(0.08), CT, CW / 3 - Inches(0.16), Inches(2.3))
    tx(sl, t_, l + Inches(0.08), CT + Inches(0.42), CW / 3 - Inches(0.16), Inches(0.6),
       sz=Pt(23), bold=True, color=ACCENT_BLUE, align=PP_ALIGN.CENTER)
    tx(sl, d_, l + Inches(0.35), CT + Inches(1.15), CW / 3 - Inches(0.7), Inches(0.9),
       sz=Pt(15), align=PP_ALIGN.CENTER)
card(sl, CL, CT + Inches(2.5), CW, CAH - Inches(2.5))
pic(sl, "fig7_noise.png", CL + Inches(0.15), CT + Inches(2.62), CW - Inches(0.3), Inches(2.8))
notes(sl, "On the third point — the plot shows one row of one frame. The jitter is measurement "
          "noise. No model can predict that, so a floor on achievable error exists. This becomes "
          "important later: it is why chasing accuracy forever does not pay, and why the "
          "uncertainty channel is where the value is.")

# ── 6 Primer ──────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "A Two-Minute Primer", "For anyone without a machine-learning background")
footer(sl, 6)
card(sl, CL, CT, CW, CAH)
kv(sl, CL + Inches(0.35), CT + Inches(0.42), CW - Inches(0.7), [
    ("A model", "a function with millions of adjustable numbers, called weights"),
    ("Training", "nudge every weight until the output matches known answers"),
    ("Neural operator", "maps whole FIELDS to fields, via Fourier modes — a learned spectral method"),
    ("Fine-tuning", "start from weights someone else trained, continue on your own data"),
    ("Held-out test", "never train on what you test on, or you measure memorisation"),
], sz=Pt(18), gap=Inches(0.92), lw=Inches(3.0))
notes(sl, "Spend two minutes here, no more. The one worth dwelling on for this audience is the "
          "neural operator: the FNO transforms the field into Fourier modes, multiplies by learned "
          "weights, transforms back. It is essentially a spectral method where the coefficients are "
          "learned from data rather than derived. That framing usually lands instantly with people "
          "who have written spectral solvers.")

# ── 7 Constraints ─────────────────────────────────────────────────────────────
sl = slide()
header(sl, "The Rules", "Four constraints that shaped every decision")
footer(sl, 7)
for i, (t_, d_) in enumerate([("256 MB", "total package size.\nThe given model is 201 MB."),
                              ("5 minutes", "hard runtime limit —\nand speed is scored."),
                              ("1 per day", "submissions. One real\nmeasurement per 24 h."),
                              ("Top 10 only", "are shortlisted for the\nfinal private re-scoring.")]):
    l = CL + i * (CW / 4)
    big(sl, l + Inches(0.06), CT + Inches(0.3), CW / 4 - Inches(0.12), Inches(2.4), t_, d_)
box(sl, CL, CT + Inches(3.1), CW, Inches(2.1), fill=WHITE, border=ORANGE, bw=Pt(2.0))
tx(sl, "One measurement per day makes guessing unaffordable.",
   CL, CT + Inches(3.35), CW, Inches(0.5), sz=Pt(22), bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
tx(sl, "Every submission had to be justified by a written prediction, and every prediction checked "
       "afterwards.  The project became a measurement programme — 12,000 lines of decision record, "
       "186 numbered sections.",
   CL + Inches(0.8), CT + Inches(3.95), CW - Inches(1.6), Inches(1.1), sz=Pt(15), align=PP_ALIGN.CENTER)
notes(sl, "This slide explains the whole working style. With 24 hours between measurements you "
          "cannot iterate by trial and error — you must predict, submit, then reconcile. Most of "
          "what follows is the consequence of that constraint.")

# ── 8 Scoring ─────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "How a Submission Is Scored", "Five channels — the combining weights were never published")
footer(sl, 8)
card(sl, CL, CT, Inches(7.4), CAH)
rows = [("rel_l2", "overall velocity error", "46.7%", MID_BLUE),
        ("tke", "turbulent kinetic energy", "10.0%", MID_BLUE),
        ("mvpe", "error at 9 wake probes", "9.4%", MID_BLUE),
        ("time", "how fast the code runs", "9.7%", MID_BLUE),
        ("sps", "quality of the UNCERTAINTY", "24.7%", ORANGE)]
y = CT + Inches(0.45)
for n, d_, w_, c in rows:
    tx(sl, n, CL + Inches(0.35), y, Inches(1.5), Inches(0.45), sz=Pt(19), bold=True, color=c)
    tx(sl, d_, CL + Inches(2.0), y + Inches(0.05), Inches(3.7), Inches(0.45), sz=Pt(15))
    tx(sl, w_, CL + Inches(5.7), y, Inches(1.4), Inches(0.45), sz=Pt(19), bold=True, color=c,
       align=PP_ALIGN.RIGHT)
    y += Inches(0.82)
box(sl, CL + Inches(7.6), CT, CW - Inches(7.6), CAH, fill=WHITE, border=ORANGE, bw=Pt(2.0))
tx(sl, "We recovered the\nhidden weights", CL + Inches(7.85), CT + Inches(0.35), CW - Inches(8.1),
   Inches(1.0), sz=Pt(23), bold=True, color=ORANGE)
tx(sl, "Least squares on 40 leaderboard rows.\nMaximum residual: 0.005.",
   CL + Inches(7.85), CT + Inches(1.5), CW - Inches(8.1), Inches(0.9), sz=Pt(16))
tx(sl, "This turned \"improve the model\" into a budgeting problem —\nand revealed that uncertainty "
       "is worth a quarter of the score.",
   CL + Inches(7.85), CT + Inches(2.7), CW - Inches(8.1), Inches(1.6), sz=Pt(15), color=DARK_BLUE)
notes(sl, "The organisers publish the five channel scores but not how they combine into the final "
          "number. We solved for the weights by least squares across 40 leaderboard rows and got a "
          "fit accurate to 0.005. That is what let us decide where to spend effort — and it showed "
          "the uncertainty channel, which we had been ignoring, was worth 24.7%.")

# ── 9 sps ─────────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "The Unusual Channel: Uncertainty", "We may return a lower and upper bound for every value")
footer(sl, 9)
card(sl, CL, CT, CW, Inches(4.0))
pic(sl, "fig5_sps.png", CL + Inches(0.15), CT + Inches(0.12), CW - Inches(0.3), Inches(3.75))
tx(sl, "Narrow and correct is the only thing that pays.  Being cautious is punished almost as hard "
       "as being wrong.",
   CL, CT + Inches(4.3), CW, Inches(0.5), sz=Pt(19), bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
notes(sl, "Work through the four rows. Key consequence for later: because the reward decays "
          "exponentially with width, this channel is set by the TYPICAL error, whereas the accuracy "
          "channel is an L2 norm dominated by the WORST errors. They are nearly independent "
          "objectives — a model can be no more accurate overall and still win this channel outright.")

# ── 10 Methodology ────────────────────────────────────────────────────────────
sl = slide()
header(sl, "How We Avoided Fooling Ourselves", "The most important thing the project produced")
footer(sl, 10)
card(sl, CL, CT, CW / 2 - Inches(0.15), CAH, "The failure", title_color=ORANGE)
bullets(sl, ["Offline gains kept vanishing on the leaderboard.",
             "Cause: we were testing on conditions we had trained on.",
             "One change predicted +0.22 and scored −0.04.",
             "We were measuring memorisation, not skill."],
        CL + Inches(0.3), CT + Inches(0.85), CW / 2 - Inches(0.75), Inches(4.2), sz=Pt(17), gap=Pt(16))
card(sl, CL + CW / 2 + Inches(0.15), CT, CW / 2 - Inches(0.15), CAH, "The fix", title_color=GREEN)
bullets(sl, ["Hold out whole Reynolds numbers, never time windows.",
             "Decide the pass mark before seeing the result.",
             "Measure how much of an offline gain actually survives.",
             "Prove byte-for-byte what a change cannot affect."],
        CL + CW / 2 + Inches(0.45), CT + Inches(0.85), CW / 2 - Inches(0.75), Inches(4.2),
        sz=Pt(17), gap=Pt(16))
notes(sl, "Tell the story: early on several changes looked strong offline and then did nothing live. "
          "The cause was that our validation split was not condition-disjoint — windows from the "
          "same run appeared on both sides. Once fixed, most of our apparent gains evaporated, which "
          "was painful but necessary. Everything after this point is measured honestly.")

# ── 11 Approach 1 ─────────────────────────────────────────────────────────────
sl = slide()
header(sl, "Approach 1 of 3:  The Model", "Improving the forecast itself")
footer(sl, 11)
card(sl, CL, CT, CW, CAH)
y = kv(sl, CL + Inches(0.4), CT + Inches(0.5), CW - Inches(0.8), [
    ("Fine-tuning", "continue training the given simulation model on our 81 real runs"),
    ("Model soup", "train several times, then average the weights"),
    ("Ensembling", "run several models, average the answers, correct a constant offset"),
    ("TKE head", "a second network that rescales the predicted fluctuations"),
], sz=Pt(19), gap=Inches(0.86), lw=Inches(2.9))
box(sl, CL + Inches(0.4), y + Inches(0.25), CW - Inches(0.8), Inches(1.15), fill=OFF_WHITE,
    border=ORANGE, bw=Pt(1.6))
tx(sl, "But: six weeks of this made the model only 3.7% better on unseen Reynolds numbers.  "
       "Slide 18 explains why.",
   CL + Inches(0.7), y + Inches(0.55), CW - Inches(1.4), Inches(0.7), sz=Pt(17), bold=True,
   color=ORANGE, align=PP_ALIGN.CENTER)
notes(sl, "Fine-tuning was the single biggest step — it took us from roughly 77 to 78.45. Model "
          "soup means training several times from different random starts and averaging the weights; "
          "it cancels each run's idiosyncrasies. The TKE head was worth +2.21 on that channel alone. "
          "But flag the caveat now and pay it off on slide 18 — this is the thread of the talk.")

# ── 12 Approach 2 ─────────────────────────────────────────────────────────────
sl = slide()
header(sl, "Approach 2 of 3:  Uncertainty", "24.7% of the score, and free of the accuracy channels")
footer(sl, 12)
card(sl, CL, CT, CW, Inches(2.5), "How our bounds are produced")
for i, (n, d_) in enumerate([("1", "a network predicts how wrong\nthe forecast will be, per point"),
                             ("2", "that is sorted into 24 bins;\na table gives each a width"),
                             ("3", "a third network re-centres\nthe interval where it covers best")]):
    l = CL + Inches(0.4) + i * ((CW - Inches(0.8)) / 3)
    tx(sl, n, l, CT + Inches(0.75), Inches(0.5), Inches(0.6), sz=Pt(30), bold=True, color=ACCENT_BLUE)
    tx(sl, d_, l + Inches(0.6), CT + Inches(0.8), (CW - Inches(0.8)) / 3 - Inches(0.8), Inches(1.2), sz=Pt(15))
box(sl, CL, CT + Inches(2.7), CW, Inches(1.1), fill=WHITE, border=GREEN, bw=Pt(2.0))
tx(sl, "Only the bounds move — so the forecast stays byte-identical and the accuracy channels "
       "provably cannot change.",
   CL + Inches(0.4), CT + Inches(3.0), CW - Inches(0.8), Inches(0.6), sz=Pt(18), bold=True,
   color=GREEN, align=PP_ALIGN.CENTER)
card(sl, CL, CT + Inches(3.95), CW, CAH - Inches(3.95), "Why this became the focus")
tx(sl, "Uncertainty explained ~68% of our gap to first place.  One rival matched our accuracy to "
       "three decimals and still scored 0.58 higher — the channels are independent.",
   CL + Inches(0.4), CT + Inches(4.5), CW - Inches(0.8), Inches(0.9), sz=Pt(16))
notes(sl, "The green box is the important one for a sceptical audience: because the prediction array "
          "is untouched, these submissions carry zero accuracy risk. We verified that byte-for-byte "
          "before shipping each one. Every gain since 21 September came from this channel with the "
          "model completely frozen.")

# ── 13 Approach 3 ─────────────────────────────────────────────────────────────
sl = slide()
header(sl, "Approach 3 of 3:  Speed", "Runtime is 9.7% of the score")
footer(sl, 13)
card(sl, CL, CT, CW / 2 - Inches(0.15), CAH, "What we did")
bullets(sl, ["Moved per-element work onto the GPU.",
             "Fused three networks into one — a third fewer GPU calls.",
             "Half-precision weights, unpacked on the GPU.",
             "Built the fixed coordinate grid once, not every call."],
        CL + Inches(0.3), CT + Inches(0.85), CW / 2 - Inches(0.75), Inches(3.0), sz=Pt(17), gap=Pt(16))
box(sl, CL + Inches(0.3), CT + Inches(4.0), CW / 2 - Inches(0.75), Inches(1.1), fill=OFF_WHITE,
    border=LIGHT_BLUE, bw=Pt(1.2))
tx(sl, "7.8 s against a 300 s limit.", CL + Inches(0.3), CT + Inches(4.3), CW / 2 - Inches(0.75),
   Inches(0.5), sz=Pt(20), bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
card(sl, CL + CW / 2 + Inches(0.15), CT, CW / 2 - Inches(0.15), CAH, "The lesson", title_color=GREEN)
kv(sl, CL + CW / 2 + Inches(0.45), CT + Inches(0.95), CW / 2 - Inches(0.75), [
    ("100–116%", "of host-side savings survived to the leaderboard"),
    ("~5%", "of GPU kernel tuning survived"),
], sz=Pt(20), gap=Inches(1.2), lw=Inches(1.9), col=GREEN)
tx(sl, "Where you remove work matters far more than how much.",
   CL + CW / 2 + Inches(0.45), CT + Inches(3.35), CW / 2 - Inches(0.75), Inches(0.8),
   sz=Pt(17), bold=True, color=DARK_BLUE)
tx(sl, "One idea was rejected on measurement: cutting the FNO's internal padding gave a real 1.74× "
       "speed-up and destroyed accuracy — a 60× worse trade.",
   CL + CW / 2 + Inches(0.45), CT + Inches(4.15), CW / 2 - Inches(0.75), Inches(1.1), sz=Pt(14))
notes(sl, "The asymmetry between host-side and GPU-side savings is worth stating clearly — it is a "
          "transferable engineering lesson. The padding example is a good one for this audience: it "
          "was a genuine speed-up that would have cost us far more in accuracy than it gained in "
          "time, and we only knew because we priced it before submitting.")

# ── 14 The climb ──────────────────────────────────────────────────────────────
sl = slide()
header(sl, "The Steps That Moved the Score", "Only submissions that improved on the best so far")
footer(sl, 14)
card(sl, CL, CT, CW, Inches(4.5))
pic(sl, "fig1_climb.png", CL + Inches(0.15), CT + Inches(0.15), CW - Inches(0.3), Inches(4.2))
tx(sl, "The last two steps changed only the bounds — the forecast was byte-identical.",
   CL, CT + Inches(4.75), CW, Inches(0.5), sz=Pt(18), bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
notes(sl, "Point out three places: SHIFT_v2 is where the uncertainty channel was added — the largest "
          "single jump. FA1BMT is the TKE head. The final two points are pure bounds calibration "
          "with the model frozen. Also note the flat stretch in the middle: that is six weeks of "
          "accuracy work producing almost nothing, which is the puzzle slide 18 resolves.")

# ── 15 Worked example ─────────────────────────────────────────────────────────
sl = slide()
header(sl, "Getting It Wrong, Then Right", "22 – 23 September")
footer(sl, 15)
card(sl, CL, CT, CW / 2 - Inches(0.15), CAH, "22 Sep — the mistake", title_color=ORANGE)
kv(sl, CL + Inches(0.35), CT + Inches(0.95), CW / 2 - Inches(0.8), [
    ("Predicted", "+0.11  (widen the intervals 20%)"),
    ("Actual", "−0.48  — worst submission of the project"),
    ("Cause", "a calibration constant was 2.14× too large"),
], sz=Pt(18), gap=Inches(1.05), lw=Inches(1.9), col=ORANGE)
card(sl, CL + CW / 2 + Inches(0.15), CT, CW / 2 - Inches(0.15), CAH, "23 Sep — the recovery",
     title_color=GREEN)
kv(sl, CL + CW / 2 + Inches(0.45), CT + Inches(0.95), CW / 2 - Inches(0.8), [
    ("Predicted", "+0.31  (narrow them instead)"),
    ("Actual", "+0.20  — crossing 80 for the first time"),
    ("Now", "the curve is pinned; this lever is finished"),
], sz=Pt(18), gap=Inches(1.05), lw=Inches(1.9), col=GREEN)
box(sl, CL, CT + Inches(4.3), CW, Inches(1.2), fill=WHITE, border=ACCENT_BLUE, bw=Pt(2.0))
tx(sl, "A wrong prediction that was written down in advance becomes a measurement.",
   CL, CT + Inches(4.6), CW, Inches(0.6), sz=Pt(20), bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)
notes(sl, "This is the slide that front-runs the obvious question, so do not skip it. The failed "
          "submission is what let us solve for the calibration constant — one equation, one unknown. "
          "The answer was 1.00, not 2.14, meaning the intervals should have been narrowed all along. "
          "Root cause: an earlier successful change had altered interval shape and width together and "
          "we credited the gain to width. The next day's submission, built on the corrected constant, "
          "crossed 80.")

# ── 16 Positioning ────────────────────────────────────────────────────────────
sl = slide()
header(sl, "Where We Stand", "50 teams published;  competition closes 27 September")
footer(sl, 16)
card(sl, CL, CT, CW, Inches(4.3))
pic(sl, "fig4_field.png", CL + Inches(0.15), CT + Inches(0.12), CW - Inches(0.3), Inches(4.05))
tx(sl, "The whole field spans less than two points on a 100-point scale.  We are 2.5% behind the leader.",
   CL, CT + Inches(4.55), CW, Inches(0.5), sz=Pt(18), bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
notes(sl, "Be straightforward here. We are at 80.06, rank 1 is 82.10, the top-10 cutoff is 81.74. "
          "In relative terms that is a 2.5% difference, but the field is so tightly packed that 2.5% "
          "spans rank 1 to outside the top 50. The left panel is there to make the scale honest — "
          "every team is inside one dot on the full range.")

# ── 17 Gap ────────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "What the Gap Consists Of", "Reverse-engineered from the rank-1 competitor's published scores")
footer(sl, 17)
card(sl, CL, CT, Inches(7.2), CAH)
pic(sl, "fig2_gap.png", CL + Inches(0.1), CT + Inches(0.25), Inches(7.0), CAH - Inches(0.5))
card(sl, CL + Inches(7.4), CT, CW - Inches(7.4), CAH, "How we know")
bullets(sl, ["Their errors are 10–16% smaller — and they are 30% FASTER.",
             "Faster AND more accurate cannot come from tuning.",
             "Against a perfect-uncertainty ORACLE: we are at 83.6%, they are at 90.7%.",
             "So: ~58% better model, ~42% better uncertainty."],
        CL + Inches(7.65), CT + Inches(0.75), CW - Inches(7.9), Inches(4.3), sz=Pt(16), gap=Pt(18))
notes(sl, "Explain the oracle: for a given set of forecast errors there is a best possible set of "
          "intervals — knowing each error exactly and sizing the interval to just cover it. That is "
          "an upper bound no method can beat. We reach 83.6% of it; the leader reaches 90.7%. This "
          "is how we separated 'they have a better model' from 'they have better uncertainty' using "
          "only published numbers.")

# ── 18 Main finding ───────────────────────────────────────────────────────────
sl = slide()
header(sl, "Main Finding", "Our offline test cannot measure the thing we were tuning")
footer(sl, 18)
card(sl, CL, CT, CW, Inches(3.5))
pic(sl, "fig3_proxy.png", CL + Inches(0.15), CT + Inches(0.15), CW - Inches(0.3), Inches(3.2))
box(sl, CL, CT + Inches(3.7), CW, Inches(1.8), fill=WHITE, border=ORANGE, bw=Pt(2.0))
tx(sl, "For accuracy the slope is −3.2 — meaningless.  The offline score varies 3× less than the "
       "real one.", CL + Inches(0.4), CT + Inches(3.95), CW - Inches(0.8), Inches(0.5),
   sz=Pt(19), bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
tx(sl, "This is why six weeks of fine-tuning produced 3.7%:  we were selecting models with an "
       "instrument that cannot measure them.  The other two channels ARE calibrated.",
   CL + Inches(0.4), CT + Inches(4.55), CW - Inches(0.8), Inches(0.9), sz=Pt(16), align=PP_ALIGN.CENTER)
notes(sl, "Each point is one past submission: its offline score against the score it actually got. "
          "For a useful test these should lie on a rising line. For rel_l2 they do not — the fit is "
          "negative and meaningless, because our offline score spans 0.028 while the live score spans "
          "0.098. The test has no resolution in that channel. The uncertainty panel, by contrast, is "
          "clean: slope 0.62, and that matches a factor we derived independently from a single live "
          "anchor. This is the result I would most like feedback on.")

# ── 19 Closed ─────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "What We Tested and Closed", "Negative results, each with a number")
footer(sl, 19)
card(sl, CL, CT, CW, CAH)
tests = [("Extra inputs for the uncertainty model", "+0.0005"),
         ("Sensitivity of the forecast to its input", "0.000"),
         ("Averaging three uncertainty estimates", "+0.0004"),
         ("A new uncertainty net trained on the true score", "−0.0004"),
         ("U-Net — the benchmark's best architecture", "−16%"),
         ("MWT — small enough to ensemble many copies", "−60%"),
         ("Joint simulation + real training, 3 variants", "0")]
y = CT + Inches(0.45)
for name, val in tests:
    tx(sl, name, CL + Inches(0.4), y, Inches(8.5), Inches(0.45), sz=Pt(17), color=DARK_GRAY)
    tx(sl, val, CL + Inches(9.2), y, Inches(3.0), Inches(0.45), sz=Pt(17), bold=True,
       color=ORANGE, align=PP_ALIGN.RIGHT)
    y += Inches(0.62)
tx(sl, "Each one closes a direction with a measurement rather than an opinion.",
   CL + Inches(0.4), y + Inches(0.25), CW - Inches(0.8), Inches(0.5), sz=Pt(17), bold=True,
   color=DARK_BLUE, align=PP_ALIGN.CENTER)
notes(sl, "Do not read every row. Pick two: the U-Net result, because it is the benchmark's own "
          "best-ranked architecture and it is 16% worse on our data — published rankings do not "
          "transfer. And the last row: adding 81,000 simulation frames to training did not beat the "
          "starting checkpoint in any of three configurations, which is what told us the model is "
          "converged.")

# ── 20 Current state + next ───────────────────────────────────────────────────
sl = slide()
header(sl, "Where We Are, and What Is Next", "23 September 2026")
footer(sl, 20)
for i, (lab, val, col) in enumerate([("our score", "80.061", GREEN), ("rank 1", "82.101", DARK_GRAY),
                                     ("top-10 cutoff", "81.742", ORANGE), ("submissions left", "3", MID_BLUE)]):
    metric(sl, CL + i * (CW / 4) + Inches(0.06), CT, CW / 4 - Inches(0.12), lab, val, color=col)
card(sl, CL, CT + Inches(1.4), CW / 2 - Inches(0.15), CAH - Inches(1.4), "The ceiling, measured",
     title_color=ORANGE)
tx(sl, "Even a mathematically PERFECT uncertainty model scores 81.727 — still below the 81.742 cutoff.",
   CL + Inches(0.35), CT + Inches(2.1), CW / 2 - Inches(0.85), Inches(1.1), sz=Pt(17), bold=True,
   color=DARK_BLUE)
tx(sl, "So a top-10 place provably requires a better underlying model, not better bounds — and the "
       "model is converged.", CL + Inches(0.35), CT + Inches(3.3), CW / 2 - Inches(0.85), Inches(1.2),
   sz=Pt(15))
card(sl, CL + CW / 2 + Inches(0.15), CT + Inches(1.4), CW / 2 - Inches(0.15), CAH - Inches(1.4),
     "Next", title_color=GREEN)
bullets(sl, ["Ship the final calibrated submission (~80.09).",
             "Pretrain our OWN backbone on the simulation data — the identified route.",
             "Rebuild the uncertainty stack on it.",
             "Write up the calibration finding."],
        CL + CW / 2 + Inches(0.45), CT + Inches(2.05), CW / 2 - Inches(0.8), Inches(3.0),
        sz=Pt(16), gap=Pt(14))
notes(sl, "Close on the honest position. The 80 target is met. Top 10 is not reachable in the days "
          "left, and we can prove why rather than guess — that is the perfect-uncertainty number. "
          "The identified next step is pretraining our own backbone on the released simulation data, "
          "because the evidence says that is where the leaders' advantage comes from: outside "
          "pretrained models are banned, so there is nowhere else it can be. That is a multi-day "
          "training job, which is exactly why it did not fit inside the competition window.")

out = os.path.join(HERE, "UGP_RealPDE_Presentation.pptx")
prs.save(out)
print("wrote", out, os.path.getsize(out), "bytes")
