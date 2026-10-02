"""Patch the EDITED deck in place: add 3 new slides, rewrite the Speed slide, renumber footers.

Works on ~/Downloads/UGP_RealPDE_Presentation.pptx so Aryamann's own edits are preserved.
New slides are appended, then moved into position by reordering the slide-ID list.
"""
import copy, os, re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

SRC = os.path.expanduser("~/Downloads/UGP_RealPDE_Presentation.pptx")
DST = os.path.expanduser("~/Desktop/sem7/UGP/UGP_RealPDE_Presentation_v2.pptx")
FIG = os.path.expanduser("~/Desktop/sem7/UGP/figs")

DARK_BLUE, MID_BLUE, ACCENT_BLUE = RGBColor(0x1A, 0x2E, 0x4A), RGBColor(0x1F, 0x4E, 0x79), RGBColor(0x2E, 0x86, 0xC1)
LIGHT_BLUE, WHITE, OFF_WHITE = RGBColor(0xD6, 0xE4, 0xF0), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xF4, 0xF6, 0xF9)
DARK_GRAY, GREEN, ORANGE = RGBColor(0x2C, 0x3E, 0x50), RGBColor(0x1E, 0x8B, 0x4C), RGBColor(0xCA, 0x6F, 0x1E)

prs = Presentation(SRC)
W, H = prs.slide_width, prs.slide_height
CL = Inches(0.45); CW = Inches(12.88) - CL
CT = Inches(1.5); CB = Inches(7.05); CAH = CB - CT
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def box(sl, l, t, w, h, fill=None, border=None, bw=Pt(1.2)):
    s = sl.shapes.add_shape(1, l, t, w, h); s.line.width = bw
    if fill: s.fill.solid(); s.fill.fore_color.rgb = fill
    else: s.fill.background()
    if border: s.line.color.rgb = border
    else: s.line.fill.background()
    return s


def tx(sl, text, l, t, w, h, sz=Pt(14), bold=False, color=DARK_GRAY, align=PP_ALIGN.LEFT):
    tb = sl.shapes.add_textbox(l, t, w, h); tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]; p.text = text; p.alignment = align
    p.font.name = "Arial"; p.font.size = sz; p.font.bold = bold; p.font.color.rgb = color
    return tb


def bullets(sl, items, l, t, w, h, sz=Pt(16), color=DARK_GRAY, gap=Pt(8)):
    tb = sl.shapes.add_textbox(l, t, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + it
        p.font.name = "Arial"; p.font.size = sz; p.font.color.rgb = color; p.space_after = gap
    return tb


def pic(sl, name, l, t, maxw, maxh):
    path = os.path.join(FIG, name); iw, ih = Image.open(path).size
    s = min(maxw / iw, maxh / ih); w, h = int(iw * s), int(ih * s)
    sl.shapes.add_picture(path, l + (maxw - w) // 2, t + (maxh - h) // 2, width=w, height=h)


def header(sl, title, sub):
    box(sl, 0, 0, W, H, fill=OFF_WHITE)
    box(sl, 0, 0, W, Inches(1.15), fill=DARK_BLUE)
    box(sl, 0, Inches(1.15), W, Inches(0.08), fill=ACCENT_BLUE)
    tx(sl, title, Inches(0.5), Inches(0.18), W - Inches(1.0), Inches(0.55), sz=Pt(30), bold=True, color=WHITE)
    tx(sl, sub, Inches(0.52), Inches(0.72), W - Inches(1.0), Inches(0.35), sz=Pt(16), color=LIGHT_BLUE)


def footer(sl, num):
    box(sl, 0, H - Inches(0.35), W, Inches(0.35), fill=DARK_BLUE)
    tx(sl, "UGP  |  NeurIPS 2026 RealPDE Competition, Track 1 (Sim2Real)", CL, H - Inches(0.32),
       CW, Inches(0.3), sz=Pt(12), color=LIGHT_BLUE)
    tx(sl, "%d / 21" % num, W - Inches(1.5), H - Inches(0.32), Inches(1.0), Inches(0.3),
       sz=Pt(12), bold=True, color=WHITE, align=PP_ALIGN.RIGHT)


def card(sl, l, t, w, h, title=None, title_color=DARK_BLUE):
    box(sl, l, t, w, h, fill=WHITE, border=LIGHT_BLUE, bw=Pt(1.0))
    if title:
        tx(sl, title, l + Inches(0.25), t + Inches(0.15), w - Inches(0.5), Inches(0.4),
           sz=Pt(19), bold=True, color=title_color)


def notes(sl, t):
    sl.notes_slide.notes_text_frame.text = t


new = {}

# ══ A. What we actually submit ════════════════════════════════════════════════
sl = slide(); new["deliverable"] = sl
header(sl, "What We Actually Submit", "Not a paper or a model file — a working program")
card(sl, CL, CT, CW, Inches(4.4))
pic(sl, "fig9_zip.png", CL + Inches(0.15), CT + Inches(0.12), CW - Inches(0.3), Inches(4.15))
tx(sl, "A 250 MB zip. The graders import it, hand it 20 frames, and read back three arrays.",
   CL, CT + Inches(4.65), CW, Inches(0.5), sz=Pt(19), bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)
notes(sl, "Make the deliverable concrete before talking about methods. We submit a zip containing "
          "model weights plus a Python file exposing one function, predict(). The graders run it "
          "inside an isolated container with no internet, PyTorch and NumPy only, and a 5-minute "
          "budget. It returns three arrays: the forecast, and a lower and upper bound for every "
          "single value. The size bar shows why everything is tight: the forecasting model alone "
          "eats 201 of the 256 MB, which is why we could never simply ensemble several models.")

# ══ B. The tooling we built ═══════════════════════════════════════════════════
sl = slide(); new["tooling"] = sl
header(sl, "The Tooling We Built", "You cannot optimise what you cannot see")
card(sl, CL, CT, CW / 2 - Inches(0.15), CAH, "The problem")
bullets(sl, ["The leaderboard shows one number per team.",
             "That number hides WHICH of the five channels you are losing on.",
             "With one submission per day, spending them to find out is unaffordable."],
        CL + Inches(0.3), CT + Inches(0.85), CW / 2 - Inches(0.75), Inches(2.4), sz=Pt(17), gap=Pt(16))
tx(sl, "What we built", CL + Inches(0.3), CT + Inches(3.3), CW / 2 - Inches(0.75), Inches(0.4),
   sz=Pt(19), bold=True, color=DARK_BLUE)
bullets(sl, ["A small tool that reads Codabench's public API directly.",
             "Pulls every team's per-channel scores, not just the total.",
             "Refreshes the standings on demand during analysis."],
        CL + Inches(0.3), CT + Inches(3.8), CW / 2 - Inches(0.75), Inches(1.5), sz=Pt(16), gap=Pt(12))
card(sl, CL + CW / 2 + Inches(0.15), CT, CW / 2 - Inches(0.15), CAH, "What it bought us",
     title_color=GREEN)
bullets(sl, ["Showed uncertainty was ~68% of our gap — we had been optimising the wrong channel.",
             "Found a rival matching our accuracy to 3 decimals but scoring 0.58 higher.",
             "Let us reverse-engineer the leader's model-vs-uncertainty split (slide 18)."],
        CL + CW / 2 + Inches(0.45), CT + Inches(0.85), CW / 2 - Inches(0.75), Inches(2.6),
        sz=Pt(16), gap=Pt(16))
box(sl, CL + CW / 2 + Inches(0.45), CT + Inches(3.55), CW / 2 - Inches(0.75), Inches(1.6),
    fill=OFF_WHITE, border=ORANGE, bw=Pt(1.6))
tx(sl, "Is this allowed?", CL + CW / 2 + Inches(0.7), CT + Inches(3.75), CW / 2 - Inches(1.2),
   Inches(0.4), sz=Pt(17), bold=True, color=ORANGE)
tx(sl, "Yes. It reads the same public endpoint the website itself uses — no login, no private data. "
       "The rules restrict network calls DURING evaluation; this is offline analysis on our own machine.",
   CL + CW / 2 + Inches(0.7), CT + Inches(4.18), CW / 2 - Inches(1.2), Inches(0.9), sz=Pt(13.5))
notes(sl, "Be upfront about the legality question because someone will ask. Codabench serves the "
          "leaderboard to the browser through a public REST endpoint, and at the time it exposed "
          "per-channel subscores for every submission. We read that endpoint programmatically "
          "instead of clicking through the website. No authentication, no private data, nothing "
          "hidden. The competition rules forbid network calls from inside the submitted code during "
          "evaluation — that is a completely different thing, and our submission makes none. "
          "Worth noting the endpoint has since been restricted, so only totals are public now.")

# ══ C. Inside each submission ═════════════════════════════════════════════════
sl = slide(); new["inside"] = sl
header(sl, "Inside Each Submission", "What actually changed, and what it was worth")
card(sl, CL, CT, CW, CAH)
hdr_y = CT + Inches(0.32)
for lab, x, w in (("submission", 0.35, 2.5), ("what changed", 3.0, 6.4), ("model underneath", 9.5, 2.0),
                  ("score", 11.6, 0.95)):
    tx(sl, lab, CL + Inches(x), hdr_y, Inches(w), Inches(0.35), sz=Pt(13), bold=True,
       color=ACCENT_BLUE, align=(PP_ALIGN.RIGHT if lab == "score" else PP_ALIGN.LEFT))
rows = [("SOUP_v1", "fine-tune on real data, then average several runs' weights", "FNO soup", "78.45"),
        ("SHIFT_v2", "ADDED UNCERTAINTY: U-Net + 24-bin width table + off-centre intervals", "+ W96 U-Net", "79.25"),
        ("TMEAN", "correct a constant offset in the time-mean", "same", "79.37"),
        ("SV2", "retrain the backbone on more of the real data", "soup_v2", "79.46"),
        ("SPEED_SAFE", "host-side speed fixes; outputs bit-identical", "same", "79.49"),
        ("FITA1B", "a better fine-tuned backbone", "FITA1B FNO", "79.49"),
        ("FA1BMT", "ADDED THE TKE HEAD: rescale predicted fluctuations", "+ TKE net", "79.59"),
        ("FA1BMLOT7", "move the blank-region mask onto the GPU", "same", "79.75"),
        ("LUT150", "reshape the per-bin interval widths", "same (frozen)", "79.86"),
        ("NARROW080", "multiply every interval width by 0.80", "same (frozen)", "80.06")]
y = hdr_y + Inches(0.42)
for nm, what, mdl, sc in rows:
    em = nm in ("SHIFT_v2", "FA1BMT", "NARROW080")
    tx(sl, nm, CL + Inches(0.35), y, Inches(2.5), Inches(0.4), sz=Pt(13.5), bold=True,
       color=(ORANGE if em else MID_BLUE))
    tx(sl, what, CL + Inches(3.0), y, Inches(6.4), Inches(0.4), sz=Pt(12.5),
       color=(DARK_BLUE if em else DARK_GRAY), bold=em)
    tx(sl, mdl, CL + Inches(9.5), y, Inches(2.0), Inches(0.4), sz=Pt(12), color=DARK_GRAY)
    tx(sl, sc, CL + Inches(11.6), y, Inches(0.95), Inches(0.4), sz=Pt(13.5), bold=True,
       color=(GREEN if em else DARK_GRAY), align=PP_ALIGN.RIGHT)
    y += Inches(0.44)
tx(sl, "The last two changed only the interval widths — the forecast was byte-identical.",
   CL + Inches(0.35), y + Inches(0.12), CW - Inches(0.7), Inches(0.4), sz=Pt(14), bold=True,
   color=MID_BLUE)
notes(sl, "This is the 'specific steps' table. Three rows carry most of the story, highlighted in "
          "orange. SHIFT_v2 is where the uncertainty machinery was added at all — the largest single "
          "jump in the project. FA1BMT added the TKE head. NARROW080 is the final one, and it is the "
          "cleanest illustration of the whole approach: the forecasting model is completely frozen "
          "and byte-identical, we only multiplied every interval width by 0.80, and that was worth "
          "+0.20. Note the 'model underneath' column: after FITA1B the backbone stops changing "
          "entirely. Everything after that is uncertainty and engineering.")

# ══ D. Speed, rewritten ═══════════════════════════════════════════════════════
sl = slide(); new["speed"] = sl
header(sl, "Approach 3 of 3:  Speed", "Why runtime is scored at all, and how much it was worth")
card(sl, CL, CT, CW, Inches(3.5))
pic(sl, "fig8_time.png", CL + Inches(0.15), CT + Inches(0.15), CW - Inches(0.3), Inches(3.2))
card(sl, CL, CT + Inches(3.65), CW / 2 - Inches(0.15), CAH - Inches(3.65), "What we did")
bullets(sl, ["Moved per-point work off the CPU onto the GPU.",
             "Merged three small networks into one.",
             "Stored weights at half precision."],
        CL + Inches(0.3), CT + Inches(4.2), CW / 2 - Inches(0.75), Inches(1.3), sz=Pt(15), gap=Pt(8))
card(sl, CL + CW / 2 + Inches(0.15), CT + Inches(3.65), CW / 2 - Inches(0.15), CAH - Inches(3.65),
     "Why we stopped", title_color=ORANGE)
tx(sl, "We are already on the flat part of the curve. Going from 7.8 s to 4 s would gain only "
       "+0.25 of a point — and the one change big enough to matter destroyed accuracy.",
   CL + CW / 2 + Inches(0.45), CT + Inches(4.2), CW / 2 - Inches(0.75), Inches(1.3), sz=Pt(15))
notes(sl, "Start with the curve, because the formula is unintuitive. The score is "
          "100/(1 + sqrt(t/728.96)). It is deliberately gentle: taking 2.5 minutes still scores 65, "
          "and the hard failure is only at 5 minutes. We run in 7.8 seconds, which scores 90.6 — we "
          "are far out on the flat part, so further speed-ups buy almost nothing. That is the point "
          "of the slide: we did the cheap engineering (GPU, merged networks, half precision), "
          "confirmed we were in the flat region, and stopped. One tempting idea — cutting the "
          "model's internal padding — gave a genuine 1.74x speed-up and wrecked accuracy by 60 times "
          "more than the time gain was worth. We priced it before submitting rather than after.")

# ══ reorder: build the target sequence ════════════════════════════════════════
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
idx = {id(s._element): i for i, s in enumerate(prs.slides)}


def pos(sl):
    return idx[id(sl._element)]


old_speed = pos(prs.slides[10])            # slide 11 in their deck = Approach 3 of 3: Speed
order = ([0, 1, 2, 3] + [pos(new["deliverable"])] + [4, 5, 6, 7] + [pos(new["tooling"])]
         + [8, 9] + [pos(new["speed"])] + [pos(new["inside"])] + [11, 12, 13, 14, 15, 16, 17])
assert old_speed not in order, "the old Speed slide must be dropped"
assert len(set(order)) == len(order), "duplicate slide in the new order"
new_ids = [ids[i] for i in order]
for e in ids:
    sldIdLst.remove(e)
for e in new_ids:
    sldIdLst.append(e)

# ══ renumber every footer "N / M" ═════════════════════════════════════════════
TOT = len(list(sldIdLst))
pat = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
for i, sl in enumerate(prs.slides, 1):
    found = False
    for sh in sl.shapes:
        if sh.has_text_frame and pat.match(sh.text_frame.text or ""):
            found = True
            p = sh.text_frame.paragraphs[0]
            for r in p.runs:
                r.text = ""
            (p.runs[0] if p.runs else p.add_run()).text = "%d / %d" % (i, TOT)
    if not found and i > 1:          # the new slides were built without a footer bar
        footer(sl, i)
        for sh in sl.shapes:
            if sh.has_text_frame and pat.match(sh.text_frame.text or ""):
                p = sh.text_frame.paragraphs[0]
                for r in p.runs:
                    r.text = ""
                (p.runs[0] if p.runs else p.add_run()).text = "%d / %d" % (i, TOT)
prs.save(DST)
print("wrote", DST)
print("%d slides" % TOT)
for i, sl in enumerate(prs.slides, 1):
    ts = [s.text_frame.text.split("\n")[0].strip() for s in sl.shapes
          if s.has_text_frame and s.text_frame.text.strip() and not pat.match(s.text_frame.text)]
    print("%2d  %s" % (i, ts[0][:58] if ts else "(figure)"))
