"""Refresh the deck for the 24 Sep live result: 80.061320 -> 80.081266.

Only text and two figures change. The new result is an important NEGATIVE: the stack moved sps by
-0.0003 and the +0.02 came entirely from time noise, so slide 14 gains a row that says so.
"""
import os, re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

P = os.path.expanduser("~/Desktop/sem7/UGP/UGP_presentation.pptx")
FIG = os.path.expanduser("~/Desktop/sem7/UGP/figs")
GREEN, ORANGE, MID = RGBColor(0x1E, 0x8B, 0x4C), RGBColor(0xCA, 0x6F, 0x1E), RGBColor(0x1F, 0x4E, 0x79)
DARK_GRAY = RGBColor(0x2C, 0x3E, 0x50)

prs = Presentation(P)
sl = list(prs.slides)

# ── 1. text substitutions across every slide and its notes ───────────────────
SUBS = [("80.061", "80.081"), ("80.0613", "80.0813"), ("80.06132", "80.081266"),
        ("39.558", "39.558"), ("90.563", "90.771")]
nsub = 0
for s in sl:
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                t0 = run.text
                for a, b in SUBS:
                    if a in run.text and a != b:
                        run.text = run.text.replace(a, b)
                if run.text != t0:
                    nsub += 1
    if s.has_notes_slide:
        tf = s.notes_slide.notes_text_frame
        t0 = tf.text
        t = t0
        for a, b in SUBS:
            if a != b:
                t = t.replace(a, b)
        if t != t0:
            tf.text = t
            nsub += 1
print("text substitutions:", nsub)

# ── 2. swap the two regenerated figures ──────────────────────────────────────
def swap(slide_idx, fname):
    s = sl[slide_idx]
    pics = [sh for sh in s.shapes if sh.shape_type == 13]
    if not pics:
        print("  no picture on slide", slide_idx + 1); return
    old = pics[0]
    l, t, w, h = old.left, old.top, old.width, old.height
    old._element.getparent().remove(old._element)
    path = os.path.join(FIG, fname)
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    nw, nh = int(iw * sc), int(ih * sc)
    s.shapes.add_picture(path, l + (w - nw) // 2, t + (h - nh) // 2, width=nw, height=nh)
    print("  slide %d <- %s" % (slide_idx + 1, fname))

titles = []
for s in sl:
    ts = [sh.text_frame.text.split("\n")[0].strip() for sh in s.shapes
          if sh.has_text_frame and sh.text_frame.text.strip()]
    titles.append(ts[0] if ts else "")
i_climb = next(i for i, t in enumerate(titles) if "Steps That Moved" in t)
i_stand = next(i for i, t in enumerate(titles) if "Where We Stand" in t)
swap(i_climb, "fig1_climb.png")
swap(i_stand, "fig4_field.png")

# ── 3. slide "Inside Each Submission": add the STACK row ─────────────────────
i_inside = next(i for i, t in enumerate(titles) if "Inside Each Submission" in t)
s = sl[i_inside]
rows = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip() == "NARROW080"]
if rows:
    y = rows[0].top + Inches(0.44)
    CL = Inches(0.45)
    def cell(txt, x, w, sz, bold, col, align=PP_ALIGN.LEFT):
        tb = s.shapes.add_textbox(CL + Inches(x), y, Inches(w), Inches(0.4))
        tb.text_frame.word_wrap = True
        p = tb.text_frame.paragraphs[0]; p.text = txt; p.alignment = align
        p.font.name = "Arial"; p.font.size = Pt(sz); p.font.bold = bold; p.font.color.rgb = col
    cell("STACK", 0.35, 2.5, 13.5, True, MID)
    cell("re-tune the interval scalars once more", 3.0, 6.4, 12.5, False, DARK_GRAY)
    cell("same (frozen)", 9.5, 2.0, 12, False, DARK_GRAY)
    cell("80.08", 11.6, 0.95, 13.5, True, DARK_GRAY, PP_ALIGN.RIGHT)
    # move the closing caption down and re-point it at the new result
    caps = [sh for sh in s.shapes if sh.has_text_frame and "byte-identical" in sh.text_frame.text]
    if caps:
        c = caps[0]
        c.top = y + Inches(0.5)
        p = c.text_frame.paragraphs[0]
        for r in p.runs:
            r.text = ""
        r = p.runs[0] if p.runs else p.add_run()
        r.text = ("STACK moved the uncertainty channel by −0.0003 — nothing. Its +0.02 was "
                  "favourable time noise. The bounds lever is finished.")
        r.font.name = "Arial"; r.font.size = Pt(13.5); r.font.bold = True; r.font.color.rgb = ORANGE
    print("  added STACK row to slide", i_inside + 1)

prs.save(P)
print("saved", P)
PY = None
