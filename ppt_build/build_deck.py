"""Build the SIH 2026 idea deck for PS 26228 on top of the official template.

Design rules followed here:
  * The template's SIH branding — logo, footer bar, slide numbers, team-name
    badge — is never touched. Only the placeholder body text is replaced.
  * The mandated section pointers ("Detailed explanation of the proposed
    solution", "Potential challenges and risks", ...) are kept as the headings
    of each block, because SIH's own instructions say not to change them.
  * Six slides maximum, including the title slide. The instructions slide is
    deleted.
  * Every slide carries a diagram, chart or screenshot. No slide is bullets
    on white.
"""

from __future__ import annotations

import copy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------
# Palette — chosen for the subject: defence-grade assurance, traffic-light
# dispositions. The same colours the product's own report uses, so the deck
# and the software look like one thing.
# --------------------------------------------------------------------------
NAVY = RGBColor(0x13, 0x29, 0x4B)
STEEL = RGBColor(0x1C, 0x72, 0x93)
GREEN = RGBColor(0x1A, 0x7F, 0x4B)
RED = RGBColor(0xC0, 0x39, 0x2B)
AMBER = RGBColor(0xB0, 0x6F, 0x00)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTE = RGBColor(0x5A, 0x64, 0x72)
PANEL = RGBColor(0xEE, 0xF2, 0xF6)
PANEL_G = RGBColor(0xEA, 0xF5, 0xEF)
PANEL_R = RGBColor(0xFD, 0xF0, 0xEE)
PANEL_A = RGBColor(0xFF, 0xF8, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD5, 0xDD, 0xE5)

HEAD_FONT = "Cambria"      # safe-list serif, pairs with the template's titles
BODY_FONT = "Calibri"      # safe-list sans, renders true-to-width in QA

# Usable canvas: below the title band, above the footer bar.
TOP = 1.30
BOTTOM = 6.82
LEFT = 0.36
RIGHT = 12.97


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _fmt(run, size, *, bold=False, color=INK, font=BODY_FONT, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    return tb, tf


def para(tf, text, size, *, bold=False, color=INK, font=BODY_FONT, first=False,
         space_after=4, align=PP_ALIGN.LEFT, italic=False, space_before=0,
         line_spacing=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    if line_spacing:
        p.line_spacing = line_spacing
    r = p.add_run()
    r.text = text
    _fmt(r, size, bold=bold, color=color, font=font, italic=italic)
    return p


def rich(tf, parts, size, *, first=False, space_after=4, space_before=0,
         line_spacing=None, align=PP_ALIGN.LEFT):
    """A paragraph built from (text, bold, colour) tuples — for bold lead-ins."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    if line_spacing:
        p.line_spacing = line_spacing
    for text, bold, color in parts:
        r = p.add_run()
        r.text = text
        _fmt(r, size, bold=bold, color=color)
    return p


def card(slide, x, y, w, h, *, fill=WHITE, line=LINE, line_w=1.0, radius=None,
         shadow=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    if radius is not None:
        shape.adjustments[0] = radius
    if not shadow:
        shape.shadow.inherit = False
    shape.text_frame.text = ""
    return shape


def chip(slide, x, y, w, h, text, *, fill, color=WHITE, size=10.5, bold=True):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background(); s.shadow.inherit = False
    s.adjustments[0] = 0.5
    tf = s.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    _fmt(r, size, bold=bold, color=color)
    return s


def section_head(slide, x, y, w, text, *, color=NAVY, size=13.5, dot=True):
    """A block heading with a small colour dot — the deck's one repeated motif."""
    if dot:
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.055),
                                   Inches(0.115), Inches(0.115))
        d.fill.solid(); d.fill.fore_color.rgb = color
        d.line.fill.background(); d.shadow.inherit = False
        tx = x + 0.21
    else:
        tx = x
    tb, tf = textbox(slide, tx, y, w - (tx - x), 0.30)
    para(tf, text, size, bold=True, color=color, font=HEAD_FONT, first=True,
         space_after=0)
    return tb


def arrow(slide, x, y, w, h, *, color=NAVY, shape=MSO_SHAPE.RIGHT_ARROW):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color
    s.line.fill.background(); s.shadow.inherit = False
    return s


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


# --------------------------------------------------------------------------
# template surgery
# --------------------------------------------------------------------------


def drop(shape):
    shape._element.getparent().remove(shape._element)


def clear_body(slide, keep=("Title", "Slide Number", "Footer", "Oval", "Picture",
                            "Rectangle")):
    """Remove the template's placeholder body text box, keep the furniture."""
    for sh in list(slide.shapes):
        if any(sh.name.startswith(k) for k in keep):
            continue
        drop(sh)


def set_title(slide, text, *, size=30, color=NAVY):
    for sh in slide.shapes:
        if sh.name.startswith("Title"):
            tf = sh.text_frame
            tf.clear()
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = text
            _fmt(r, size, bold=True, color=color, font=HEAD_FONT)
            return sh
    return None


def set_team(slide, name):
    for sh in slide.shapes:
        if sh.name.startswith("Oval"):
            tf = sh.text_frame
            tf.clear()
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = name
            _fmt(r, 10, bold=True, color=NAVY)


def delete_slide(prs, index):
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    prs.part.drop_rel(slides[index].rId)
    xml_slides.remove(slides[index])


# ==========================================================================
# slides
# ==========================================================================

A = Path("assets")


def slide2(slide):
    """IDEA TITLE — proposed solution, how it addresses the problem, uniqueness."""
    set_title(slide, "CVASSURE  —  INTEGRITY YOU CAN PROVE", size=26)

    tb, tf = textbox(slide, LEFT, 1.14, 12.6, 0.30)
    para(tf, "Prove the data, the model and the answer — offline, for computer-vision "
             "pipelines fed by many contributors   ·   PS 26228", 12.5, color=MUTE,
         italic=True, first=True, align=PP_ALIGN.CENTER, space_after=0)

    # ---- left: proposed solution ----------------------------------------
    card(slide, LEFT, 1.52, 6.05, 3.32, fill=PANEL)
    section_head(slide, LEFT + 0.20, 1.64, 5.7, "Proposed Solution", color=NAVY)
    tb, tf = textbox(slide, LEFT + 0.41, 1.92, 5.45, 0.24)
    para(tf, "Three questions, answered before anything goes into service:", 11.2,
         color=MUTE, italic=True, first=True, space_after=0)

    tb, tf = textbox(slide, LEFT + 0.30, 2.24, 5.55, 1.10)
    for i, (q, a) in enumerate([
        ("Was the training data tampered with — and ", "who sent it?"),
        ("Is this the model we were given, or has it been ", "swapped or edited?"),
        ("Is the record of what the model answered ", "trustworthy?"),
    ]):
        rich(tf, [(f"{i+1}.  ", True, STEEL), (q, False, INK), (a, True, NAVY)],
             12.3, first=(i == 0), space_after=6)

    chip(slide, LEFT + 0.30, 3.50, 4.55, 0.32,
         "cvassure audit --dataset data/ --model m.onnx --out results/",
         fill=NAVY, size=9.8)
    tb, tf = textbox(slide, LEFT + 0.30, 3.94, 5.55, 0.80)
    rich(tf, [("One command", True, GREEN),
              ("  →  one self-contained HTML report, a traffic-light verdict, and a "
               "coverage table that also lists what we ", False, INK),
              ("cannot", True, RED), (" detect.", False, INK)],
         12.0, first=True, space_after=0)

    # ---- right: how it addresses the problem -----------------------------
    card(slide, 6.68, 1.52, 6.29, 1.59, fill=WHITE)
    section_head(slide, 6.88, 1.63, 5.9, "How it addresses the problem", color=STEEL)
    tb, tf = textbox(slide, 6.96, 1.94, 5.86, 1.10)
    for i, (lead, rest) in enumerate([
        ("Names the source, not 900 files.",
         " Beta-Binomial risk per contributor — an order an officer can act on."),
        ("Works black-box.",
         " Detects a swapped model from its answers alone; refuses checks above the tier."),
        ("Proof for the log.",
         " Signed, hash-chained receipts — cryptography, not statistics."),
    ]):
        rich(tf, [("\u25b8 ", True, STEEL), (lead, True, NAVY), (rest, False, INK)],
             10.8, first=(i == 0), space_after=3)

    # ---- right: innovation ----------------------------------------------
    card(slide, 6.68, 3.25, 6.29, 1.68, fill=PANEL_A)
    section_head(slide, 6.88, 3.36, 5.9, "Innovation and uniqueness", color=AMBER)
    tb, tf = textbox(slide, 6.96, 3.65, 5.86, 1.16)
    for i, (lead, rest) in enumerate([
        ("Plain English, enforced in code.",
         " A finding with jargon or no number is rejected at construction."),
        ("The detectors cannot see the answer key.",
         " A test reads the source and fails the build if they could."),
        ("It publishes its own weaknesses.",
         " Coverage is generated from measurements, never hand-written."),
    ]):
        rich(tf, [("\u25b8 ", True, AMBER), (lead, True, NAVY), (rest, False, INK)],
             10.6, first=(i == 0), space_after=2)

    # ---- KPI strip -------------------------------------------------------
    kpis = [
        ("1.00", "duplicate floods &\nforeign images caught", GREEN),
        ("0.93", "pasted markers &\nlabel flips caught", GREEN),
        ("100%", "log tampering caught,\nright failure named", NAVY),
        ("0.20", "blended triggers — we\nreport this as a gap", RED),
        ("273", "tests  ·  0 network calls\n225 measured cells", STEEL),
    ]
    w, gap = 2.42, 0.19
    for i, (big, small, colour) in enumerate(kpis):
        x = LEFT + i * (w + gap)
        card(slide, x, 5.00, w, 1.28,
             fill=PANEL_R if colour == RED else WHITE, line=colour, line_w=1.5)
        tb, tf = textbox(slide, x + 0.10, 5.10, w - 0.20, 0.46)
        para(tf, big, 24, bold=True, color=colour, font=HEAD_FONT, first=True,
             align=PP_ALIGN.CENTER, space_after=0)
        tb, tf = textbox(slide, x + 0.08, 5.60, w - 0.16, 0.56)
        for j, ln in enumerate(small.split("\n")):
            para(tf, ln, 9.3, color=MUTE, first=(j == 0), align=PP_ALIGN.CENTER,
                 space_after=0)

    tb, tf = textbox(slide, LEFT, 6.40, 12.6, 0.30)
    para(tf, "Measured over 225 swept cells (5 attacks × 5 poison rates × 3 access "
             "tiers × 3 seeds) plus 12 tampering attempts — graded on the median "
             "cell, never the best.", 10.3, color=MUTE, italic=True, first=True,
         align=PP_ALIGN.CENTER, space_after=0)


def slide3(slide):
    """TECHNICAL APPROACH — stack, tiers, the pipeline, and the eleven checks."""
    set_title(slide, "TECHNICAL APPROACH", size=29)

    # ---- technologies ----------------------------------------------------
    section_head(slide, LEFT, 1.20, 6.0, "Technologies to be used", color=NAVY)
    groups = [
        ("Core", "Python 3.11  ·  NumPy  ·  SciPy  ·  scikit-learn  ·  Pillow", STEEL),
        ("Vision", "PyTorch / TorchScript  ·  ONNX Runtime  ·  bundled ResNet-18", STEEL),
        ("Crypto", "Ed25519 (cryptography)  ·  SHA-256 hash chains  ·  pHash", GREEN),
        ("Formats", "COCO  ·  YOLO  ·  ImageFolder   |   ONNX  ·  TorchScript", NAVY),
        ("Output", "Matplotlib  ·  Jinja2  ·  single-file offline HTML", AMBER),
    ]
    y = 1.54
    for label, items, colour in groups:
        chip(slide, LEFT, y + 0.015, 0.86, 0.235, label, fill=colour, size=9.5)
        tb, tf = textbox(slide, LEFT + 0.98, y, 5.35, 0.28)
        para(tf, items, 10.7, color=INK, first=True, space_after=0)
        y += 0.305

    # ---- access tiers ----------------------------------------------------
    section_head(slide, 7.05, 1.20, 5.9, "Access tiers — declared, and enforced",
                 color=AMBER)
    card(slide, 7.05, 1.52, 5.92, 1.52, fill=PANEL_A)
    rows = [("Tier 0", "answers only", "fingerprint + every data check", GREEN),
            ("Tier 1", "+ weights", "per-layer digest, weight statistics", STEEL),
            ("Tier 2", "+ internals", "spectral signature, trigger reconstruction", NAVY)]
    y = 1.63
    for label, what, gets, colour in rows:
        chip(slide, 7.20, y + 0.01, 0.72, 0.225, label, fill=colour, size=9.5)
        tb, tf = textbox(slide, 8.02, y, 4.82, 0.26)
        rich(tf, [(what, True, colour), ("  \u2192  ", False, MUTE), (gets, False, INK)],
             10.5, first=True, space_after=0)
        y += 0.285
    tb, tf = textbox(slide, 7.20, 2.53, 5.62, 0.44)
    para(tf, "The handle refuses anything above the declared tier — even when the file "
             "format would allow it.", 10.0, color=AMBER, italic=True, first=True,
         space_after=0)

    # ---- methodology / flow ---------------------------------------------
    section_head(slide, LEFT, 3.22, 8.0,
                 "Methodology — one pass over the intake, then one report", color=STEEL)
    flow(slide, LEFT, 3.56)

    # ---- the eleven checks ----------------------------------------------
    section_head(slide, LEFT, 5.72, 8.0,
                 "The eleven checks, and the tier each one needs", color=NAVY)
    checks = [
        ("near_duplicate", GREEN), ("ood", GREEN), ("label_noise", GREEN),
        ("trigger_freq", GREEN), ("contributor", RED), ("shift", GREEN),
        ("fingerprint  t0", GREEN), ("weight_digest  t1", STEEL),
        ("weight_stats  t1", STEEL), ("spectral_signature  t2", NAVY),
        ("trigger_recon  t2", NAVY),
    ]
    x = LEFT
    for label, colour in checks:
        w = 0.10 + 0.073 * len(label)
        chip(slide, x, 6.06, w, 0.30, label, fill=colour, size=9.8)
        x += w + 0.10
    tb, tf = textbox(slide, LEFT, 6.46, 12.6, 0.28)
    para(tf, "Every one degrades to a clean 'we could not check this, and here is why' "
             "at a lower tier — never a crash, never an invented number. A test runs "
             "the whole suite at tier 0 and asserts zero crashes.",
         10.2, color=MUTE, italic=True, first=True, space_after=0)


def flow(slide, ox, oy):
    """The pipeline, as native shapes so it stays crisp and stays editable."""
    box_w, box_h = 2.19, 1.13
    gap = 0.36
    stages = [
        ("1  INGEST", NAVY, WHITE,
         ["COCO · YOLO · folder", "ONNX · TorchScript", "signed receipt log"]),
        ("2  CHECK", STEEL, WHITE,
         ["data · model · shift", "eleven detectors", "graceful at every tier"]),
        ("3  AGGREGATE", RED, PANEL_R,
         ["per-contributor risk", "Beta-Binomial interval", "accept / review / quarantine"]),
        ("4  SCORE", NAVY, WHITE,
         ["AUROC · TPR@1%FPR", "ECE + calibration", "bootstrap intervals"]),
        ("5  REPORT", GREEN, PANEL_G,
         ["traffic-light verdict", "coverage + limitations", "hash-chained audit log"]),
    ]
    for i, (title, colour, fill, lines) in enumerate(stages):
        x = ox + i * (box_w + gap)
        card(slide, x, oy, box_w, box_h, fill=fill, line=colour, line_w=1.75)
        tb, tf = textbox(slide, x + 0.10, oy + 0.09, box_w - 0.20, 0.24)
        para(tf, title, 11.3, bold=True, color=colour, font=HEAD_FONT, first=True,
             align=PP_ALIGN.CENTER, space_after=0)
        tb, tf = textbox(slide, x + 0.08, oy + 0.38, box_w - 0.16, 0.70)
        for j, ln in enumerate(lines):
            para(tf, ln, 9.4, color=INK, first=(j == 0), align=PP_ALIGN.CENTER,
                 space_after=1)
        if i < len(stages) - 1:
            arrow(slide, x + box_w + 0.055, oy + box_h/2 - 0.10, gap - 0.11, 0.20,
                  color=STEEL)

    # the answer-key rail underneath
    y2 = oy + box_h + 0.14
    card(slide, ox, y2, 12.61, 0.72, fill=PANEL, line=MUTE, line_w=1.0)
    tb, tf = textbox(slide, ox + 0.18, y2 + 0.09, 12.25, 0.56)
    rich(tf, [("Validated blind:  ", True, NAVY),
              ("a seeded attack harness poisons the data and keeps the answer key in a "
               "separate folder.  ", False, INK),
              ("detect/ cannot import attack/", True, RED),
              (" — a test reads the source and fails the build if it ever could.",
               False, INK)],
         10.8, first=True, space_after=2)
    para(tf, "No retraining anywhere in the audit (PS 2.2.6)  ·  --offline-assert "
             "replaces the socket layer and fails the run if anything reaches the "
             "network  ·  encoder weights bundled in the repo.",
         10.0, color=MUTE, italic=True, space_after=0)


def slide4(slide):
    """FEASIBILITY AND VIABILITY — with the measured sweep and coverage table."""
    set_title(slide, "FEASIBILITY AND VIABILITY", size=29)

    # ---- left: the chart -------------------------------------------------
    section_head(slide, LEFT, 1.18, 6.6,
                 "Measured, not asserted — 225 cells, 3 seeds each", color=NAVY)
    card(slide, LEFT, 1.50, 6.42, 3.68, fill=WHITE)
    picture(slide, A / "chart_sweep.png", 0.87, 1.60, h=3.02)
    tb, tf = textbox(slide, LEFT + 0.16, 4.76, 6.12, 0.26)
    para(tf, "Four attacks hold above 'strong' down to 1% poison. Blended triggers "
             "do not — and we say so.",
         9.6, color=MUTE, italic=True, first=True, space_after=0)

    # ---- right: feasibility ---------------------------------------------
    section_head(slide, 7.15, 1.18, 5.8, "Analysis of the feasibility", color=GREEN)
    card(slide, 7.15, 1.50, 5.82, 1.40, fill=PANEL_G)
    tb, tf = textbox(slide, 7.32, 1.61, 5.50, 1.20)
    for i, (lead, rest) in enumerate([
        ("Already built and measured.",
         " Not a concept — 273 tests, 225 swept cells, four results tables."),
        ("Commodity hardware.",
         " 347 images audited in 13 s on a laptop CPU. No GPU, no cluster, no cloud."),
        ("Deploys into an air-gap.",
         " Encoder weights ship inside the repository; nothing is downloaded."),
    ]):
        rich(tf, [("\u2713 ", True, GREEN), (lead, True, NAVY), (rest, False, INK)],
             10.8, first=(i == 0), space_after=4)

    # ---- risks + mitigations --------------------------------------------
    section_head(slide, 7.15, 3.02, 5.8,
                 "Potential challenges and risks  \u2192  strategies", color=RED)
    card(slide, 7.15, 3.34, 5.82, 1.74, fill=WHITE)
    tb, tf = textbox(slide, 7.32, 3.45, 5.50, 1.55)
    risks = [
        ("Faint blended triggers evade us (0.20).",
         "Reported as unsupported, not hidden. Tier-2 spectral check is the next build."),
        ("No contributor metadata in the intake.",
         "Regex fallback on paths; otherwise the report says so instead of guessing."),
        ("Vendor supplies a black box.",
         "Fingerprinting still proves substitution at tier 0; coverage marks the rest."),
        ("An attacker who knows our checks.",
         "Stated in every report. The detector set is pluggable and instantly measurable."),
    ]
    for i, (risk, fix) in enumerate(risks):
        rich(tf, [("!  ", True, RED), (risk, True, NAVY)], 10.5,
             first=(i == 0), space_after=0, space_before=0 if i == 0 else 3)
        rich(tf, [("    \u2192 ", True, GREEN), (fix, False, INK)], 10.0, space_after=0)

    # ---- measured coverage table ----------------------------------------
    section_head(slide, LEFT, 5.28, 8.0,
                 "Measured coverage — graded on the median cell, never the best",
                 color=STEEL)
    coverage_table(slide, LEFT, 5.60)


def coverage_table(slide, x, y):
    """The measured coverage, written straight onto the slide."""
    data = [
        ("What an attacker did", "Typical TPR@1%FPR", "Range", "False alarms", "Status"),
        ("The same photograph submitted many times", "1.00", "0.45 – 1.00", "3.0%", "strong"),
        ("Images from a completely different source", "1.00", "0.73 – 1.00", "1.3%", "strong"),
        ("A marker pasted onto the image", "0.93", "0.67 – 1.00", "1.7%", "strong"),
        ("The picture is fine, the label is wrong", "0.93", "0.83 – 1.00", "1.0%", "strong"),
        ("A faint pattern over the whole image", "0.20", "0.00 – 0.67", "1.1%", "UNSUPPORTED"),
        ("Any tampering with the inference log", "1.00", "proof, not an estimate", "0.0%", "strong"),
    ]
    widths = [4.55, 1.95, 2.20, 1.65, 2.26]
    rows, cols = len(data), len(data[0])
    tbl_shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y),
                                       Inches(sum(widths)), Inches(0.165 * rows))
    table = tbl_shape.table
    table.first_row = False
    table.horz_banding = False
    for i, w in enumerate(widths):
        table.columns[i].width = Inches(w)
    for r, row in enumerate(data):
        table.rows[r].height = Inches(0.165)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.04)
            cell.margin_top = Emu(0); cell.margin_bottom = Emu(0)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = NAVY
            elif row[4] == "UNSUPPORTED":
                cell.fill.fore_color.rgb = PANEL_R
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else PANEL
            tf = cell.text_frame
            tf.word_wrap = False
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run(); run.text = value
            if r == 0:
                _fmt(run, 9.6, bold=True, color=WHITE)
            elif c == 4:
                _fmt(run, 9.6, bold=True,
                     color=RED if value == "UNSUPPORTED" else GREEN)
            else:
                _fmt(run, 9.6, bold=(c == 1), color=INK)


def slide5(slide):
    """IMPACT AND BENEFITS — the report, the source verdict, the heatmap."""
    set_title(slide, "IMPACT AND BENEFITS", size=29)

    # ---- the verdict screenshot -----------------------------------------
    section_head(slide, LEFT, 1.18, 8.0, "What an analyst actually receives",
                 color=NAVY)
    picture(slide, A / "shot_verdict.png", LEFT, 1.50, w=6.45)
    tb, tf = textbox(slide, LEFT, 2.52, 6.45, 0.28)
    para(tf, "Real output. One offline HTML file — no dashboard to host, nothing to "
             "install on the analyst's machine.", 10.0, color=MUTE, italic=True,
         first=True, space_after=0)

    # ---- charts ----------------------------------------------------------
    card(slide, LEFT, 2.86, 3.25, 2.78, fill=WHITE)
    picture(slide, A / "chart_contrib.png", 0.49, 2.96, h=2.15)
    tb, tf = textbox(slide, LEFT + 0.08, 5.20, 3.09, 0.26)
    para(tf, "Bars: what we found.  Stars: the truth.", 9.2, color=MUTE,
         align=PP_ALIGN.CENTER, first=True, space_after=0)

    card(slide, 3.72, 2.86, 3.43, 2.78, fill=WHITE)
    picture(slide, A / "chart_heatmap.png", 3.85, 2.96, h=2.15)
    tb, tf = textbox(slide, 3.80, 5.20, 3.27, 0.26)
    para(tf, "What we did, against what was true.", 9.2, color=MUTE,
         align=PP_ALIGN.CENTER, first=True, space_after=0)

    # ---- bottom: before / after -----------------------------------------
    card(slide, LEFT, 5.78, 6.79, 0.98, fill=PANEL, line=MUTE)
    tb, tf = textbox(slide, LEFT + 0.16, 5.88, 6.47, 0.80)
    rich(tf, [("Before:  ", True, RED),
              ("an analyst is handed 900 suspect image IDs and no way to rank them.",
               False, INK)], 10.6, first=True, space_after=3)
    rich(tf, [("After:  ", True, GREEN),
              ("one line — quarantine Contributor C3, whose true poison rate was 51% "
               "and whom we scored 45% with a 36–54% interval.", False, INK)],
         10.6, space_after=0)

    # ---- right: impact ---------------------------------------------------
    section_head(slide, 7.28, 1.18, 5.7, "Potential impact on the target audience",
                 color=STEEL)
    card(slide, 7.28, 1.50, 5.69, 1.72, fill=PANEL)
    tb, tf = textbox(slide, 7.44, 1.61, 5.38, 1.52)
    for i, (lead, rest) in enumerate([
        ("Intake officer",
         " gets an order — 'quarantine Contributor D' — not 900 IDs to triage."),
        ("Model reviewer",
         " can prove a vendor model is the enrolled one, from its answers alone."),
        ("Investigator",
         " gets a named failure mode on the exact record that was edited."),
        ("Commander",
         " sees one traffic light, and an honest list of what was not checked."),
    ]):
        rich(tf, [("\u25b8 ", True, STEEL), (lead, True, NAVY), (rest, False, INK)],
             10.8, first=(i == 0), space_after=4)

    section_head(slide, 7.28, 3.36, 5.7, "Benefits of the solution", color=GREEN)
    card(slide, 7.28, 3.68, 5.69, 2.78, fill=PANEL_G)
    tb, tf = textbox(slide, 7.44, 3.79, 5.38, 2.60)
    for i, (tag, lead, rest) in enumerate([
        ("Operational", "Triage collapses",
         " from every image to a handful of sources — minutes, not days."),
        ("Security", "Poisoned data and swapped models",
         " are caught before they reach a model that guides a decision."),
        ("Legal / audit", "Every inference is provable",
         " after the fact; the audit trail is itself tamper-evident."),
        ("Economic", "Contributions are salvaged, not binned",
         " — quarantine one source instead of discarding a whole intake."),
        ("Ethical", "The system states its own limits",
         " in every report, so nobody over-trusts a clean result."),
    ]):
        rich(tf, [(tag + "  ", True, GREEN), ("·  ", False, MUTE),
                  (lead, True, NAVY), (rest, False, INK)],
             10.8, first=(i == 0), space_after=6)


def slide6(slide):
    """RESEARCH AND REFERENCES — plus where each PS clause is answered."""
    set_title(slide, "RESEARCH AND REFERENCES", size=29)

    section_head(slide, LEFT, 1.16, 6.2, "Attack literature we test against",
                 color=RED)
    card(slide, LEFT, 1.48, 6.20, 1.80, fill=PANEL_R)
    tb, tf = textbox(slide, LEFT + 0.18, 1.59, 5.86, 1.62)
    for i, (name, cite) in enumerate([
        ("BadNets", " \u2014 Gu, Dolan-Gavitt & Garg, 2017. Patch triggers.  arXiv:1708.06733"),
        ("Blended / invisible triggers", " \u2014 Chen et al., 2017. The attack we measure "
                                         "worst on.  arXiv:1712.05526"),
        ("Spectral Signatures", " \u2014 Tran, Li & Madry, NeurIPS 2018. Our tier-2 "
                                "check.  arXiv:1811.00636"),
        ("Neural Cleanse", " \u2014 Wang et al., IEEE S&P 2019. Trigger reconstruction, "
                           "MAD anomaly index."),
        ("Dataset poisoning survey", " \u2014 Goldblum et al., TPAMI 2023. "
                                     "Multi-contributor threat model."),
    ]):
        rich(tf, [(name, True, NAVY), (cite, False, INK)], 10.4,
             first=(i == 0), space_after=3)

    section_head(slide, LEFT, 3.40, 6.2, "Methods and standards", color=STEEL)
    card(slide, LEFT, 3.72, 6.20, 1.42, fill=WHITE)
    tb, tf = textbox(slide, LEFT + 0.18, 3.83, 5.86, 1.24)
    for i, (name, cite) in enumerate([
        ("Ed25519", " \u2014 RFC 8032. Keys generated locally, never transmitted."),
        ("Isotonic calibration", " \u2014 Zadrozny & Elkan, KDD 2002. Score \u2192 probability."),
        ("Benjamini\u2013Hochberg", " \u2014 JRSS-B 1995. Controls false alarms across the "
                                "nine shift measurements."),
        ("COCO / YOLO", " \u2014 Lin et al., ECCV 2014; Ultralytics data.yaml. Both "
                        "ingest formats PS 2.2.6 requires."),
    ]):
        rich(tf, [(name, True, NAVY), (cite, False, INK)], 10.4,
             first=(i == 0), space_after=3)

    section_head(slide, LEFT, 5.26, 6.2, "Datasets", color=AMBER)
    card(slide, LEFT, 5.58, 6.20, 0.72, fill=PANEL_A)
    tb, tf = textbox(slide, LEFT + 0.18, 5.68, 5.86, 0.56)
    para(tf, "SYNTH-10 \u2014 procedural, ships with the repo so every result reproduces "
             "air-gapped  ·  CIFAR-10  ·  GTSRB  ·  VisDrone / DOTA subset, in both "
             "COCO and YOLO form.", 10.4, color=INK, first=True, space_after=0)

    # ---- right: our own results -----------------------------------------
    section_head(slide, 7.05, 1.16, 5.9, "Our own measured results", color=GREEN)
    card(slide, 7.05, 1.48, 5.92, 2.72, fill=WHITE)
    picture(slide, A / "chart_calib.png", 7.18, 1.58, h=2.28)
    tb, tf = textbox(slide, 10.02, 1.66, 2.82, 2.12)
    for i, (lead, rest) in enumerate([
        ("Calibration works.", " Expected Calibration Error falls from 0.277 to "
                               "0.079 once the combined score is calibrated as a "
                               "quantity in its own right."),
        ("Sanity-checked first.", " Random scores must give AUROC 0.5, perfect "
                                 "scores 1.0 \u2014 both are tests, written before any "
                                 "detector existed."),
    ]):
        rich(tf, [(lead, True, NAVY), (rest, False, INK)], 9.8,
             first=(i == 0), space_after=6)
    tb, tf = textbox(slide, 7.20, 3.88, 5.62, 0.28)
    para(tf, "Every headline number carries a bootstrap 95% interval; every swept cell "
             "runs at three seeds.", 9.8, color=MUTE, italic=True, first=True,
         space_after=0)

    section_head(slide, 7.05, 4.34, 5.9, "Repository and reproduction", color=NAVY)
    card(slide, 7.05, 4.66, 5.92, 1.64, fill=PANEL)
    tb, tf = textbox(slide, 7.22, 4.77, 5.58, 1.44)
    for i, (cmd, rest) in enumerate([
        ("make reproduce", "  regenerates every number and figure from scratch, offline."),
        ("make verify", "  runs 273 tests plus the assertion that nothing touched the "
                        "network."),
        ("make demo", "  the four-minute judge sequence, laptop in airplane mode."),
    ]):
        rich(tf, [(cmd, True, NAVY), (rest, False, INK)], 10.6,
             first=(i == 0), space_after=4)
    para(tf, "Full source, the seeded attack harness and all four results tables ship "
             "in the repository.", 9.8, color=MUTE, italic=True, space_after=0)

    # ---- where each problem-statement clause is answered -----------------
    tb, tf = textbox(slide, LEFT, 6.44, 1.55, 0.28)
    para(tf, "PS 26228 covered:", 10.4, bold=True, color=NAVY, font=HEAD_FONT,
         first=True, space_after=0)
    clauses = [
        "2.2.1  data integrity",
        "2.2.2  model integrity",
        "2.2.3  provenance",
        "2.2.4  calibration + shift",
        "2.2.5  analyst report",
        "2.2.6  formats · tiers · offline",
    ]
    x = LEFT + 1.58
    for label in clauses:
        w = 0.20 + 0.066 * len(label)
        chip(slide, x, 6.42, w, 0.30, "\u2713  " + label, fill=GREEN, size=9.5)
        x += w + 0.08


# ==========================================================================


def main() -> None:
    prs = Presentation("template.pptx")
    builders = {2: slide2, 3: slide3, 4: slide4, 5: slide5, 6: slide6}

    for idx, build in builders.items():
        slide = prs.slides[idx - 1]
        clear_body(slide)
        set_team(slide, "Team Name")
        build(slide)

    delete_slide(prs, 6)          # the instructions slide
    prs.save("SIH2026_PS26228_cvassure.pptx")
    print("wrote SIH2026_PS26228_cvassure.pptx")


if __name__ == "__main__":
    main()
