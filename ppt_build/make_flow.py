"""Pipeline flow diagram for the SIH deck.

Drawn rather than screenshotted so it matches the deck palette exactly and
stays legible when a judge is six feet from a projector.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY, STEEL, RED, GREEN, AMBER = "#13294B", "#1C7293", "#C0392B", "#1A7F4B", "#B06F00"
INK, MUTE, PANEL = "#1A1A1A", "#5C5C5C", "#EEF1F5"

fig, ax = plt.subplots(figsize=(13.0, 8.4))
ax.set_xlim(0, 130); ax.set_ylim(0, 84); ax.axis("off")

def card(x, y, w, h, title, lines, edge=STEEL, fill="#FFFFFF", tsize=13.5, lsize=11.2,
         tcolor=None, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w/2, y + h - 3.4, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold", color=tcolor or edge, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + 2.6, y + h - 8.6 - i*4.5, ln, ha="left", va="top", fontsize=lsize,
                color=INK, zorder=3)

def arrow(x1, y1, x2, y2, color=NAVY, lw=2.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=19, linewidth=lw, color=color, zorder=1))

def band(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                linewidth=0, facecolor=color, alpha=0.10, zorder=0))
    ax.text(x + 1.8, y + h - 1.0, label, ha="left", va="top", fontsize=11,
            fontweight="bold", color=color, zorder=1)

# ---- INPUTS -------------------------------------------------------------
band(0.5, 47.5, 27, 33.5, "1  INTAKE", NAVY)
card(2.0, 66.0, 24, 12.5, "Contributed data", ["COCO  ·  YOLO  ·  ImageFolder",
     "+ contributor metadata"], edge=NAVY)
card(2.0, 57.0, 24, 7.5, "Vendor model", ["ONNX  ·  TorchScript"], edge=NAVY)
card(2.0, 48.5, 24, 7.0, "Inference log", ["signed receipts (JSONL)"], edge=NAVY)

# ---- ACCESS TIER GATE ---------------------------------------------------
card(31.5, 55.0, 20, 23.5, "ACCESS TIER", [
    "0  answers only", "1  + weights", "2  + internals",
    "", "Refuses anything", "above the declared", "tier — by design"],
    edge=AMBER, fill="#FFF8EC", tsize=12.5, lsize=10.6)

# ---- DETECTORS ----------------------------------------------------------
band(55.0, 40.0, 43, 41, "2  ELEVEN CHECKS", STEEL)
card(56.5, 63.5, 20, 15.0, "DATA", ["near-duplicate", "out-of-distribution",
     "label noise", "trigger / patch"], edge=STEEL, tsize=12, lsize=10.4)
card(77.5, 63.5, 19, 15.0, "MODEL", ["fingerprint  [t0]", "weight digest [t1]",
     "weight stats  [t1]", "trigger recon [t2]"], edge=STEEL, tsize=12, lsize=10.4)
card(56.5, 51.5, 20, 10.0, "PROVENANCE", ["Ed25519 + hash chain",
     "6 named failures"], edge=GREEN, tsize=12, lsize=10.4)
card(77.5, 51.5, 19, 10.0, "SHIFT", ["drift vs tampering,",
     "attributed to causes"], edge=GREEN, tsize=12, lsize=10.4)
card(56.5, 41.0, 40, 8.5, "CONTRIBUTOR AGGREGATION",
     ["Beta-Binomial risk per source  →  who do we blame?"],
     edge=RED, fill="#FDF0EE", tsize=12, lsize=10.8)

# ---- SCORING ------------------------------------------------------------
band(101.0, 40.0, 28, 41, "3  MEASURE", NAVY)
card(102.5, 60.0, 25, 18.5, "SCORING HARNESS", [
    "AUROC · TPR@1%FPR", "ECE · bootstrap CIs", "isotonic calibration",
    "fit / calibrate / test"], edge=NAVY, tsize=12, lsize=10.4)
card(102.5, 41.0, 25, 17.0, "ANSWER KEY", [
    "seeded attack harness", "writes ground truth",
    "", "detect/ cannot import", "attack/ — enforced"],
    edge=MUTE, fill=PANEL, tsize=12, lsize=10.2, tcolor=INK)

# ---- OUTPUT -------------------------------------------------------------
card(2.0, 20.0, 60, 16.0, "ONE SELF-CONTAINED HTML REPORT", [
    "Traffic-light verdict  ·  contributor risk with credible intervals",
    "model verdict + access-tier caveats  ·  provenance result",
    "coverage table, including what we CANNOT detect"],
    edge=GREEN, fill="#EAF5EF", tsize=14, lsize=11.4, lw=2.6)
card(66.0, 20.0, 28, 16.0, "TAMPER-EVIDENT", [
    "audit log of the audit:", "every detector run",
    "hash-chained, so the", "report cannot be edited"],
    edge=GREEN, tsize=12.5, lsize=10.8)
card(98.0, 20.0, 30, 16.0, "AIR-GAPPED", [
    "--offline-assert blocks", "the socket layer and",
    "fails the run if anything", "reaches the network"],
    edge=AMBER, fill="#FFF8EC", tsize=12.5, lsize=10.8)

# ---- arrows -------------------------------------------------------------
arrow(26.5, 71.0, 32.0, 69.0)
arrow(26.5, 60.0, 32.0, 65.0)
arrow(26.5, 51.5, 32.0, 60.0)
arrow(52.2, 67.0, 56.8, 70.0)
arrow(52.2, 63.0, 56.8, 56.0)
arrow(97.2, 70.0, 102.8, 70.0)
arrow(97.2, 56.0, 102.8, 62.0)
arrow(76.0, 40.5, 60.0, 37.0)      # aggregation -> report
arrow(114.0, 40.5, 100.0, 37.0)    # measure -> offline card

ax.text(65, 10.0, "No retraining anywhere in the audit  ·  no network access  ·  "
        "every number measured against a seeded answer key",
        ha="center", va="center", fontsize=12.5, style="italic", color=MUTE)

fig.savefig("assets/flow_pipeline.png", dpi=210, bbox_inches="tight",
            facecolor="white", pad_inches=0.12)
print("wrote assets/flow_pipeline.png")
