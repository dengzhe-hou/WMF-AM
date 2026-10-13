"""Shared style for the redesigned figures (draft, 2026-10-06).

Typography follows the global print rules (Arial, 7 pt text, 8.5 pt bold panel labels, vector output).
Colors follow Okabe-Ito with a lighter written-steps orange, fixed per condition across all figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "results"
BUDGET = HERE.parent / "results/budget"
OUT = HERE / "figures"
MM = 1 / 25.4
TEXT_WIDTH = 160 * MM      # ACL two-column text width
COLUMN_WIDTH = 77 * MM     # ACL column width

COLORS = {
    "A": "#0072B2",        # core, 16 tokens, one-integer rule
    "C": "#E69F00",        # single-step arithmetic
    "D": "#009E73",        # cancellation
    "H": "#CC79A7",        # other user and system instructions
    "I": "#E9A36A",        # written steps
    "A512": "#56B4E9",     # core prompt, 512 tokens
    "grey": "#8C8C8C",
    "light": "#EDEDED",
}

# Short labels from NAMING.md (2026-10-09). Figures show these names. The letters remain only as codes
# in the released files and in the Code column of the appendix bank table (tab:bank).
LABELS = {
    "A": "Core", "B": "Depth", "C": "Arithmetic", "D": "Cancellation", "E": "Assignment", "F": "Sets/counts",
    "G": "Wordings", "H": "Instructions", "A512": "Large budget", "I": "Written steps",
    "K": "Intermediate states", "L": "Multi-target",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": .6, "lines.linewidth": 1.2, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "state-probe-figures-20261006",
    })


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")


def save(fig, stem: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        if ext == "pdf":
            kwargs["metadata"] = {"Creator": stem + ".py", "CreationDate": None, "ModDate": None}
        if ext == "svg":
            kwargs["metadata"] = {"Date": None, "Creator": stem + ".py"}
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)
