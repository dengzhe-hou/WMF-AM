#!/usr/bin/env python3
"""Plot verified depth summaries; no model calls, fitting, or new statistics."""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator

from figstyle import COLORS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / "results"
OUT = HERE / "figures"
MM = 1 / 25.4


def apply_project_style():
    """Reuse the existing project's rcParams without executing its output setup.

    Extracting the literal style avoids importing code that creates paper/figures.
    Typography is overridden only to satisfy the current global print rules.
    """
    source = HERE / "style/gen_protocol_safe_figures.py"
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if ast.unparse(call.func) == "plt.rcParams.update":
                plt.rcParams.update(ast.literal_eval(call.args[0]))
                break
    else:
        raise ValueError("Existing project rcParams style was not found")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": .75, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "wmf-am-panel18-20261004",
    })


def save_figure(fig, stem):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        if ext == "pdf":
            kwargs["metadata"] = {"Creator": stem + ".py", "CreationDate": None, "ModDate": None}
        if ext == "svg":
            kwargs["metadata"] = {"Date": None, "Creator": stem + ".py"}
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def main():
    apply_project_style()
    with (SOURCE / "stability_20261004/depth_pooled.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10 and all(r["n_models"] == "18" and r["n_unique_items"] == "20" and r["n_scored"] == "360" for r in rows)
    ks = [int(r["k"]) for r in rows]
    fig, ax = plt.subplots(figsize=(180 * MM, 76 * MM))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.79)
    for field, label, color, marker, style in [
        ("accuracy", "Correct", COLORS["A"], "o", "-"),
        ("compliance_rate", "Exactly one integer", "#6E6E6E", "s", "--"),
        ("truncation_rate", "Truncated", "#999999", "^", ":"),
    ]:
        ax.plot(ks, [100 * float(r[field]) for r in rows], label=label,
                color=color, marker=marker, markersize=3.2, linestyle=style)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlabel="Updates, K (log scale)", ylabel="Responses (%)", ylim=(-2, 102), xlim=(2.75, 110))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", linewidth=.4, color="#D7D7D7", zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.05), ncol=3, frameon=False, handlelength=2.2)
    fig.text(.10, .035, "18 models × 20 shared items per K. Items are regenerated across K. Truncation can overlap correctness.", fontsize=7)
    save_figure(fig, "depth")


if __name__ == "__main__":
    main()
