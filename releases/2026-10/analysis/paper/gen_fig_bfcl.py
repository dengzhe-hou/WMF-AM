#!/usr/bin/env python3
"""Draft Figure 3: Kendall tau-b of the core, arithmetic, and cancellation items (codes A, C, D) with the three
BFCL criteria, with stored 95% intervals."""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt

from figstyle import COLORS, COLUMN_WIDTH, LABELS, MM, SOURCE, apply_style, save

CRITERIA = [("overall", "Overall"), ("multi_turn", "Multi-Turn"), ("agentic", "Agentic")]
BLOCKS = [(b, LABELS[b]) for b in "ACD"]


def main():
    apply_style()
    rows = {(r["criterion"], r["block"]): r for r in csv.DictReader((SOURCE / "bfcl_comparison_20261004/associations.csv").open())}
    assert len(rows) == 9
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 50 * MM))
    fig.subplots_adjust(left=.2, right=.97, bottom=.2, top=.86)
    y, ticks, labels = 0, [], []
    for ci, (crit, cname) in enumerate(CRITERIA):
        for b, bname in BLOCKS:
            r = rows[crit, b]
            t, lo, hi = float(r["tau_b"]), float(r["ci95_low"]), float(r["ci95_high"])
            ax.plot([lo, hi], [y, y], color=COLORS[b], linewidth=1.1, solid_capstyle="butt")
            ax.scatter([t], [y], s=12, color=COLORS[b], zorder=3, linewidths=0)
            y -= 1
        ticks.append(y + 2); labels.append(cname)
        y -= .8
    ax.axvline(0, color="#7A7A7A", linewidth=.6)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-.5, 1); ax.set_xlabel("Kendall $\\tau_b$ with BFCL (95% interval)", fontsize=7.2)  # subscript 0.7 x 7.2 pt stays above 5 pt
    for b, bname in BLOCKS:
        ax.scatter([], [], s=12, color=COLORS[b], label=bname)
    ax.legend(loc="lower center", bbox_to_anchor=(.42, 1.0), ncol=3, frameon=False, fontsize=6, handletextpad=.1, columnspacing=.7)
    save(fig, "fig_bfcl")


if __name__ == "__main__":
    main()
