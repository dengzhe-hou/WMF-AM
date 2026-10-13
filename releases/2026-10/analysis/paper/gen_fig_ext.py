#!/usr/bin/env python3
"""Section 4.2 figure: the intermediate-state queries (a) and the multi-target items with their readout control (b).

Reads the stored model-bootstrap summary only (mean over 18 models, pointwise 2.5th and 97.5th percentiles of
10,000 resamples over models). Nothing is recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib.pyplot as plt

from figstyle import COLORS, COLUMN_WIDTH, MM, apply_style

ANALYSIS = HERE.parent / "results/ext/analysis.json"
FIG_HEIGHT = 47.0                     # mm
QUERY_POSITIONS = [("q1", "1st"), ("q2", "2nd"), ("q3", "3rd")]
TARGET_COUNTS = [1, 2, 4]
OFFSET = .13                          # horizontal offset of the two conditions in panel b


def percent(entry: dict) -> tuple[float, float, float]:
    """Mean and stored 95% interval, in percent."""
    low, high = entry["ci95"]
    return 100 * entry["mean_over_models"], 100 * low, 100 * high


def points(ax, xs, rows, color, filled: bool, label=None):
    means = [row[0] for row in rows]
    errors = [[row[0] - row[1] for row in rows], [row[2] - row[0] for row in rows]]
    return ax.errorbar(xs, means, yerr=errors, fmt="o", ms=3.4, color=color, ecolor=color,
                       markerfacecolor=color if filled else "white", markeredgecolor=color,
                       markeredgewidth=.8, elinewidth=.8, capsize=1.6, capthick=.8, label=label, zorder=3)


def build():
    apply_style()
    data = json.loads(ANALYSIS.read_text())
    assert data["meta"]["n_models"] == 18 and data["meta"]["missing_trials"] == 0
    assert data["meta"]["bootstrap"]["n"] == 10000 and data["meta"]["bootstrap"]["unit"] == "model"
    by_position = data["K"]["accuracy_by_query_position"]
    accuracy = data["L"]["accuracy"]
    queries = [percent(by_position[key]) for key, _ in QUERY_POSITIONS]
    tracking = [percent(accuracy[f"M{m}_tracking"]) for m in TARGET_COUNTS]
    readout = [percent(accuracy[f"M{m}_readout"]) for m in TARGET_COUNTS]

    fig = plt.figure(figsize=(COLUMN_WIDTH, FIG_HEIGHT * MM))
    width_mm = COLUMN_WIDTH / MM
    bottom, top = 9.6, FIG_HEIGHT - 2.6                   # mm
    positions = [(11.5, 25.0), (49.4, 26.3)]               # (left, width) in mm
    axes = [fig.add_axes([left / width_mm, bottom / FIG_HEIGHT, width / width_mm, (top - bottom) / FIG_HEIGHT])
            for left, width in positions]

    # a. Accuracy by the queried update. Each prompt names a slot ("immediately after slot 01"), and that slot holds
    #    the queried entity's first, second, or third update.
    ax = axes[0]
    points(ax, range(3), queries, COLORS["A"], filled=True)
    ax.set_xticks(range(3))
    ax.set_xticklabels([label for _, label in QUERY_POSITIONS])
    ax.set(xlim=(-.5, 2.5), xlabel="State after target update")

    # b. Tracking against the readout control, which redirects the same updates to other entities.
    ax = axes[1]
    xs = range(len(TARGET_COUNTS))
    points(ax, [x - OFFSET for x in xs], tracking, COLORS["A"], filled=True, label="Tracking")
    points(ax, [x + OFFSET for x in xs], readout, COLORS["grey"], filled=False, label="Readout control")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(m) for m in TARGET_COUNTS])
    ax.set(xlim=(-.5, 2.5), xlabel="Queried entities, M")
    ax.legend(loc="upper left", bbox_to_anchor=(-.01, 1.03), frameon=False, fontsize=6.0, handlelength=1.0,
              handletextpad=.35, labelspacing=.35, borderaxespad=0)

    for ax in axes:
        ax.set(ylim=(0, 100), ylabel="Correct (%)")
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", length=2.0)

    # Panel letters sit at the left edge of each panel's y-axis labels, as in Figure 3.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for letter, axis in zip("ab", axes):
        labels_left = axis.yaxis.get_tightbbox(renderer).x0
        x = fig.transFigure.inverted().transform((labels_left, 0))[0] - .64 / width_mm
        fig.text(x, 1 - 1.2 / FIG_HEIGHT, letter, fontsize=8.5, weight="bold", va="top")
    return fig, axes, {"queries": queries, "tracking": tracking, "readout": readout}


def main() -> None:
    fig, _, _ = build()
    stem = HERE / "figures" / "fig_ext"
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        if extension == "pdf":
            kwargs["metadata"] = {"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None}
        elif extension == "svg":
            kwargs["metadata"] = {"Creator": Path(__file__).name, "Date": None}
        fig.savefig(stem.with_suffix("." + extension), **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
