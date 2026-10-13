#!/usr/bin/env python3
"""Figure 3: arithmetic comparison, sequence depth, and response protocols.

Reads the stored CSV/JSON results only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

from figstyle import BUDGET, COLORS, LABELS, MM, SOURCE, TEXT_WIDTH, apply_style
def short(name: str) -> str:
    return (name.replace("-Instruct", "").replace("-350m", "-350M")
            .replace("Claude ", "").replace("granite", "Granite"))



def main() -> None:
    apply_style()
    with (SOURCE / "BFCL_INPUTS_20261004.csv").open() as stream:
        raw = list(csv.DictReader(stream))
    models = [
        (short(row["model"]), 100 * int(row["A_correct"]) / 60,
         100 * int(row["C_correct"]) / 90)
        for row in raw
    ]
    models.sort(key=lambda row: row[1])
    with (SOURCE / "stability_20261004/depth_pooled.csv").open() as stream:
        depth = list(csv.DictReader(stream))
    totals = json.loads((BUDGET / "summary.json").read_text())["totals"]
    assert len(models) == 18 and len(depth) == 10
    assert all(int(row["n_models"]) == len(models) and int(row["n_missing"]) == 0 for row in depth)
    counts = [totals[key] for key in ("A_one", "A_last", "A512_last", "I_last")]
    denominator = len(models) * 60

    # 2026-10-09: 7 mm of empty space below the legend removed (height 72 -> 65 mm). The panels keep their
    # absolute size and the distance of their tops from the figure top, so print sizes do not change.
    fig_height = 65.0
    fig = plt.figure(figsize=(TEXT_WIDTH, fig_height * MM))
    bottom, height = (.29 * 72 - 7.0) / fig_height, .66 * 72 / fig_height
    # 2026-10-09: panel a moved left and widened, panel b widened, so the left margin matches the right one
    # (it was 9 mm) and panel a no longer touches panel b. Positions in mm of the 160 mm width.
    positions = [(left / 160, width / 160) for left, width in ((20.4, 36.1), (72.4, 38.0), (124.2, 34.0))]
    axes = [fig.add_axes([left, bottom, width, height]) for left, width in positions]
    # Panel letters sit at the left edge of each panel's y-axis labels. Positions are set after drawing.

    # a. Same 18 models, same ordering, larger labels and greater row spacing.
    ax = axes[0]
    for index, (_, core, arithmetic) in enumerate(models):
        ax.plot([core, arithmetic], [index, index], color="#C9C9C9", lw=1.0, zorder=1)
        ax.scatter(core, index, s=13, color=COLORS["A"], lw=0, zorder=2)
        ax.scatter(arithmetic, index, s=13, color=COLORS["C"], lw=0, zorder=2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([row[0] for row in models], fontsize=6.2)
    ax.set(xlim=(-3, 103), ylim=(-.8, len(models) - .2), xlabel="Correct (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.scatter([], [], s=13, color=COLORS["A"], label="Core, 3–7 updates")
    ax.scatter([], [], s=13, color=COLORS["C"], label="Arithmetic, 1 update")
    ax.legend(loc="upper center", bbox_to_anchor=(.40, -.185), frameon=False,
              ncol=2, fontsize=6.2, handletextpad=.25, columnspacing=1.1,
              borderaxespad=0, handlelength=1.0)

    # b. The truncation flag overlaps the two answer outcomes; it is not a third category.
    ax = axes[1]
    ks = [int(row["k"]) for row in depth]
    accurate = [100 * float(row["accuracy"]) for row in depth]
    compliant_wrong = [100 * int(row["n_compliant_wrong"]) / int(row["n_expected"]) for row in depth]
    truncated = [100 * float(row["truncation_rate"]) for row in depth]
    ax.plot(ks, accurate, color=COLORS["A"], marker="o", ms=3.0, lw=1.2, label="correct")
    ax.plot(ks, compliant_wrong, color="#6E6E6E", marker="s", ms=2.8, lw=1.2,
            ls="--", label="one integer, wrong")
    ax.plot(ks, truncated, color="#999999", marker="^", ms=3.0, lw=1.1,
            markerfacecolor="white", markeredgewidth=.8, ls=":", label="truncated")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlabel="Updates, K (log scale)", ylabel="Responses (%)",
           xlim=(2.7, 115), ylim=(-2, 102))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="center right", bbox_to_anchor=(1.015, .49), frameon=False,
              fontsize=6.0, handlelength=1.7, handletextpad=.45, labelspacing=.5,
              borderaxespad=0)

    # c. Four zero-baseline bars show correct totals without mixing totals and increments.
    ax = axes[2]
    stage_colors = [COLORS["A"], "#999999", COLORS["A512"], COLORS["I"]]
    for index, (hi, color) in enumerate(zip(counts, stage_colors)):
        ax.bar(index, hi, color=color, width=.66, edgecolor="none")
        ax.text(index, hi + 18, str(hi), ha="center", va="bottom", fontsize=6.6)
    ax.axhline(denominator, color="#BBBBBB", lw=.6, ls="--")
    ax.text(-.5, denominator + 16, f"of {denominator:,}", va="bottom", fontsize=6.0,
            color="#777777")
    # Bars 1 and 2 are the same core responses under the two scoring rules. A row under the protocol
    # names gives each bar's scoring rule, so the bars need no letter codes.
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABELS["A"], LABELS["A"], "Large\nbudget", "Written\nsteps"], fontsize=6.2,
                       linespacing=1.0, va="top")
    ax.tick_params(axis="x", length=0, pad=4)
    rule_y = -172
    for (lo, hi), rule in [((-.3, .3), "one integer"), ((.78, 3.33), "last integer")]:
        ax.plot([lo, hi], [rule_y, rule_y], color="#999999", lw=.6, clip_on=False)
        ax.text((lo + hi) / 2, rule_y - 16, rule, ha="center", va="top", fontsize=6.0, color="#555555",
                clip_on=False)
    ax.set(xlim=(-.62, 3.62), ylim=(0, 1170), ylabel="Correct core responses")
    ax.set_yticks([0, 250, 500, 750, 1000])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for letter, axis in zip("abc", axes):
        labels_left = axis.yaxis.get_tightbbox(renderer).x0
        x = fig.transFigure.inverted().transform((labels_left, 0))[0] - .004
        fig.text(x, 1 - .035 * 72 / fig_height, letter, fontsize=8.5, weight="bold", va="top")

    stem = HERE / "figures" / "fig_main_results"
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
