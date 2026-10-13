#!/usr/bin/env python3
"""Plot all 18 paired protocol summaries; no new statistical estimates."""
from __future__ import annotations

import csv
import matplotlib.pyplot as plt
from gen_depth import apply_project_style, save_figure, SOURCE, MM
from figstyle import COLORS, LABELS


def main():
    apply_project_style()
    with (SOURCE / "BFCL_INPUTS_20261004.csv").open() as handle:
        models = list(csv.DictReader(handle))
    with (SOURCE / "protocol_conditions.csv").open() as handle:
        conditions = list(csv.DictReader(handle))
    lookup = {(r["model"], r["block"]): r for r in conditions}
    assert len(models) == 18 and len(conditions) == 54
    names = [r["model"].replace("-Instruct", "").replace("-350m", "-350M") for r in models]
    keys = [r["run_label"] for r in models]
    fig, ax = plt.subplots(figsize=(180 * MM, 113 * MM))
    fig.subplots_adjust(left=.31, right=.965, bottom=.13, top=.90)
    ys = list(range(18))
    offsets = {"A": -.18, "H": 0, "I": .18}
    for y, key in zip(ys, keys):
        points = [100 * int(lookup[key, b]["correct"]) / 60 for b in "AHI"]
        ax.plot(points, [y + offsets[b] for b in "AHI"], color="#BBBBBB", linewidth=.7, zorder=1)
    for block, label, color, marker in [
        ("A", LABELS["A"], COLORS["A"], "o"),
        ("H", LABELS["H"], COLORS["H"], "s"),
        ("I", LABELS["I"], COLORS["I"], "^"),
    ]:
        assert all(lookup[key,block]["n_scored"] == "60" for key in keys)
        ax.scatter([100 * int(lookup[key,block]["correct"]) / 60 for key in keys],
                   [y + offsets[block] for y in ys], label=label, color=color,
                   marker=marker, s=15, linewidths=.35, edgecolors="white", zorder=3)
    ax.set_yticks(ys, names)
    ax.set(xlim=(-2, 102), ylim=(17.75, -.75), xlabel="Correct responses (%) on the same 60 base items")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", linewidth=.4, color="#D7D7D7", zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(.42, 1.035), ncol=3, frameon=False,
              handletextpad=.4, columnspacing=1.3)
    # The footer starts at the left edge of the model names, so the longer wording stays inside the figure.
    fig.canvas.draw()
    left = min(t.get_window_extent().x0 for t in ax.get_yticklabels()) / fig.bbox.width
    note = fig.text(left, .015, "All 18 models retained. The instruction variant and the written-steps protocol change multiple protocol components.", fontsize=7)
    fig.canvas.draw()
    assert note.get_window_extent().x1 <= fig.bbox.width, "footer runs past the figure edge"
    save_figure(fig, "protocol")


if __name__ == "__main__":
    main()
