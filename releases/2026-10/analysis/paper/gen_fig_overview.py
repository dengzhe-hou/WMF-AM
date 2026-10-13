#!/usr/bin/env python3
"""Figure 2: methods overview of the probe.
Four-panel layout with explicit protocol and comparison labels.
No data are read; the example is item A-k3-s2026-p0 of the frozen bank (gold 12).

The answer line under the item (15 + 2 - 2 - 3 = 12) was added on 2026-10-06 at the user's request."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import COLORS, LABELS, MM, TEXT_WIDTH, apply_style, save

FS = 6.2


def box(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=.7, edgecolor="#4D4D4D", facecolor="white"))
    ax.text(x + .012, y + h - .035, title, fontsize=7.2, fontweight="bold", va="top")


def main():
    apply_style()
    fig = plt.figure(figsize=(TEXT_WIDTH, 50 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    top, h = .96, .92
    xs = [.006, .282, .538, .786]; ws = [.236, .216, .208, .208]

    # 1 generate
    box(ax, xs[0], top - h, ws[0], h, "1  Generate an item")
    item = ("Leo: 15 points, James: 18 points,\nCarol: 15 points\n"
            "1. Carol gives 2 points to Leo.\n2. Leo loses 2 points.\n3. Leo loses 3 points.\n"
            "How many points does Leo have?")
    t = ax.text(xs[0] + .013, top - .13, item, fontsize=5.0, va="top", family="DejaVu Sans Mono", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["light"], edgecolor="none"))
    fig.canvas.draw()
    y_item = ax.transData.inverted().transform(t.get_bbox_patch().get_window_extent())[0][1]
    ax.text(xs[0] + .013, y_item - .08, "Code computes the answer:", fontsize=FS, va="top")
    ax.text(xs[0] + .013, y_item - .17, "15 + 2 \u2212 2 \u2212 3 = 12", fontsize=FS, va="top")
    ax.text(xs[0] + .013, top - h + .045, "K = 3 to 100 updates per item.", fontsize=FS, va="bottom")

    # 2 vary the response protocol. The core protocol and the instruction
    # variant share the 16-token budget and one-integer rule, so they form one group with little room; the
    # large-budget and written-steps protocols share 512 tokens and form the group with more room. A thin grey
    # bracket marks each group, and a note under its last row names the room (the panel says "steps", as the
    # prompts do). A single arrow from top to bottom was dropped on 10-10: it implied an order inside each group.
    # Each protocol name (NAMING.md short label) runs into the first line of its description, and a small
    # square in the protocol's color keys it.
    box(ax, xs[1], top - h, ws[1], h, "2  Vary the response\n    protocol")
    groups = [("little room to write steps",
               [("A", "\u201cRespond with ONLY", ["the final number.\u201d"], "16 tokens \u00b7 one integer"),
                ("H", "", ["new user/system prompts"], "16 tokens \u00b7 one integer")]),
              ("more room to write steps",
               [("A512", "core prompt", [], "512 tokens\none-integer / last-integer"),
                ("I", "", ["\u201cThink step by step \u2026\u201d"], "512 tokens \u00b7 last integer")])]
    pitch, gap = .045, .024
    x_bar, x_key, x_text = xs[1] + .009, xs[1] + .017, xs[1] + .031
    y = top - .18
    to_data = ax.transData.inverted()
    for note, rows in groups:
        y_top = y
        for key, first, more, tok in rows:
            ax.add_patch(plt.Rectangle((x_key, y - .034), .0085, .0275, facecolor=COLORS[key], edgecolor="none"))
            name = ax.text(x_text, y, LABELS[key], fontsize=FS, va="top", fontweight="bold")
            if first:
                fig.canvas.draw()
                name_right = to_data.transform(name.get_window_extent())[1][0]
                ax.text(name_right + .008, y, first, fontsize=FS, va="top")
            for line in more:
                y -= pitch
                ax.text(x_text, y, line, fontsize=FS, va="top")
            y -= pitch
            for i, token_line in enumerate(tok.split("\n")):
                if i:
                    y -= pitch
                ax.text(x_text, y, token_line, fontsize=FS - .3, va="top", color="#555555")
            y -= pitch + gap
        y += gap
        ax.text(x_text, y, note, fontsize=FS - .3, va="top", color="#555555", style="italic")
        ax.plot([x_bar, x_bar], [y_top - .008, y - pitch + .008], color=COLORS["grey"], linewidth=.6,
                solid_capstyle="butt")
        y -= pitch + gap

    # 3 score
    box(ax, xs[2], top - h, ws[2], h, "3  Score the answer")
    ax.text(xs[2] + .013, top - .13, "Response", fontsize=FS - .3, color="#555555", va="top")
    ax.text(xs[2] + .085, top - .13, "Result", fontsize=FS - .3, color="#555555", va="top")
    examples = [("12", "correct", "#1A1A1A"),
                ("28", "one integer, wrong", "#1A1A1A"),
                ("17, 15, 12", "several integers, wrong by\nthe one-integer rule, correct\nby the last-integer rule", "#555555")]
    y = top - .23
    for resp, res, col in examples:
        ax.text(xs[2] + .013, y, "“" + resp + "”", fontsize=FS, va="top", family="DejaVu Sans Mono")
        rx = xs[2] + .085
        if len(resp) > 4:          # long response: its result goes on the lines below
            y -= .075
            rx = xs[2] + .022
        ax.text(rx, y, res, fontsize=FS, va="top", color=col, linespacing=1.15)
        y -= .11 + .062 * res.count("\n")
    ax.text(xs[2] + .013, top - h + .045, "Responses that hit the token\nlimit are flagged as truncated.",
            fontsize=FS - .3, va="bottom", color="#555555", linespacing=1.25)

    # 4 compare
    box(ax, xs[3], top - h, ws[3], h, "4  Compare")
    pairs = [("Length", "3 to 100 updates"),
             ("States", "earlier vs later\nand 1, 2, or 4 targets"),
             ("Tasks", "core vs arithmetic\nand simpler tasks"),
             ("Variants", "5 wordings,\n4 item seeds"),
             ("External", "probe scores vs\nBFCL scores")]
    y = top - .14
    for head, body in pairs:
        ax.text(xs[3] + .013, y, head, fontsize=FS, va="top", fontweight="bold")
        ax.text(xs[3] + .075, y, body, fontsize=FS, va="top", linespacing=1.15)
        y -= .1 + .062 * body.count("\n")

    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i] + ws[i] + .005, top - h / 2), (xs[i + 1] - .005, top - h / 2),
                                     arrowstyle="-|>", mutation_scale=10, linewidth=1.3, color="#4D4D4D",
                                     shrinkA=0, shrinkB=0))
    save(fig, "fig_overview")


if __name__ == "__main__":
    main()
