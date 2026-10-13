#!/usr/bin/env python3
"""Figure 1: one item with and without written steps, and accuracy as updates accumulate.

Example aligned and framed on 2026-10-11. Redesigned on 2026-10-10. A coauthor read the former four-box overview as a pipeline, and the user asked for the
composition of the benchmark papers surveyed that day (an example on the left and a result with a named gap on
the right, as in Code Simulation Challenges, BABILong and Let Me Speak Freely?). Frames are square and white.
(a) Item A-k3-s2026-p0 of the frozen bank (gold 12) as answered by Claude Haiku 4.5 under the core protocol
    ("28", wrong) and under the written-steps protocol (correct). The response lines are short excerpts of the
    stored responses, copied with their ASCII hyphens.
(b) Pooled accuracy of the 18 models by the number of updates K. The core protocol uses the one-integer rule
    and the written-steps protocol the last-integer rule, both on the 60 core items (K = 3, 5, 7), read from
    protocol_pairs.csv (comparison A_to_I). The core protocol continues on the depth items (K = 10 to 100,
    other items from the same generator), read from stability_20261004/depth_pooled.csv. Single-step
    arithmetic (separate items, core protocol) is read from BFCL_INPUTS_20261004.csv.
The figure never draws an internal state: final answers cannot show how a model carries the state.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches  # noqa: F401  (Ellipse in panel a)

from figstyle import COLORS, MM, SOURCE, TEXT_WIDTH, apply_style, save

FS = 6.2
GREY = "#555555"
MONO = "DejaVu Sans Mono"


def title(ax, x, y, letter, text):
    ax.text(x, y, letter, fontsize=8.5, fontweight="bold", va="top")
    ax.text(x + .02, y - .006, text, fontsize=7, fontweight="bold", va="top")


def frame(ax, x, y, w, h):
    ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=.5, edgecolor="#9A9A9A", facecolor="white"))


def sheet(ax, x, y, w, h):
    """A note with a folded corner, drawn with two polygons."""
    from matplotlib.patches import Polygon
    f = .14 * w
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f * 2.6), (x + w - f, y + h), (x, y + h)],
                         closed=True, facecolor="white", edgecolor="#8C8C8C", linewidth=.6))
    ax.add_patch(Polygon([(x + w - f, y + h), (x + w - f, y + h - f * 2.6), (x + w, y + h - f * 2.6)],
                         closed=True, facecolor="#E6E6E6", edgecolor="#8C8C8C", linewidth=.6))


def by_k():
    core, steps = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    with (SOURCE / "protocol_pairs.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["comparison"] != "A_to_I":
                continue
            k = int(row["source_item_id"].split("-k")[1].split("-")[0])
            core[k][0] += int(row["a_correct"]); core[k][1] += 1
            steps[k][0] += int(row["variant_correct"]); steps[k][1] += 1
    assert sum(n for _, n in core.values()) == 1080 and sum(c for c, _ in steps.values()) == 950
    with (SOURCE / "stability_20261004/depth_pooled.csv").open() as stream:
        depth = {int(r["k"]): 100 * float(r["accuracy"]) for r in csv.DictReader(stream) if r["block"] == "B"}
    with (SOURCE / "BFCL_INPUTS_20261004.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    arith = 100 * sum(int(r["C_correct"]) for r in rows) / sum(int(r["C_n"]) for r in rows)
    pct = lambda d: {k: 100 * c / n for k, (c, n) in sorted(d.items())}
    return pct(core), pct(steps), dict(sorted(depth.items())), arith


def head(ax, x0, y0, w, h, color):
    """Our own line drawing of a head in profile, facing right (no external artwork)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    pts = [((.30, .00), Path.MOVETO), ((.30, .18), Path.LINETO),
           ((.05, .35), Path.CURVE4), ((.00, .75), Path.CURVE4), ((.35, .97), Path.CURVE4),
           ((.65, 1.10), Path.CURVE4), ((.95, .90), Path.CURVE4), ((.92, .62), Path.CURVE4),
           ((1.00, .48), Path.LINETO), ((.92, .44), Path.LINETO), ((.93, .36), Path.LINETO),
           ((.93, .28), Path.CURVE4), ((.85, .25), Path.CURVE4), ((.72, .26), Path.CURVE4),
           ((.70, .00), Path.LINETO)]
    verts = [(x0 + u * w, y0 + v * h) for (u, v), _ in pts]
    ax.add_patch(PathPatch(Path(verts, [c for _, c in pts]), facecolor="white", edgecolor=color, linewidth=.8))


# Claude symbol, path data copied from https://claude.ai/favicon.svg (viewBox 0 0 248 248, fill #D97757), 2026-10-11.
CLAUDE_SYMBOL = ("M52.4285 162.873L98.7844 136.879L99.5485 134.602L98.7844 133.334H96.4921L88.7237 132.862L62.2346 132.153L39.3113 131.207L17.0249 130.026L11.4214 128.844L6.2 121.873L6.7094 118.447L11.4214 115.257L18.171 115.847L33.0711 116.911L55.485 118.447L71.6586 119.392L95.728 121.873H99.5485L100.058 120.337L98.7844 119.392L97.7656 118.447L74.5877 102.732L49.4995 86.1905L36.3823 76.62L29.3779 71.7757L25.8121 67.2858L24.2839 57.3608L30.6515 50.2716L39.3113 50.8623L41.4763 51.4531L50.2636 58.1879L68.9842 72.7209L93.4357 90.6804L97.0015 93.6343L98.4374 92.6652L98.6571 91.9801L97.0015 89.2625L83.757 65.2772L69.621 40.8192L63.2534 30.6579L61.5978 24.632C60.9565 22.1032 60.579 20.0111 60.579 17.4246L67.8381 7.49965L71.9133 6.19995L81.7193 7.49965L85.7946 11.0443L91.9074 24.9865L101.714 46.8451L116.996 76.62L121.453 85.4816L123.873 93.6343L124.764 96.1155H126.292V94.6976L127.566 77.9197L129.858 57.3608L132.15 30.8942L132.915 23.4505L136.608 14.4708L143.994 9.62643L149.725 12.344L154.437 19.0788L153.8 23.4505L150.998 41.6463L145.522 70.1215L141.957 89.2625H143.994L146.414 86.7813L156.093 74.0206L172.266 53.698L179.398 45.6635L187.803 36.802L193.152 32.5484H203.34L210.726 43.6549L207.415 55.1159L196.972 68.3492L188.312 79.5739L175.896 96.2095L168.191 109.585L168.882 110.689L170.738 110.53L198.755 104.504L213.91 101.787L231.994 98.7149L240.144 102.496L241.036 106.395L237.852 114.311L218.495 119.037L195.826 123.645L162.07 131.592L161.696 131.893L162.137 132.547L177.36 133.925L183.855 134.279H199.774L229.447 136.524L237.215 141.605L241.8 147.867L241.036 152.711L229.065 158.737L213.019 154.956L175.45 145.977L162.587 142.787H160.805V143.85L171.502 154.366L191.242 172.089L215.82 195.011L217.094 200.682L213.91 205.172L210.599 204.699L188.949 188.394L180.544 181.069L161.696 165.118H160.422V166.772L164.752 173.152L187.803 207.771L188.949 218.405L187.294 221.832L181.308 223.959L174.813 222.777L161.187 203.754L147.305 182.486L136.098 163.345L134.745 164.2L128.075 235.42L125.019 239.082L117.887 241.8L111.902 237.31L108.718 229.984L111.902 215.452L115.722 196.547L118.779 181.541L121.58 162.873L123.291 156.636L123.14 156.219L121.773 156.449L107.699 175.752L86.304 204.699L69.3663 222.777L65.291 224.431L58.2867 220.768L58.9235 214.27L62.8713 208.48L86.304 178.705L100.44 160.155L109.551 149.507L109.462 147.967L108.959 147.924L46.6977 188.512L35.6182 189.93L30.7788 185.44L31.4156 178.115L33.7079 175.752L52.4285 162.873Z")


def logo_path(d=CLAUDE_SYMBOL, size=248.0):
    """Single SVG path (absolute M, L, H, V, C, Z) to a matplotlib Path in unit coordinates, y up."""
    import re
    from matplotlib.path import Path
    verts, codes, cur, start = [], [], (0.0, 0.0), (0.0, 0.0)
    for cmd, args in re.findall(r'([MLHVCZ])([^MLHVCZ]*)', d):
        nums = [float(v) for v in re.findall(r'-?[\d.]+', args)]
        if cmd == "M":
            cur = start = (nums[0], nums[1]); verts.append(cur); codes.append(Path.MOVETO)
        elif cmd == "L":
            for i in range(0, len(nums), 2):
                cur = (nums[i], nums[i + 1]); verts.append(cur); codes.append(Path.LINETO)
        elif cmd == "H":
            cur = (nums[0], cur[1]); verts.append(cur); codes.append(Path.LINETO)
        elif cmd == "V":
            cur = (cur[0], nums[0]); verts.append(cur); codes.append(Path.LINETO)
        elif cmd == "C":
            for i in range(0, len(nums), 6):
                pts = [(nums[i], nums[i + 1]), (nums[i + 2], nums[i + 3]), (nums[i + 4], nums[i + 5])]
                verts += pts; codes += [Path.CURVE4] * 3; cur = pts[-1]
        elif cmd == "Z":
            verts.append(start); codes.append(Path.CLOSEPOLY); cur = start
    return Path([(x / size, 1 - y / size) for x, y in verts], codes)


def main():
    apply_style()
    core, steps, depth, arith = by_k()
    pooled_core, pooled_steps = sum(core.values()) / len(core), sum(steps.values()) / len(steps)
    fig = plt.figure(figsize=(TEXT_WIDTH, 50 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # a  translation from a human working-memory updating task (Miyake et al., 2000; updating that transforms a held
    #    value, Ecker et al., 2010) to the LLM probe, row by row, with one stored example at the bottom.
    #    Rows follow the 10-10 reviews: the probe's sequence stays visible, so "Notes" says only that little room is
    #    left for written steps, and "Length" (not load) is what the probe varies.
    xa, wa = .004, .525
    fs = FS - .4
    title(ax, xa, .985, "a", "From a human working-memory task to the LLM probe")
    xl, xh, xr, xp = xa + .010, xa + .082, xa + .258, xa + .284
    ax.text(xh, .875, "Human updating task", fontsize=FS - .2, fontweight="bold", va="top")
    ax.text(xp, .875, "LLM probe", fontsize=FS - .2, fontweight="bold", va="top")
    ax.plot([xa + .002, xa + wa + .006], [.83, .83], color="#9A9A9A", linewidth=.5)
    # 10-11 review: core updates transform a held value rather than
    #    replace it (transformation as a component of updating, Ecker et al., 2010), and what is never written in the
    #    prompt is each entity's running value.
    rows = [("Hold", "a few items", "running values of three entities"),
            ("Update", "each operation changes\na value held in mind", "gains, losses, and transfers"),
            ("Length", "a short series of changes", "K = 3 to 100 updates"),
            ("Notes", "none, values kept in mind", "little room (16 tokens, one integer)"),
            ("Report", "the final values", "one final value, checked against\na computed answer")]
    y = .80
    for head_, human, probe in rows:
        nl = max(human.count("\n"), probe.count("\n")) + 1
        ax.text(xl, y, head_, fontsize=fs, color=GREY, fontweight="bold", va="top")
        ax.text(xh, y, human, fontsize=fs, va="top", linespacing=1.15)
        ax.annotate("", (xr + .016, y - .019), (xr - .004, y - .019),
                    arrowprops=dict(arrowstyle="-|>", color="#666666", lw=.6, mutation_scale=6, shrinkA=0, shrinkB=0))
        ax.text(xp, y, probe, fontsize=fs, va="top", linespacing=1.15)
        y -= .046 * nl + .020
    # 10-10: the user asked for a border around the table so that it reads as one unit
    ax.add_patch(plt.Rectangle((xa + .002, y + .012), wa + .004, .905 - (y + .012), linewidth=.6,
                               edgecolor="#4D4D4D", facecolor="none"))
    # one stored example (Claude Haiku 4.5, item A-k3-s2026-p0): probe and written-steps control
    example_text_start = len(ax.texts)
    example_patch_start = len(ax.patches)
    ye = y - .02
    ax.text(xl, ye, "Example", fontsize=fs, color=GREY, fontweight="bold", va="top")
    ax.text(xh, ye, "Leo 15, James 18, Carol 15. Carol gives 2 to Leo.\nLeo loses 2. Leo loses 3. Leo? (answer: 12)",
            fontsize=fs, va="top", linespacing=1.15)
    ax.text(xa + wa - .004, ye, "Claude Haiku 4.5", fontsize=fs, color=GREY, ha="right", va="top")
    yy = ye - .135
    for key, name, out, ok in [("A", "Core, answer only", "\u201c28\u201d", False),
                               ("I", "Written steps (control)", "\u201c15 + 2 = 17 \u2026 15 - 3 = 12\u201d", True)]:
        ax.add_patch(plt.Rectangle((xh, yy - .03), .0085, .0275, facecolor=COLORS[key], edgecolor="none"))
        ax.text(xh + .014, yy, name, fontsize=fs, va="top")
        resp = ax.text(xh + .158, yy, out, fontsize=fs, va="top")
        fig.canvas.draw()
        right = ax.transData.inverted().transform(resp.get_window_extent())[1][0]
        ax.text(right + .010, yy, "correct" if ok else "wrong", fontsize=fs, fontweight="bold", va="top")
        yy -= .068

    example_artists = list(ax.texts)[example_text_start:] + list(ax.patches)[example_patch_start:]

    # b  accuracy by number of updates. The probe (core protocol, answer only) is the subject; the written-steps
    #    protocol is the control that removes the working-memory-like demand.
    xc = .565
    title(ax, xc, .985, "b", "Accuracy falls as updates accumulate")
    sub = fig.add_axes([xc + .045, .16, .995 - xc - .06, .70])
    ks = list(core)
    sub.fill_between(ks, [core[k] for k in ks], [steps[k] for k in ks], color="#E6E6E6", linewidth=0, zorder=1)
    sub.plot(ks, [steps[k] for k in ks], color=COLORS["I"], linewidth=1.3, marker="o", markersize=3.2,
             markeredgecolor="white", markeredgewidth=.5, zorder=3)
    sub.plot(ks, [core[k] for k in ks], color=COLORS["A"], linewidth=1.3, marker="o", markersize=3.2,
             markeredgecolor="white", markeredgewidth=.5, zorder=3)
    dk = [ks[-1]] + list(depth)
    sub.plot(dk, [core[ks[-1]]] + list(depth.values()), color=COLORS["A"], linewidth=1.1, linestyle=(0, (3, 2)),
             marker="o", markersize=2.8, markerfacecolor="white", markevery=list(range(1, len(dk))), zorder=2)
    sub.plot([1], [arith], marker="D", markersize=3.4, color=COLORS["C"], linestyle="none", zorder=3)
    sub.text(1.12, arith - 4, f"Arithmetic,\n1 update\n{arith:.1f}%", fontsize=FS - .6, color="#8A5A00",
             ha="left", va="top", linespacing=1.1)
    sub.text(5, steps[5] + 5, "Written steps (control)", fontsize=FS - .3, color="#A55D27", ha="center", va="bottom")
    sub.text(3.05, 19, "Core,\nanswer only", fontsize=FS - .3, color=COLORS["A"], ha="left", va="top", linespacing=1.1)
    sub.text(11, depth[10] + 6, "Depth, answer only", fontsize=FS - .6, color=COLORS["A"], ha="left", va="bottom")
    sub.text(7.6, (core[7] + steps[7]) / 2, f"Across all core items\n{pooled_core:.1f}% \u2192 {pooled_steps:.1f}%",
             fontsize=FS - .5, color="#333333", ha="left", va="center", linespacing=1.15)
    sub.set_xscale("log")
    sub.set_xticks([1, 3, 5, 7, 10, 20, 50, 100], ["1", "3", "5", "7", "10", "20", "50", "100"])
    sub.minorticks_off()
    sub.set_xlim(.85, 112); sub.set_ylim(0, 100); sub.set_yticks([0, 25, 50, 75, 100])
    sub.tick_params(labelsize=6.5, length=2)
    sub.set_xlabel("Updates, K (log scale)", fontsize=7, labelpad=1.5)
    sub.set_ylabel("Correct (%), 18 models", fontsize=7, labelpad=1.5)
    # Lift the example 2 mm above its former alignment with the bottom of the right x-axis title.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    before_bottom = min(a.get_window_extent(renderer).y0 for a in example_artists)
    padding_y = 1.0 / (fig.get_figheight() * 25.4)
    frame_linewidth = .6
    example_lift_mm = 2.0
    example_lift_px = example_lift_mm * fig.dpi / 25.4
    half_stroke_px = frame_linewidth * fig.dpi / 144
    target_bottom = (sub.xaxis.label.get_window_extent(renderer).y0
                     + padding_y * fig.bbox.height + half_stroke_px + example_lift_px)
    shift_y = ax.transData.inverted().transform((0, target_bottom))[1] - ax.transData.inverted().transform((0, before_bottom))[1]
    for artist in example_artists:
        if isinstance(artist, matplotlib.text.Text):
            x, y = artist.get_position()
            artist.set_position((x, y + shift_y))
        else:
            artist.set_y(artist.get_y() + shift_y)
    fig.canvas.draw()
    after_bottom = min(a.get_window_extent(fig.canvas.get_renderer()).y0 for a in example_artists)
    assert abs(after_bottom - target_bottom) < 1e-6
    assert after_bottom >= fig.bbox.y0
    # A matching thin square frame groups the complete example without moving its content.
    renderer = fig.canvas.get_renderer()
    bounds = [a.get_window_extent(renderer) for a in example_artists]
    bottom = ax.transData.inverted().transform((0, min(b.y0 for b in bounds)))[1]
    top = ax.transData.inverted().transform((0, max(b.y1 for b in bounds)))[1]
    border = plt.Rectangle((xa + .002, bottom - padding_y), wa + .004,
                           top - bottom + 2 * padding_y, linewidth=frame_linewidth,
                           edgecolor="#4D4D4D", facecolor="none", zorder=1, clip_on=False)
    ax.add_patch(border)
    # Add 1 mm below the original canvas for the frame; preserve every artist's physical size and alignment.
    height_mm = fig.get_figheight() * 25.4
    canvas_padding_mm = 1.0
    fig.set_size_inches(fig.get_figwidth(), (height_mm + canvas_padding_mm) * MM)
    for axes in fig.axes:
        x0, y0, w, h = axes.get_position().bounds
        axes.set_position([x0, (y0 * height_mm + canvas_padding_mm) / (height_mm + canvas_padding_mm),
                           w, h * height_mm / (height_mm + canvas_padding_mm)])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    after_bottom = min(a.get_window_extent(renderer).y0 for a in example_artists)
    target_bottom = sub.xaxis.label.get_window_extent(renderer).y0 + example_lift_px
    border_bounds = border.get_window_extent(renderer)
    assert border_bounds.y0 - half_stroke_px >= fig.bbox.y0, "Example border extends below the canvas"
    assert abs(border_bounds.y0 - half_stroke_px - target_bottom) < 1e-6, "Example frame is not 2 mm above the x-axis label bottom"
    assert border_bounds.y1 < ax.transData.transform((0, ax.patches[example_patch_start - 1].get_y()))[1], "Example border reaches the table"
    # 2026-10-11 (author decision): official Claude symbol left of the model name, as tall as the text, in its own orange
    name = next(t for t in ax.texts if t.get_text() == "Claude Haiku 4.5")
    fig.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(name.get_window_extent())
    h_mm = (y1 - y0) * fig.get_figheight() * 25.4
    w = h_mm / (fig.get_figwidth() * 25.4)
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    mark = logo_path()
    ax.add_patch(PathPatch(mark, transform=Affine2D().scale(w, y1 - y0).translate(x0 - w - .004, y0) + ax.transData,
                           facecolor="#D97757", edgecolor="none"))
    save(fig, "fig_framework")


if __name__ == "__main__":
    main()
