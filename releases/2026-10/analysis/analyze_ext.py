#!/usr/bin/env python3
"""Analyse blocks K and L from scored_items.csv (output of score_ext.py).

Everything is descriptive. Intervals are 95% percentile intervals from 10,000
paired model resamples (numpy.random.RandomState(42); one index draw per
resample, shared by every statistic), the convention of the other analyses.
Models are the resampling unit; items are not resampled.

K  accuracy by query position q = 1, 2, 3 (20 trajectories each, identical text
   up to the question), the paired differences q2-q1, q3-q2, q3-q1, and the
   reference state that wrong one-integer answers equal. Also accuracy on the
   items where the slot position alone predicts the number of updates
   (ceil(slot / 3) equals the true count) and on the items where it does not.
L  accuracy by number of targets M = 1, 2, 4 for tracking and for the matched
   readout control, the paired gap readout-tracking at each M, and the change
   in tracking accuracy from M=1 to M=4.

Usage: python analyze_ext.py --scored scored_items.csv --out-dir DIR [--allow-incomplete]
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT, SEED = 10_000, 42
Q_LEVELS = (1, 2, 3)
M_LEVELS = (1, 2, 4)
CONDITIONS = ("tracking", "readout")


def interval(per_model: np.ndarray, draws_index: np.ndarray) -> list[float]:
    means = per_model[draws_index].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="analyse although some trials are missing; missing trials leave the denominators")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.scored.open(encoding="utf-8")))
    models = sorted({row["model"] for row in rows})
    n_missing = sum(row["status"] == "missing" for row in rows)
    if n_missing and not args.allow_incomplete:
        raise SystemExit(f"{n_missing} missing trials; rerun the missing requests or pass --allow-incomplete")

    def cell_key(row: dict) -> tuple:
        if row["block"] == "K":
            return ("K", int(row["query_after"]))
        return ("L", int(row["target_count"]), row["condition"])

    counts: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for row in rows:
        if row["status"] != "answered":
            continue
        cell = counts[cell_key(row)][row["model"]]
        cell[0] += int(row["correct"])
        cell[1] += 1
        cell[2] += int(row["compliant"])
        cell[3] += int(row["truncated"])

    def accuracy(key: tuple) -> np.ndarray:
        values = []
        for model in models:
            correct, total, _, _ = counts[key][model]
            if total == 0:
                raise SystemExit(f"{model}: no answered trials in cell {key}")
            values.append(correct / total)
        return np.array(values)

    rng = np.random.RandomState(SEED)
    draws_index = np.stack([rng.choice(len(models), size=len(models), replace=True) for _ in range(N_BOOT)])

    def summary(per_model: np.ndarray) -> dict:
        return {"mean_over_models": float(per_model.mean()), "ci95": interval(per_model, draws_index),
                "min": float(per_model.min()), "max": float(per_model.max())}

    def pooled(key: tuple) -> dict:
        totals = np.array([counts[key][model] for model in models]).sum(axis=0)
        return {"correct": int(totals[0]), "answered": int(totals[1]), "compliant": int(totals[2]),
                "truncated": int(totals[3])}

    k_acc = {q: accuracy(("K", q)) for q in Q_LEVELS}
    l_acc = {(m, c): accuracy(("L", m, c)) for m in M_LEVELS for c in CONDITIONS}

    position: dict[str, dict[str, list[int]]] = {"1": defaultdict(lambda: [0, 0]), "0": defaultdict(lambda: [0, 0])}
    for row in rows:
        if row["block"] == "K" and row["status"] == "answered":
            cell = position[row["position_guess_correct"]][row["model"]]
            cell[0] += int(row["correct"])
            cell[1] += 1

    def position_accuracy(flag: str) -> np.ndarray:
        return np.array([position[flag][model][0] / position[flag][model][1] for model in models])

    k_errors: dict[int, dict[str, int]] = {q: defaultdict(int) for q in Q_LEVELS}
    for row in rows:
        if row["block"] == "K" and row["status"] == "answered":
            k_errors[int(row["query_after"])][row["k_error_type"]] += 1

    result = {
        "meta": {
            "n_models": len(models), "models": models, "missing_trials": n_missing,
            "status": "exploratory; added after the main evaluation; scoring rules fixed before any model was run",
            "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                          "unit": "model", "interval": "2.5th and 97.5th percentile",
                          "pairing": "one index draw per resample, shared by all statistics"},
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
        },
        "K": {
            "accuracy_by_query_position": {f"q{q}": {**summary(k_acc[q]), **pooled(("K", q))} for q in Q_LEVELS},
            "paired_differences": {name: summary(k_acc[b] - k_acc[a])
                                   for name, (a, b) in {"q2_minus_q1": (1, 2), "q3_minus_q2": (2, 3),
                                                        "q3_minus_q1": (1, 3)}.items()},
            "response_types": {f"q{q}": dict(sorted(k_errors[q].items())) for q in Q_LEVELS},
            "by_position_guess": {
                "definition": "position guess = ceil(slot / 3); 'agrees' items are those where it equals the true update count",
                "agrees": {**summary(position_accuracy("1")),
                           "correct": int(sum(position["1"][m][0] for m in models)),
                           "answered": int(sum(position["1"][m][1] for m in models))},
                "disagrees": {**summary(position_accuracy("0")),
                              "correct": int(sum(position["0"][m][0] for m in models)),
                              "answered": int(sum(position["0"][m][1] for m in models))},
                "agrees_minus_disagrees": summary(position_accuracy("1") - position_accuracy("0")),
            },
        },
        "L": {
            "accuracy": {f"M{m}_{c}": {**summary(l_acc[(m, c)]), **pooled(("L", m, c))}
                         for m in M_LEVELS for c in CONDITIONS},
            "readout_minus_tracking": {f"M{m}": summary(l_acc[(m, "readout")] - l_acc[(m, "tracking")])
                                       for m in M_LEVELS},
            "tracking_M4_minus_M1": summary(l_acc[(4, "tracking")] - l_acc[(1, "tracking")]),
            "readout_M4_minus_M1": summary(l_acc[(4, "readout")] - l_acc[(1, "readout")]),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with (args.out_dir / "accuracy_by_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["model"] + [f"K_q{q}" for q in Q_LEVELS] + [f"L_M{m}_{c}" for m in M_LEVELS for c in CONDITIONS]
        writer.writerow(header + ["K_compliant", "K_truncated", "L_compliant", "L_truncated", "answered"])
        for index, model in enumerate(models):
            k_cells = [counts[("K", q)][model] for q in Q_LEVELS]
            l_cells = [counts[("L", m, c)][model] for m in M_LEVELS for c in CONDITIONS]
            writer.writerow([model] + [f"{cell[0]}/{cell[1]}" for cell in k_cells + l_cells]
                            + [sum(c[2] for c in k_cells), sum(c[3] for c in k_cells),
                               sum(c[2] for c in l_cells), sum(c[3] for c in l_cells),
                               sum(c[1] for c in k_cells + l_cells)])

    def pct(value: float) -> str:
        return f"{100 * value:.1f}"

    lines = ["# Extension blocks K and L", "",
             f"{len(models)} models; {n_missing} missing trials. Exploratory; added after the main evaluation.",
             "Intervals: 95% percentile, 10,000 paired model resamples (RandomState 42).", "",
             "## K: intermediate-state queries (20 trajectories per position, per model)", "",
             "| Query position | Correct / answered | Mean accuracy over models (%) | 95% interval |",
             "| --- | --- | --- | --- |"]
    for q in Q_LEVELS:
        cell = result["K"]["accuracy_by_query_position"][f"q{q}"]
        lines.append(f"| after update {q} | {cell['correct']} / {cell['answered']} | {pct(cell['mean_over_models'])} | "
                     f"[{pct(cell['ci95'][0])}, {pct(cell['ci95'][1])}] |")
    lines += ["", "| Paired difference | Mean (points) | 95% interval |", "| --- | --- | --- |"]
    for name, cell in result["K"]["paired_differences"].items():
        lines.append(f"| {name.replace('_', ' ')} | {pct(cell['mean_over_models'])} | "
                     f"[{pct(cell['ci95'][0])}, {pct(cell['ci95'][1])}] |")
    lines += ["", "Response types (counts over all models):", ""]
    for q in Q_LEVELS:
        lines.append(f"- after update {q}: " + ", ".join(f"{name} {count}" for name, count in
                                                         result["K"]["response_types"][f"q{q}"].items()))
    by_position = result["K"]["by_position_guess"]
    lines += ["", "Items where the slot position alone predicts the update count (ceil(slot / 3)): "
                  f"{by_position['agrees']['correct']} / {by_position['agrees']['answered']} correct, mean over models "
                  f"{pct(by_position['agrees']['mean_over_models'])}%. Items where it does not: "
                  f"{by_position['disagrees']['correct']} / {by_position['disagrees']['answered']} correct, mean over models "
                  f"{pct(by_position['disagrees']['mean_over_models'])}%. Difference "
                  f"{pct(by_position['agrees_minus_disagrees']['mean_over_models'])} points "
                  f"[{pct(by_position['agrees_minus_disagrees']['ci95'][0])}, {pct(by_position['agrees_minus_disagrees']['ci95'][1])}]."]
    lines += ["", "## L: multi-target tracking and matched readout control (20 trajectories per cell, per model)", "",
              "| Targets M | Tracking correct / answered | Tracking mean (%) | Readout correct / answered | "
              "Readout mean (%) | Readout minus tracking (points) | 95% interval |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in M_LEVELS:
        tracking = result["L"]["accuracy"][f"M{m}_tracking"]
        readout = result["L"]["accuracy"][f"M{m}_readout"]
        gap = result["L"]["readout_minus_tracking"][f"M{m}"]
        lines.append(f"| {m} | {tracking['correct']} / {tracking['answered']} | {pct(tracking['mean_over_models'])} | "
                     f"{readout['correct']} / {readout['answered']} | {pct(readout['mean_over_models'])} | "
                     f"{pct(gap['mean_over_models'])} | [{pct(gap['ci95'][0])}, {pct(gap['ci95'][1])}] |")
    change = result["L"]["tracking_M4_minus_M1"]
    lines += ["", f"Tracking accuracy, M=4 minus M=1: {pct(change['mean_over_models'])} points "
                  f"[{pct(change['ci95'][0])}, {pct(change['ci95'][1])}]."]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote analysis.json, accuracy_by_model.csv, REPORT.md in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
