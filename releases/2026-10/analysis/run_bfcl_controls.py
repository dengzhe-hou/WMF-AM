#!/usr/bin/env python3
"""Exploratory rank-residual analysis.

python analysis/run_bfcl_controls.py
Imports numerical kernels only; never executes analyze.main().
"""

import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from analyze import clean, kendall, model_weights, signs_for, taub
from run_bfcl_comparison import CRITERIA, BLOCKS, SEED, N_BOOT, fmt, interval, summary, write_csv

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
OUT = PANEL / "bfcl_controls_20261004"
SPECS = (
    ("all18", "none", ()),
    ("all18", "single_turn", ("single_turn",)),
    ("known_params12", "none", ()),
    ("known_params12", "single_turn", ("single_turn",)),
    ("known_params12", "log_params", ("log_params",)),
    ("known_params12", "single_turn_and_log_params", ("single_turn", "log_params")),
)


def compute(rows, columns):
    """Refit ranks/residuals within each common bootstrap sample."""
    n = len(rows)
    weights = model_weights(n, N_BOOT, SEED)
    x = {b: np.array([int(r[b + "_correct"]) for r in rows]) for b in BLOCKS}
    z = {"single_turn": np.array([float(r["bfcl_single_turn_mean_1216"]) for r in rows])}
    if "log_params" in columns:
        z["log_params"] = np.log10([int(r["params"]) for r in rows])
    controls = [z[c] for c in columns]
    results, differences = [], []
    for criterion in CRITERIA:
        y = np.array([float(r[f"bfcl_{criterion}_1216"]) for r in rows])
        points, boot = {}, {}
        for b in BLOCKS:
            points[b] = clean(kendall(x[b], y, controls))
            boot[b] = taub(*signs_for(x[b], y, controls, weights), weights)
            results.append({"criterion": criterion, "block": b, "n_models": n,
                            "tau_b": points[b], **summary(boot[b])})
        for b in ("C", "D"):
            differences.append({"criterion": criterion, "comparison": f"A-{b}", "n_models": n,
                                "delta_tau_b": points["A"] - points[b]
                                if points["A"] is not None and points[b] is not None else None,
                                **summary(boot["A"] - boot[b])})
    return results, differences


def main():
    source = PANEL / "BFCL_INPUTS_20261004.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    assert len(all_rows) == len({r["model"] for r in all_rows}) == 18
    panels = {"all18": all_rows, "known_params12": [r for r in all_rows if r["params"]]}
    assert len(panels["known_params12"]) == 12
    associations, differences = [], []
    for panel, adjustment, columns in SPECS:
        a, d = compute(panels[panel], columns)
        prefix = {"panel": panel, "adjustment": adjustment,
                  "statistic": "raw Kendall tau-b" if not columns else "Kendall tau-b on rank residuals"}
        associations.extend({**prefix, **r} for r in a)
        differences.extend({**prefix, **r} for r in d)
    meta = {
        "source": str(source),
        "status": "exploratory, after model scores and unadjusted associations were visible",
        "panels": {p: [r["model"] for r in rows] for p, rows in panels.items()},
        "params_unavailable": [r["model"] for r in all_rows if not r["params"]],
        "statistic": "Average ranks of X,Y,Z; ordinary linear rank residuals; Kendall tau-b between residuals",
        "name_limit": "Not standard partial Kendall, a conditional-independence test, or held-out prediction",
        "controls": {"single_turn": "(BFCL Non-Live + Live)/2, 2025-12-16",
                     "log_params": "log10 public parameter count; rank-equivalent to the positive count"},
        "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                      "pairing": "Same model draws across all predictors, criteria and adjustments within each panel",
                      "refit": "Recompute ranks and refit every residual regression in every draw",
                      "ci": "Pointwise 95% percentile, defined draws only, no multiplicity adjustment",
                      "undefined": "Retain counts; do not replace with zero"},
        "residual_ties": "Existing analyze.py threshold abs(difference)<=1e-9",
        "limitations": ["Only model sampling uncertainty; not task sampling",
                        "Convenience panel with within-provider dependence",
                        "Adjustment does not remove prompt/compliance/truncation confounding",
                        "No residual permutation p or automatic claim threshold"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "associations.csv", associations)
    write_csv(OUT / "paired_differences.csv", differences)
    (OUT / "analysis.json").write_text(json.dumps({"meta": meta, "associations": associations,
        "paired_differences": differences}, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    labels = {"none": "unadjusted", "single_turn": "single-turn BFCL", "log_params": "parameter count",
              "single_turn_and_log_params": "single-turn BFCL + parameter count"}
    lines = ["# Exploratory BFCL control analysis", "",
             "The full 18-model panel is retained; the 12 models with available parameter counts are used only for the corresponding control and same-panel comparisons. The three BFCL criteria are reported in parallel. "
             "This is an exploratory analysis carried out after model scores and unadjusted associations were visible.", "",
             "Method: X, Y, and the control variables are converted to average ranks, and Kendall τ-b is computed on the residuals of linear regressions. "
             "The accurate name is rank-residual association; it is not standard partial Kendall, a conditional-independence test, or a cross-validated predictive gain. "
             "10,000 paired model bootstrap draws, seed=42; ranks and residual fits are redone in every draw. The 95% percentile intervals are not adjusted for multiple comparisons.", "",
             "## Overview for core block A", "",
             "| Panel | Adjustment variable | Overall (95% interval) | Multi-Turn (95% interval) | Agentic (95% interval) |",
             "| --- | --- | --- | --- | --- |"]
    for panel, adjustment, _ in SPECS:
        entries = [r for r in associations if r["panel"] == panel and r["adjustment"] == adjustment and r["block"] == "A"]
        lines.append(f"| {panel} | {labels[adjustment]} | " +
                     " | ".join(f"{fmt(r['tau_b'])} {interval(r)}" for r in entries) + " |")
    lines += ["", "## All A/C/D statistics", "",
              "Unadjusted rows are raw τ-b and adjusted rows are rank-residual τ-b. Intervals and point statistics both retain ties.", "",
              "| Panel | Adjustment | BFCL | Task | τ-b | 95% interval | Valid/undefined draws |",
              "| --- | --- | --- | --- | ---: | --- | --- |"]
    for r in associations:
        lines.append(f"| {r['panel']} | {labels[r['adjustment']]} | {r['criterion']} | {r['block']} | "
                     f"{fmt(r['tau_b'])} | {interval(r)} | {r['boot_valid']}/{r['boot_undefined']} |")
    lines += ["", "## Paired differences", "",
              "A minus C or D, computed on the same model sample under the same adjustment; an interval that spans 0 does not prove equality.", "",
              "| Panel | Adjustment | BFCL | Comparison | Δτ-b | 95% interval | Valid/undefined draws |",
              "| --- | --- | --- | --- | ---: | --- | --- |"]
    for r in differences:
        lines.append(f"| {r['panel']} | {labels[r['adjustment']]} | {r['criterion']} | {r['comparison']} | "
                     f"{fmt(r['delta_tau_b'])} | {interval(r)} | {r['boot_valid']}/{r['boot_undefined']} |")
    lines += ["", "## Panels, sources, and limits of interpretation", "",
              "The 12 models with available parameter counts: " + ", ".join(meta["panels"]["known_params12"]) + ".", "",
              "The 6 models with unknown parameter counts: " + ", ".join(meta["params_unavailable"]) + ". Parameter counts are not imputed or guessed from names.", "",
              "Comparisons before and after a control must use the same panel; the change caused by going from 18 to 12 models must not be read as an effect of the parameter control. "
              "Intervals include model resampling only, not item sampling, and cannot remove provider dependence, model selection, or format/truncation confounding. "
              "There is no residual permutation p-value, no automatic decision on manuscript conclusions, and no new model request.", "",
              "The input is BFCL_INPUTS_20261004.csv in the parent directory; full precision, the actual model sets, and software versions are in analysis.json. "
              "Reproduce: `python3 analysis/run_bfcl_controls.py`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "A": [r for r in associations if r["block"] == "A"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
