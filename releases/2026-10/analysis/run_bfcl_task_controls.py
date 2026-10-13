#!/usr/bin/env python3
"""Specified 2026-10-04: A versus three BFCL criteria, controlling C, D, or both.

Reproduce: python3 analysis/run_bfcl_task_controls.py
Uses the previously verified rank-residual and paired model-bootstrap kernels.
Does not execute analyze.main(), run models, or change any previous result.
"""

import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from analyze import clean, kendall, model_weights, signs_for, taub
from run_bfcl_comparison import CRITERIA, N_BOOT, SEED, fmt, interval, summary, write_csv

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
OUT = PANEL / "bfcl_task_controls_20261004"
SPECS = (("C", ("C",)), ("D", ("D",)), ("C_and_D", ("C", "D")))
LABELS = {"C": "single-step arithmetic C", "D": "cancellation control D", "C_and_D": "single-step arithmetic C + cancellation control D"}


def main():
    source = PANEL / "BFCL_INPUTS_20261004.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["model"] for row in rows]
    assert len(names) == len(set(names)) == 18
    for block, denominator in (("A", 60), ("C", 90), ("D", 100)):
        assert all(int(row[block + "_n"]) == denominator for row in rows)
    # Fixed denominators within each block make correct counts rank-equivalent
    # to accuracies, with no rounding-induced ties.
    scores = {block: np.array([int(row[block + "_correct"]) for row in rows])
              for block in ("A", "C", "D")}
    criteria = {criterion: np.array([float(row[f"bfcl_{criterion}_1216"]) for row in rows])
                for criterion in CRITERIA}
    baseline_source = PANEL / "bfcl_comparison_20261004/analysis.json"
    previous = json.loads(baseline_source.read_text(encoding="utf-8"))
    assert previous["meta"]["models_in_input_order"] == names
    baseline = [row for row in previous["associations"] if row["block"] == "A"]
    assert [row["criterion"] for row in baseline] == list(CRITERIA)
    weights = model_weights(len(rows), N_BOOT, SEED)
    associations = []
    for adjustment, blocks in SPECS:
        controls = [scores[block] for block in blocks]
        for criterion, y in criteria.items():
            point = clean(kendall(scores["A"], y, controls))
            bootstrap = taub(*signs_for(scores["A"], y, controls, weights), weights)
            associations.append({
                "adjustment": adjustment, "criterion": criterion, "block": "A",
                "n_models": len(rows), "tau_b": point, **summary(bootstrap),
            })
    metadata = {
        "status": "exploratory; specified after model scores, marginal associations and BFCL single-turn/parameter adjustments were visible",
        "source": str(source), "baseline_source": str(baseline_source),
        "baseline": "Previously computed A associations copied for context, not recomputed",
        "models_in_input_order": names, "n_models": len(rows),
        "criterion_snapshot": "2025-12-16", "criteria_parallel": list(CRITERIA),
        "target": "A (60 items), strict one-integer scoring",
        "control_specs": {label: list(blocks) for label, blocks in SPECS},
        "control_definitions": {"C": "Single-step arithmetic, 90 items", "D": "Yoked/cancellation control, 100 items"},
        "statistic": "Average-rank X,Y,Z; linear residuals of X and Y on Z; Kendall tau-b between residuals",
        "numerical_convention": "Existing analyze.py slope ridge 1e-12 and residual pairwise tie tolerance 1e-9",
        "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                      "pairing": "Same model draws for all criteria and all three control specifications",
                      "refit": "Re-rank and refit both residual regressions in each sample",
                      "ci": "Pointwise 95% percentile over defined draws, no multiplicity adjustment",
                      "undefined": "Retain counts; do not replace undefined tau by zero"},
        "limits": ["Not standard partial Kendall or a conditional-independence test",
                   "No held-out prediction or comparison of predictive gains",
                   "No causal working-memory mechanism inference",
                   "Model bootstrap only; no item resampling or correction for provider dependence",
                   "Control scores have finite-item measurement error and retain ceiling/ties",
                   "Prompt/compliance/truncation confounding remains",
                   "No model exclusion, p value, or automatic manuscript-claim decision"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "associations.csv", associations)
    (OUT / "analysis.json").write_text(json.dumps({"meta": metadata,
        "baseline_associations": baseline, "associations": associations}, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# A–BFCL association after controlling for simple tasks", "",
             "2026-10-04. All 18 models are retained; single-step arithmetic C, cancellation control D, and both are controlled in turn; "
             "the three columns Overall/Multi-Turn/Agentic are reported in parallel. This is an exploratory analysis carried out after existing results had been seen.", "",
             "A is the core block with 60 items, C is single-step arithmetic with 90 items, and D is the cancellation control with 100 items; all keep the fixed strict numeric scoring. "
             "X/Y/Z are converted to average ranks, X and Y are each regressed linearly on Z, and Kendall τ-b is computed between the residuals. "
             "Ranks and fits are redone in every model resample; the three configurations and three criteria share the same 10,000 draws, RandomState(42).", "",
             "## Associations and intervals", "",
             "The unadjusted row reuses the earlier full 18-model results as a reference; the other rows are rank-residual associations. Brackets give pointwise 95% model-bootstrap percentile intervals.", "",
             "| Control variable | Overall | Multi-Turn | Agentic |",
             "| --- | --- | --- | --- |",
             "| unadjusted (earlier result) | " + " | ".join(f"{fmt(row['tau_b'])} {interval(row)}" for row in baseline) + " |"]
    for adjustment, _ in SPECS:
        entries = [row for row in associations if row["adjustment"] == adjustment]
        lines.append(f"| {LABELS[adjustment]} | " + " | ".join(
            f"{fmt(row['tau_b'])} {interval(row)}" for row in entries) + " |")
    lines += ["", "## Valid resample counts", "",
              "τ is undefined when the residuals on one side are all tied; it is not filled with 0 and no model is removed, and the interval uses only the draws in which the statistic is defined.", "",
              "| Control variable | BFCL | Valid | Undefined |", "| --- | --- | ---: | ---: |"]
    for row in associations:
        lines.append(f"| {LABELS[row['adjustment']]} | {row['criterion']} | {row['boot_valid']} | {row['boot_undefined']} |")
    lines += ["", "## All models in the input", "",
              "The table gives the existing inputs, with scores and denominators reported as they are. BFCL keeps the 0–100 scale of the public table.", "",
              "| Model | A | C | D | Overall | Multi-Turn | Agentic |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model']} | " + " | ".join(
            f"{row[block + '_correct']}/{row[block + '_n']}" for block in ("A", "C", "D")) + " | " +
            " | ".join(row[f"bfcl_{criterion}_1216"] for criterion in CRITERIA) + " |")
    lines += ["", "## Limits of interpretation", "",
              "This analysis examines whether A and BFCL remain associated after adjusting for the linear rank relation with the specified controls. "
              "It is not a predictive gain on held-out models, and it cannot directly establish a working-memory mechanism or rule out every general-ability explanation. "
              "C/D come from a finite number of items with ties and ceiling effects, and the measurement error of the control variables themselves is not removed.", "",
              "The 18 models are a fixed convenience panel; same-provider dependence, item sampling, and format/truncation issues remain. "
              "Intervals are not adjusted for multiple comparisons; spanning 0 does not prove zero association, and not spanning 0 is no substitute for validating a mechanism or generalization. "
              "No new primary column is chosen, and no model filter, residual permutation p-value, combined single-turn/parameter control, or manuscript-conclusion threshold is added.", "",
              "Full precision, sources, and software versions are in analysis.json. Reproduce: `python3 analysis/run_bfcl_task_controls.py`. "
              "This analysis added no model call and changed no earlier result, E/F scoring, or manuscript text."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "associations": associations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
