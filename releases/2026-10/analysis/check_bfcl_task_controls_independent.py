#!/usr/bin/env python3
"""Independent check of A--BFCL rank residuals controlling C, D, or both.

Uses explicit repeated bootstrap rows, SciPy average ranks, NumPy least squares
with an intercept, and unordered-pair tau-b counts. No main kernel is imported.
"""

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/results/BFCL_INPUTS_20261004.csv"
OUT = ROOT / "analysis/results/bfcl_task_controls_20261004"
REFERENCE = OUT / "analysis.json"
CRITERIA = ("overall", "multi_turn", "agentic")
SPECS = (("C", ("C",)), ("D", ("D",)), ("C_and_D", ("C", "D")))
N_BOOT, SEED = 10_000, 42
TIE_TOL, COMPARE_TOL = 1e-9, 1e-12


def statistic(values, controls):
    ranks = rankdata(values, method="average", axis=0)
    control_ranks = rankdata(controls, method="average", axis=0)
    design = np.column_stack([np.ones(len(values)), control_ranks])
    residual = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
    left, right = np.triu_indices(len(values), k=1)
    differences = residual[left] - residual[right]
    signs = np.sign(differences).astype(np.int32)
    signs[np.abs(differences) <= TIE_TOL] = 0
    numerator = signs[:, 0] @ signs[:, 1:]
    denominator = np.sqrt(np.count_nonzero(signs[:, 0]) * np.count_nonzero(signs[:, 1:], axis=0))
    result = np.full(3, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def finite(value):
    return float(value) if np.isfinite(value) else None


def summary(values):
    defined = values[np.isfinite(values)]
    interval = np.percentile(defined, [2.5, 97.5]) if len(defined) else [np.nan, np.nan]
    return {"ci95_low": finite(interval[0]), "ci95_high": finite(interval[1]),
            "boot_valid": int(len(defined)), "boot_undefined": int(len(values) - len(defined))}


def compare(reference, independent, fields):
    keys = ("adjustment", "criterion", "block")
    actual = {tuple(row[key] for key in keys): row for row in reference["associations"]}
    expected_keys = {tuple(row[key] for key in keys) for row in independent}
    assert len(actual) == len(reference["associations"]) == 9
    assert set(actual) == expected_keys
    differences, maximum = [], 0.0
    for expected in independent:
        key = tuple(expected[column] for column in keys)
        observed = actual[key]
        for field in fields:
            first, second = expected[field], observed[field]
            if first is None or second is None:
                equal = first is second
            elif field.startswith("boot_") or field == "n_models":
                equal = first == second
            else:
                difference = abs(first - second)
                maximum = max(maximum, difference)
                equal = difference <= COMPARE_TOL
            if not equal:
                differences.append({"key": key, "field": field, "independent": first, "main": second})
    return {"passed": not differences, "checked_values": len(independent) * len(fields),
            "maximum_absolute_numeric_difference": maximum, "discrepancies": differences}


def main():
    start = time.monotonic()
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8", newline="")))
    assert len(rows) == len({row["model"] for row in rows}) == 18
    values = np.array([[int(row["A_correct"])] +
                       [float(row["bfcl_" + criterion + "_1216"]) for criterion in CRITERIA]
                      for row in rows])
    associations, prepared = [], []
    for adjustment, columns in SPECS:
        controls = np.array([[int(row[column + "_correct"]) for column in columns] for row in rows])
        points = statistic(values, controls)
        prepared.append((adjustment, controls))
        for ci, criterion in enumerate(CRITERIA):
            associations.append({"adjustment": adjustment, "criterion": criterion,
                                 "block": "A", "n_models": 18, "tau_b": finite(points[ci])})
    report = {
        "meta": {"source": str(SOURCE), "reference": str(REFERENCE),
                 "models_in_input_order": [row["model"] for row in rows],
                 "method": "Explicit repeated rows; scipy.stats.rankdata average ranks; numpy.linalg.lstsq with intercept; independent unordered-pair tau-b counts",
                 "no_main_kernel_import": True,
                 "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                               "draws": "choice(18, size=18, replace=True), resetting RNG for each adjustment",
                               "pairing": "Identical index stream for all criteria and adjustments",
                               "refit": "Recompute average ranks and refit residuals within each repeated-row sample",
                               "interval": "2.5th and 97.5th numpy.percentile of defined draws"},
                 "residual_tie_tolerance": TIE_TOL, "comparison_tolerance": COMPARE_TOL,
                 "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}},
        "associations": associations,
    }
    if REFERENCE.exists():
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        report["point_check"] = compare(reference, associations, ("tau_b", "n_models"))
        print("POINT_CHECK " + json.dumps(report["point_check"], ensure_ascii=False), flush=True)
    else:
        print("POINTS_READY: 9 points; main output pending", flush=True)
    lookup = {(row["adjustment"], row["criterion"]): row for row in associations}
    for adjustment, controls in prepared:
        rng = np.random.RandomState(SEED)
        draws = np.full((N_BOOT, 3), np.nan)
        for i in range(N_BOOT):
            indices = rng.choice(len(rows), size=len(rows), replace=True)
            draws[i] = statistic(values[indices], controls[indices])
        for ci, criterion in enumerate(CRITERIA):
            lookup[(adjustment, criterion)].update(summary(draws[:, ci]))
        print(f"BOOTSTRAP_DONE {adjustment}: 10000 draws; cumulative {time.monotonic() - start:.1f}s", flush=True)
    if REFERENCE.exists():
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        report["point_check"] = compare(reference, associations, ("tau_b", "n_models"))
        report["bootstrap_check"] = compare(reference, associations, ("ci95_low", "ci95_high", "boot_valid", "boot_undefined"))
        report["passed"] = report["point_check"]["passed"] and report["bootstrap_check"]["passed"]
    else:
        report["passed"] = None
        report["comparison_status"] = "Main output not yet available; independent values complete"
    report["meta"]["elapsed_seconds"] = time.monotonic() - start
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEPENDENT_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Independent check of the task controls for A versus BFCL", "",
             "Only the three specified control configurations (C, D, C+D) and the three BFCL columns are checked, 9 items in total. The original 18 models and the CSV order are retained.",
             "Explicit repeated model rows, SciPy average ranks, NumPy least squares with an intercept, and independent pair-counting τ-b are used; the main weighted kernel is not called.",
             "Each configuration recomputes all 10,000 shared draws; ranks and fits are redone in every draw; residual differences with absolute value ≤1e-9 count as ties.", "",
             f"Check passed: {report['passed']}."]
    for name in ("point_check", "bootstrap_check"):
        if name in report:
            check = report[name]
            lines += ["", f"{name}: passed={check['passed']}; values checked={check['checked_values']}; maximum numeric difference={check['maximum_absolute_numeric_difference']:.17g}."]
            lines += ["- " + json.dumps(item, ensure_ascii=False) for item in check["discrepancies"]]
    lines += ["", "| Control | BFCL | Independent τ-b | Independent 95% interval | Valid / undefined |",
              "| --- | --- | ---: | --- | --- |"]
    for row in associations:
        lines.append(f"| {row['adjustment']} | {row['criterion']} | {row['tau_b']} | [{row['ci95_low']}, {row['ci95_high']}] | {row['boot_valid']} / {row['boot_undefined']} |")
    lines += ["", "The check covers the numerical implementation only; it adds no p-value, model filter, other control combination, family analysis, or manuscript conclusion. Full-precision records are in INDEPENDENT_CHECK.json."]
    (OUT / "INDEPENDENT_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "INDEPENDENT_CHECK.json"), "passed": report["passed"],
                      "point_check": report.get("point_check"), "bootstrap_check": report.get("bootstrap_check")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
