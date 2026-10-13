#!/usr/bin/env python3
"""Independent numerical check of the BFCL rank-residual analysis.

Expands every bootstrap sample into repeated rows, uses SciPy average ranks and
NumPy least squares with an explicit intercept, and counts unordered pairs.
Does not import any analysis runner or the main numerical kernel.
"""

import csv
import json
import platform
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
SOURCE = PANEL / "BFCL_INPUTS_20261004.csv"
OUT = PANEL / "bfcl_controls_20261004"
REFERENCE = OUT / "analysis.json"
CRITERIA = ("overall", "multi_turn", "agentic")
BLOCKS = ("A", "C", "D")
SPECS = (
    ("all18", "none", ()),
    ("all18", "single_turn", ("single_turn",)),
    ("known_params12", "none", ()),
    ("known_params12", "single_turn", ("single_turn",)),
    ("known_params12", "log_params", ("log_params",)),
    ("known_params12", "single_turn_and_log_params", ("single_turn", "log_params")),
)
N_BOOT = 10_000
SEED = 42
TIE_TOL = 1e-9
COMPARE_TOL = 1e-12


def statistics(values, control_values):
    """Return a 3-criterion by 3-block matrix for explicit sampled rows."""
    if control_values.shape[1]:
        ranks = rankdata(values, method="average", axis=0)
        control_ranks = rankdata(control_values, method="average", axis=0)
        design = np.column_stack([np.ones(len(values)), control_ranks])
        coefficients = np.linalg.lstsq(design, ranks, rcond=None)[0]
        measured = ranks - design @ coefficients
        tolerance = TIE_TOL
    else:
        measured = values
        tolerance = 0.0
    left, right = np.triu_indices(len(values), k=1)
    differences = measured[left] - measured[right]
    signs = np.sign(differences).astype(np.int32)
    signs[np.abs(differences) <= tolerance] = 0
    block_signs, criterion_signs = signs[:, :3], signs[:, 3:]
    numerator = criterion_signs.T @ block_signs
    non_tied_blocks = np.count_nonzero(block_signs, axis=0)
    non_tied_criteria = np.count_nonzero(criterion_signs, axis=0)
    denominator = np.sqrt(non_tied_criteria[:, None] * non_tied_blocks[None, :])
    output = np.full((3, 3), np.nan)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def exact_rank(values):
    """Small exact rational rank, only to diagnose observed zero residuals."""
    matrix = [[Fraction(int(value)) for value in row] for row in values]
    pivot_row = 0
    for column in range(len(matrix[0])):
        pivot = next((row for row in range(pivot_row, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        divisor = matrix[pivot_row][column]
        matrix[pivot_row] = [entry / divisor for entry in matrix[pivot_row]]
        for row in range(pivot_row + 1, len(matrix)):
            factor = matrix[row][column]
            if factor:
                matrix[row] = [entry - factor * basis for entry, basis in zip(matrix[row], matrix[pivot_row])]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return pivot_row


def diagnose_undefined(values, controls, diagnostics):
    ranks = rankdata(values, method="average", axis=0)
    control_ranks = rankdata(controls, method="average", axis=0)
    design = np.column_stack([np.ones(len(values)), control_ranks])
    residual = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
    ranges = np.ptp(residual, axis=0)
    exact_design = np.column_stack([np.ones(len(values), dtype=int), (2 * control_ranks).astype(int)])
    design_rank = exact_rank(exact_design)
    diagnostics["affected_bootstrap_draws"] += 1
    for j, label in enumerate((*BLOCKS, *CRITERIA)):
        if ranges[j] > TIE_TOL:
            continue
        record = diagnostics["variables"][label]
        record["zero_residual_count"] += 1
        record["maximum_residual_range"] = max(record["maximum_residual_range"], float(ranges[j]))
        if np.ptp(values[:, j]) == 0:
            record["constant_sample_values"] += 1
        elif exact_rank(np.column_stack([exact_design, (2 * ranks[:, j]).astype(int)])) == design_rank:
            record["nonconstant_exact_linear_fit"] += 1
        else:
            record["near_zero_without_exact_fit"] += 1


def summary(values):
    valid = values[np.isfinite(values)]
    interval = np.percentile(valid, [2.5, 97.5]) if len(valid) else [np.nan, np.nan]
    return {"ci95_low": finite_or_none(interval[0]), "ci95_high": finite_or_none(interval[1]),
            "boot_valid": int(len(valid)), "boot_undefined": int(len(values) - len(valid))}


def compare(actual, expected, fields):
    discrepancies = []
    maximum = 0.0
    checked = 0
    for section, identity in (("associations", "block"), ("paired_differences", "comparison")):
        lookup = {(row["panel"], row["adjustment"], row["criterion"], row[identity]): row
                  for row in actual[section]}
        expected_keys = {(row["panel"], row["adjustment"], row["criterion"], row[identity])
                         for row in expected[section]}
        if len(lookup) != len(actual[section]) or set(lookup) != expected_keys:
            discrepancies.append({"section": section, "error": "row identity mismatch"})
            continue
        for row in expected[section]:
            key = (row["panel"], row["adjustment"], row["criterion"], row[identity])
            counterpart = lookup[key]
            for field in fields[section]:
                first, second = row[field], counterpart[field]
                checked += 1
                if first is None or second is None:
                    equal = first is second
                elif field.startswith("boot_") or field == "n_models":
                    equal = first == second
                else:
                    difference = abs(first - second)
                    maximum = max(maximum, difference)
                    equal = difference <= COMPARE_TOL
                if not equal:
                    discrepancies.append({"section": section, "key": key, "field": field,
                                          "independent": first, "main": second})
    return {"passed": not discrepancies, "checked_values": checked,
            "maximum_absolute_numeric_difference": maximum, "discrepancies": discrepancies}


def main():
    start = time.monotonic()
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8", newline="")))
    panels = {"all18": rows, "known_params12": [row for row in rows if row["params"]]}
    assert len(rows) == len({row["model"] for row in rows}) == 18
    assert len(panels["known_params12"]) == 12
    prepared = []
    associations, paired = [], []
    for panel, adjustment, columns in SPECS:
        subset = panels[panel]
        values = np.array([[int(row[block + "_correct"]) for block in BLOCKS] +
                           [float(row["bfcl_" + criterion + "_1216"]) for criterion in CRITERIA]
                          for row in subset])
        controls = np.empty((len(subset), len(columns)))
        for j, column in enumerate(columns):
            controls[:, j] = ([float(row["bfcl_single_turn_mean_1216"]) for row in subset]
                              if column == "single_turn" else
                              np.log10([int(row["params"]) for row in subset]))
        point = statistics(values, controls)
        prepared.append((panel, adjustment, values, controls, point))
        for ci, criterion in enumerate(CRITERIA):
            prefix = {"panel": panel, "adjustment": adjustment, "criterion": criterion,
                      "n_models": len(subset)}
            for bi, block in enumerate(BLOCKS):
                associations.append({**prefix, "block": block, "tau_b": finite_or_none(point[ci, bi])})
            for bi, block in ((1, "C"), (2, "D")):
                paired.append({**prefix, "comparison": "A-" + block,
                               "delta_tau_b": finite_or_none(point[ci, 0] - point[ci, bi])})
    assert len(associations) == 54 and len(paired) == 36
    report = {
        "meta": {"source": str(SOURCE), "reference": str(REFERENCE),
                 "method": "Explicit repeated model rows; scipy.stats.rankdata(method='average'); numpy.linalg.lstsq with intercept; independent unordered-pair tau-b counts",
                 "no_import_of_main_kernel": True,
                 "panels": {name: [row["model"] for row in subset] for name, subset in panels.items()},
                 "n_boot": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                 "pairing": "Reset RandomState(42) for each specification; all specifications on each panel share the identical index stream",
                 "refit": "Ranks and least-squares residuals recomputed on repeated rows in each resample",
                 "residual_tie_tolerance": TIE_TOL, "comparison_tolerance": COMPARE_TOL,
                 "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}},
        "associations": associations, "paired_differences": paired,
    }
    point_fields = {"associations": ("tau_b", "n_models"),
                    "paired_differences": ("delta_tau_b", "n_models")}
    if REFERENCE.exists():
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        report["point_check"] = compare(reference, report, point_fields)
        print("POINT_CHECK " + json.dumps(report["point_check"], ensure_ascii=False), flush=True)
    else:
        print("POINTS_READY 54 associations and 36 differences; main output not present yet", flush=True)

    association_index = {(r["panel"], r["adjustment"], r["criterion"], r["block"]): r for r in associations}
    paired_index = {(r["panel"], r["adjustment"], r["criterion"], r["comparison"]): r for r in paired}
    diagnostics = {
        "scope": "known_params12/single_turn_and_log_params undefined resamples only",
        "method": "For each zero-range least-squares residual, check whether target sample values are constant; otherwise compare exact rational ranks of [1, 2*control_ranks] and [1, 2*control_ranks, 2*target_rank]",
        "affected_bootstrap_draws": 0,
        "variables": {label: {"zero_residual_count": 0, "constant_sample_values": 0,
                              "nonconstant_exact_linear_fit": 0, "near_zero_without_exact_fit": 0,
                              "maximum_residual_range": 0.0} for label in (*BLOCKS, *CRITERIA)},
    }
    for panel, adjustment, values, controls, point in prepared:
        rng = np.random.RandomState(SEED)
        draws = np.empty((N_BOOT, 3, 3))
        for i in range(N_BOOT):
            indices = rng.choice(len(values), size=len(values), replace=True)
            draws[i] = statistics(values[indices], controls[indices])
            if adjustment == "single_turn_and_log_params" and not np.isfinite(draws[i]).all():
                diagnose_undefined(values[indices], controls[indices], diagnostics)
        for ci, criterion in enumerate(CRITERIA):
            for bi, block in enumerate(BLOCKS):
                association_index[(panel, adjustment, criterion, block)].update(summary(draws[:, ci, bi]))
            for bi, block in ((1, "C"), (2, "D")):
                paired_index[(panel, adjustment, criterion, "A-" + block)].update(summary(draws[:, ci, 0] - draws[:, ci, bi]))
        print(f"BOOTSTRAP_DONE {panel}/{adjustment}: 10000 paired draws; cumulative {time.monotonic() - start:.1f}s", flush=True)
    report["undefined_resample_diagnostic"] = diagnostics

    interval_fields = ("ci95_low", "ci95_high", "boot_valid", "boot_undefined")
    if REFERENCE.exists():
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        assert reference["meta"]["panels"] == report["meta"]["panels"]
        report["point_check"] = compare(reference, report, point_fields)
        report["bootstrap_check"] = compare(reference, report, {section: interval_fields for section in point_fields})
        report["passed"] = report["point_check"]["passed"] and report["bootstrap_check"]["passed"]
    else:
        report["passed"] = None
        report["comparison_status"] = "pending main analysis.json"
    report["meta"]["elapsed_seconds"] = time.monotonic() - start
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEPENDENT_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Independent check of the rank-residual analysis", "",
             "Every bootstrap sample is expanded into repeated model rows; SciPy average ranks, NumPy least squares with an intercept, and an independent pair-counting τ-b are used. The main analysis kernel is not called.", "",
             "All 6 specifications, 54 association point values, and 36 paired differences are checked, together with the intervals and valid counts over all 10,000 paired draws of each specification. Residual differences with absolute value ≤1e-9 count as ties.", "",
             f"Overall result: {report['passed']}."]
    for key in ("point_check", "bootstrap_check"):
        if key in report:
            check = report[key]
            lines += ["", f"{key}: passed={check['passed']}; values checked={check['checked_values']}; maximum numeric difference={check['maximum_absolute_numeric_difference']:.17g}."]
            for discrepancy in check["discrepancies"]:
                lines.append("- " + json.dumps(discrepancy, ensure_ascii=False))
    lines += ["", "The independent implementation uses ordinary least squares without the 1e-12 numerical ridge of the main kernel; the actual results test whether this implementation difference affects the fixed tie rule or the statistics.",
              "Full-precision independent results and item-by-item differences are in INDEPENDENT_CHECK.json. No new metric, model filter, p-value, or research conclusion is added here."]
    lines += ["", "Undefined draws in the 12-model two-control specification are additionally checked for exact rational linear dependence; this only explains the observed undefined counts.",
              "", "| Residual variable | Draws with all residuals tied | Original variable constant | Non-constant but ranks fitted exactly by the controls | Numerically near zero only, not an exact fit | Maximum residual range |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, record in diagnostics["variables"].items():
        lines.append(f"| {label} | {record['zero_residual_count']} | {record['constant_sample_values']} | {record['nonconstant_exact_linear_fit']} | {record['near_zero_without_exact_fit']} | {record['maximum_residual_range']:.3g} |")
    (OUT / "INDEPENDENT_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "INDEPENDENT_CHECK.json"), "passed": report["passed"],
                      "point_check": report.get("point_check"), "bootstrap_check": report.get("bootstrap_check")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
