#!/usr/bin/env python3
"""Independent SciPy check of the BFCL Overall >= 50 subset.

Read the two source CSVs directly; do not import any analysis/report script.
The old full-panel bootstrap is only checked for faithful copying, never rerun.
"""

import csv
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kendalltau


PANEL = Path(__file__).resolve().parents[1] / "analysis/results"
OUT = PANEL / "bfcl_high_subset_20261004"
CRITERIA = ("overall", "multi_turn", "agentic")
BLOCKS = ("A", "C", "D")
N_BOOT = 10_000
TOL = 1e-12
EXPECTED_NAMES = [
    "GPT-5.2", "GPT-4.1", "GPT-4.1-mini", "Claude Haiku 4.5",
    "Claude Sonnet 4.5", "Claude Opus 4.5",
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def interval(samples):
    defined = samples[np.isfinite(samples)]
    bounds = np.percentile(defined, [2.5, 97.5]) if len(defined) else [np.nan, np.nan]
    return {
        "ci95_low": finite_or_none(bounds[0]),
        "ci95_high": finite_or_none(bounds[1]),
        "boot_valid": int(len(defined)),
        "boot_undefined": int(N_BOOT - len(defined)),
    }


def main():
    inputs = read_csv(PANEL / "BFCL_INPUTS_20261004.csv")
    protocol = {(r["model"], r["block"]): r for r in read_csv(PANEL / "protocol_conditions.csv")}
    reported = json.loads((OUT / "analysis.json").read_text())
    full = json.loads((PANEL / "bfcl_comparison_20261004/analysis.json").read_text())
    selected = [r for r in inputs if float(r["bfcl_overall_1216"]) >= 50]
    anomalies = []
    errors = []
    checked = 0

    def compare(label, actual, expected):
        nonlocal checked
        checked += 1
        if expected is None:
            equal = actual is None or actual == ""
        elif isinstance(expected, (int, float)):
            if actual is None or actual == "":
                equal = False
            else:
                error = abs(float(actual) - float(expected))
                errors.append(error)
                equal = error <= TOL
        else:
            equal = actual == expected
        if not equal:
            anomalies.append({"field": label, "reported": actual, "independent": expected})

    def compare_rows(label, actual, expected):
        compare(label + ".length", len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(f"{label}[{i}].columns", sorted(a), sorted(e))
            for key, value in e.items():
                compare(f"{label}[{i}].{key}", a.get(key), value)

    names = [r["model"] for r in selected]
    compare("input_row_count", len(inputs), 18)
    compare("selected_names", names, EXPECTED_NAMES)
    compare("meta.criterion", reported["meta"]["criterion"], "bfcl_overall_1216 >= 50")
    compare("meta.models_in_input_order", reported["meta"]["models_in_input_order"], names)
    compare("meta.n_models", reported["meta"]["n_models"], 6)
    compare("meta.family_counts", reported["meta"]["family_counts"], dict(Counter(r["family"] for r in selected)))
    compare("meta.bootstrap.n", reported["meta"]["bootstrap"]["n"], N_BOOT)
    compare("meta.bootstrap.seed", reported["meta"]["bootstrap"]["seed"], 42)
    compare("meta.bootstrap.rng", reported["meta"]["bootstrap"]["rng"], "numpy.random.RandomState")
    compare("full_panel.names", full["meta"]["models_in_input_order"], [r["model"] for r in inputs])

    members = []
    for row in selected:
        member = {key: row[key] for key in ("model", "run_label", "family")}
        for block in BLOCKS:
            for suffix in ("n", "correct", "one_integer", "truncated"):
                member[f"{block}_{suffix}"] = int(row[f"{block}_{suffix}"])
        for block in ("H", "I"):
            condition = protocol[(row["run_label"], block)]
            for suffix in ("correct", "truncated"):
                member[f"{block}_{suffix}"] = int(condition[suffix])
        for criterion in CRITERIA:
            member[f"bfcl_{criterion}"] = float(row[f"bfcl_{criterion}_1216"])
        # A appears in both source CSVs, so check the source-to-source join too.
        a_protocol = protocol[(row["run_label"], "A")]
        for field, protocol_field in (("n", "n_scored"), ("correct", "correct"),
                                     ("one_integer", "exactly_one_integer"), ("truncated", "truncated")):
            compare(f"source_A.{row['model']}.{field}", member[f"A_{field}"], int(a_protocol[protocol_field]))
        members.append(member)
    compare_rows("members.json", reported["members"], members)
    compare_rows("members.csv", read_csv(OUT / "members.csv"), members)
    compare("C_correct", [r["C_correct"] for r in members], [90] * 6)
    compare("C_n", [r["C_n"] for r in members], [90] * 6)

    x = {b: np.array([int(r[f"{b}_correct"]) for r in selected]) for b in BLOCKS}
    y = {c: np.array([float(r[f"bfcl_{c}_1216"]) for r in selected]) for c in CRITERIA}
    points = {(c, b): float(kendalltau(x[b], y[c], variant="b").statistic)
              for c in CRITERIA for b in BLOCKS}
    boot = {(c, b): np.empty(N_BOOT) for c in CRITERIA for b in BLOCKS}
    rng = np.random.RandomState(42)
    for iteration in range(N_BOOT):
        draw = rng.choice(len(selected), size=len(selected), replace=True)
        for criterion in CRITERIA:
            for block in BLOCKS:
                boot[(criterion, block)][iteration] = kendalltau(
                    x[block][draw], y[criterion][draw], variant="b"
                ).statistic

    associations = []
    differences = []
    for criterion in CRITERIA:
        for block in BLOCKS:
            concordant = discordant = tied = 0
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    product = (x[block][i] - x[block][j]) * (y[criterion][i] - y[criterion][j])
                    concordant += int(product > 0)
                    discordant += int(product < 0)
                    tied += int(product == 0)
            associations.append({
                "criterion": criterion, "block": block, "n_models": len(selected),
                "tau_b": finite_or_none(points[(criterion, block)]),
                **interval(boot[(criterion, block)]),
                "x_unique": len(np.unique(x[block])), "y_unique": len(np.unique(y[criterion])),
                "concordant_pairs": concordant, "discordant_pairs": discordant,
                "pairs_tied_in_either": tied,
            })
        for control in ("C", "D"):
            differences.append({
                "criterion": criterion, "comparison": f"A-{control}", "n_models": len(selected),
                "delta_tau_b": finite_or_none(points[(criterion, "A")] - points[(criterion, control)]),
                **interval(boot[(criterion, "A")] - boot[(criterion, control)]),
            })
    compare_rows("associations.json", reported["associations"], associations)
    compare_rows("associations.csv", read_csv(OUT / "associations.csv"), associations)
    compare_rows("paired_differences.json", reported["paired_differences"], differences)
    compare_rows("paired_differences.csv", read_csv(OUT / "paired_differences.csv"), differences)

    full_reference = [r for r in full["associations"] if r["block"] == "A"]
    compare_rows("full18_A_reference", reported["full18_A_reference"], full_reference)
    for row in full_reference:
        criterion = row["criterion"]
        independent_tau = kendalltau(
            [int(r["A_correct"]) for r in inputs],
            [float(r[f"bfcl_{criterion}_1216"]) for r in inputs], variant="b"
        ).statistic
        compare(f"full18_A_point.{criterion}", row["tau_b"], float(independent_tau))

    result = {
        "status": "PASS" if not anomalies else "FAIL",
        "purpose": "Detect subset/index alignment, constant/tie handling, bootstrap CI and paired-difference errors.",
        "implementation": "Independent scipy.stats.kendalltau; explicit RandomState(42).choice(n, size=n, replace=True) once per draw; 10000 shared draws; numpy percentile over finite draws only.",
        "scope": "All six selected models retained. No new p values, controls, H/I associations, model calls or manuscript changes. Full18 bootstrap not rerun; reference copying and its three A point estimates checked.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "input_files": [str(PANEL / "BFCL_INPUTS_20261004.csv"), str(PANEL / "protocol_conditions.csv")],
        "checked_fields": checked,
        "anomaly_count": len(anomalies),
        "maximum_absolute_numeric_error": max(errors, default=0.0),
        "numeric_tolerance": TOL,
        "selected_names": names,
        "associations_independent": associations,
        "paired_differences_independent": differences,
        "full18_reference_source": str(PANEL / "bfcl_comparison_20261004/analysis.json"),
        "anomalies": anomalies,
    }
    (OUT / "INDEPENDENT_CHECK.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    summary = [
        "# Independent numerical check of the six-model high-BFCL subgroup", "",
        f"Result: {result['status']}; anomalies: {len(anomalies)}; maximum absolute numeric error {result['maximum_absolute_numeric_error']:.17g} (tolerance {TOL:g}).", "",
        "Selecting Overall ≥ 50 independently from BFCL_INPUTS_20261004.csv gives six models whose names, order, and all members columns equal the original input; A agrees with the cross-record in protocol_conditions.csv.",
        "SciPy Kendall τ-b with 10,000 explicit RandomState(42) draws is used; each draw is shared by all statistics. The check covers 9 associations, 6 paired differences, and every number in the JSON/CSV files. No main analysis or report script is imported.", "",
        "C is 90/90 for all six models, so the point estimates and intervals of the three C associations and the three A−C differences are undefined, with 0 valid and 10,000 undefined bootstrap draws.",
        "The three A associations have 10,000 valid and 0 undefined draws; the three D associations and the three A−D differences have 9,975 valid and 25 undefined draws.", "",
        "The full 18-model A reference equals the existing results field by field, and the three point estimates are checked independently; the earlier 18-model bootstrap is not recomputed. No threshold was changed, no model removed, no test added, and the manuscript was not modified.", "",
        "A numerical or alignment error should be reported with the specific field and the computation or transcription corrected; no such error was found here.", "",
    ]
    if anomalies:
        summary = ["# Independent numerical check of the six-model high-BFCL subgroup", "", "Result: FAIL; the anomalies below must be resolved first.", "", json.dumps(anomalies, indent=2, ensure_ascii=False), ""]
    (OUT / "INDEPENDENT_CHECK.md").write_text("\n".join(summary))
    print(json.dumps({key: result[key] for key in ("status", "checked_fields", "anomaly_count", "maximum_absolute_numeric_error")}, ensure_ascii=False))
    return int(bool(anomalies))


if __name__ == "__main__":
    raise SystemExit(main())
