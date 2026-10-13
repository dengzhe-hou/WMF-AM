#!/usr/bin/env python3
"""Independent verification of the two-rule E/F comparisons.

Aggregates existing item scores directly and computes only the 14 specified
associations. Every bootstrap resample explicitly repeats model rows; SciPy
Kendall tau-b is used without importing the main analysis kernel.
"""

import csv
import itertools
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
SOURCE = ROOT / "data/scored/scored_items.csv"
INPUT = PANEL / "BFCL_INPUTS_20261004.csv"
BANK = ROOT / "data/item_bank/item_bank.jsonl"
OUT = PANEL / "ef_comparison_20261004"
REFERENCE = OUT / "analysis.json"
RULES = {"submitted": "correct_submitted", "candidate": "correct_strict"}
CRITERIA = ("overall", "multi_turn", "agentic")
F_DOMAINS = ("schedule", "permissions", "inventory")
N_BOOT, SEED = 10_000, 42


def tau(x, y):
    return float(kendalltau(x, y, variant="b").statistic)


def finite(value):
    return float(value) if np.isfinite(value) else None


def summarize(values):
    valid = values[np.isfinite(values)]
    interval = np.percentile(valid, [2.5, 97.5]) if len(valid) else [np.nan, np.nan]
    return {"ci95_low": finite(interval[0]), "ci95_high": finite(interval[1]),
            "boot_valid": int(len(valid)), "boot_undefined": int(len(values) - len(valid))}


def count(rows, column):
    return {"n": len(rows), "correct": sum(int(row[column]) for row in rows),
            "accuracy": sum(int(row[column]) for row in rows) / len(rows),
            "truncated": sum(int(row["truncated"]) for row in rows)}


def aggregate_inputs():
    panel = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    models = [row["run_label"] for row in panel]
    assert len(models) == len(set(models)) == 18
    bank = {item["item_id"]: item for item in
            (json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines() if line)
            if item["block"] in {"A", "E", "F"}}
    assert len(bank) == 240
    rows = [row for row in csv.DictReader(SOURCE.open(encoding="utf-8", newline=""))
            if row["block"] in {"A", "E", "F"}]
    index = {(row["model"], row["item_id"]): row for row in rows}
    assert len(index) == len(rows) == 18 * 240
    assert set(index) == {(model, item_id) for model in models for item_id in bank}
    by_block, by_domain, by_k, by_domain_k = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        item = bank[row["item_id"]]
        assert row["status"] == "scored" and int(row["synthetic"]) == 0
        assert row["block"] == item["block"] and int(row["k"]) == item["k"]
        model, block, k = row["model"], item["block"], item["k"]
        by_block[(model, block)].append(row)
        if block in {"E", "F"}:
            domain = item["domain"]
            assert row["domain"] == domain
            by_domain[(model, block, domain)].append(row)
            by_k[(model, block, k)].append(row)
            by_domain_k[(model, block, domain, k)].append(row)
    assert all(len(group) == (60 if block == "A" else 90) for (_, block), group in by_block.items())
    assert all(len(group) == 30 for group in by_domain.values())
    assert all(len(group) == 30 for group in by_k.values())
    assert all(len(group) == 10 for group in by_domain_k.values())

    tables = {"by_model": [], "by_domain": [], "by_k": [], "by_domain_k": []}
    for rule, column in RULES.items():
        for model in models:
            for block in ("E", "F"):
                tables["by_model"].append({"rule": rule, "model": model, "block": block,
                                           **count(by_block[(model, block)], column)})
                domains = ("color", "location", "status") if block == "E" else ("permissions", "schedule", "inventory")
                for domain in domains:
                    tables["by_domain"].append({"rule": rule, "model": model, "block": block, "domain": domain,
                                                **count(by_domain[(model, block, domain)], column)})
                for k in (3, 5, 7):
                    tables["by_k"].append({"rule": rule, "model": model, "block": block, "k": k,
                                           **count(by_k[(model, block, k)], column)})
                for domain, k in itertools.product(domains, (3, 5, 7)):
                    tables["by_domain_k"].append({"rule": rule, "model": model, "block": block, "domain": domain, "k": k,
                                                 **count(by_domain_k[(model, block, domain, k)], column)})

    a = np.array([sum(int(row["correct_strict"]) for row in by_block[(model, "A")]) for model in models])
    bfcl = {criterion: np.array([float(row["bfcl_" + criterion + "_1216"]) for row in panel]) for criterion in CRITERIA}
    vectors = {}
    for rule, column in RULES.items():
        f = np.array([sum(int(row[column]) for row in by_block[(model, "F")]) for model in models])
        vectors[(rule, "F_vs_A")] = (f, a)
        for criterion in CRITERIA:
            vectors[(rule, "F_vs_bfcl_" + criterion)] = (f, bfcl[criterion])
        for domain in F_DOMAINS:
            domain_scores = np.array([sum(int(row[column]) for row in by_domain[(model, "F", domain)]) for model in models])
            vectors[(rule, "F_" + domain + "_vs_A")] = (domain_scores, a)
    assert len(vectors) == 14
    return models, tables, vectors


def compare_associations(reference, expected, fields):
    actual = {(row["rule"], row["comparison"]): row for row in reference["associations"]}
    expected_keys = {(row["rule"], row["comparison"]) for row in expected}
    assert len(actual) == len(reference["associations"]) == 14 and set(actual) == expected_keys
    discrepancies, maximum = [], 0.0
    for row in expected:
        key = (row["rule"], row["comparison"])
        for field in fields:
            first, second = row[field], actual[key][field]
            if first is None or second is None:
                equal = first is second
            elif field.startswith("boot_") or field == "n_models":
                equal = first == second
            else:
                difference = abs(first - second)
                maximum = max(maximum, difference)
                equal = difference <= 1e-12
            if not equal:
                discrepancies.append({"key": key, "field": field, "independent": first, "main": second})
    return {"passed": not discrepancies, "values_compared": len(expected) * len(fields),
            "maximum_absolute_numeric_difference": maximum, "discrepancies": discrepancies}


def compare_groups(expected, actual, key_fields):
    def key(row):
        return tuple(str(row[field]) for field in key_fields)
    observed = {key(row): row for row in actual}
    assert len(observed) == len(actual) == len(expected)
    assert set(observed) == {key(row) for row in expected}
    differences, maximum = [], 0.0
    checked = 0
    for row in expected:
        match = observed[key(row)]
        for field, value in row.items():
            found = match[field]
            checked += 1
            if isinstance(value, str):
                equal = value == found
            else:
                delta = abs(float(found) - value)
                maximum = max(maximum, delta)
                equal = delta == 0 if isinstance(value, int) else delta <= 1e-12
            if not equal:
                differences.append({"key": key(row), "field": field, "independent": value, "main": found})
    return {"passed": not differences, "rows_compared": len(expected), "values_compared": checked,
            "maximum_absolute_numeric_difference": maximum, "discrepancies": differences}


def finish_comparison(report):
    if REFERENCE.exists():
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        report["point_check"] = compare_associations(reference, report["associations"], ("n_models", "tau_b"))
        report["bootstrap_check"] = compare_associations(reference, report["associations"], ("ci95_low", "ci95_high", "boot_valid", "boot_undefined"))
        schema = {"accuracy_by_model": ("by_model", ("rule", "model", "block")),
                  "accuracy_by_domain": ("by_domain", ("rule", "model", "block", "domain")),
                  "accuracy_by_k": ("by_k", ("rule", "model", "block", "k")),
                  "accuracy_cells": ("by_domain_k", ("rule", "model", "block", "domain", "k"))}
        checks = {}
        pooled_groups = defaultdict(list)
        for stem, (independent_key, fields) in schema.items():
            expected = report["group_counts"][independent_key]
            checks[stem + ".json"] = compare_groups(expected, reference[stem], fields)
            actual_csv = list(csv.DictReader((OUT / (stem + ".csv")).open(encoding="utf-8", newline="")))
            checks[stem + ".csv"] = compare_groups(expected, actual_csv, fields)
            for row in expected:
                pooled_groups[(row["rule"], row["block"], row.get("domain", "ALL"), row.get("k", "ALL"))].append(row)
        expected_pooled = []
        for (rule, block, domain, k), rows in pooled_groups.items():
            n, correct, truncated = (sum(row[field] for row in rows) for field in ("n", "correct", "truncated"))
            expected_pooled.append({"rule": rule, "block": block, "domain": domain, "k": k,
                                    "n": n, "correct": correct, "accuracy": correct / n, "truncated": truncated})
        pooled_keys = ("rule", "block", "domain", "k")
        checks["pooled_accuracy.json"] = compare_groups(expected_pooled, reference["pooled_accuracy"], pooled_keys)
        pooled_csv = list(csv.DictReader((OUT / "pooled_accuracy.csv").open(encoding="utf-8", newline="")))
        checks["pooled_accuracy.csv"] = compare_groups(expected_pooled, pooled_csv, pooled_keys)
        report["group_counts"]["pooled"] = expected_pooled
        report["group_checks"] = checks
        report["passed"] = (report["point_check"]["passed"] and report["bootstrap_check"]["passed"]
                            and all(check["passed"] for check in checks.values()))
        report.pop("comparison_status", None)
    else:
        report["passed"] = None
        report["comparison_status"] = "Independent computation complete; main output pending"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEPENDENT_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Independent check of the two E/F scoring rules", "",
             "Aggregated independently from the frozen bank and the item-level scores: the submitted rule reads correct_submitted and the candidate rule reads correct_strict; A always uses correct_strict under both. All 18 models are retained, in the run_label order of BFCL_INPUTS.",
             "Only the specified 7 associations per rule (14 in total) are computed, with SciPy Kendall τ-b, explicit repeated model rows, shared RandomState(42) indices, and 10,000 bootstrap draws each. The main analysis kernel is not called.", "",
             f"Association check passed: {report['passed']}."]
    for name in ("point_check", "bootstrap_check"):
        if name in report:
            check = report[name]
            lines += ["", f"{name}: passed={check['passed']}; values checked={check['values_compared']}; maximum numeric difference={check['maximum_absolute_numeric_difference']:.17g}."]
            lines += ["- " + json.dumps(item, ensure_ascii=False) for item in check["discrepancies"]]
    lines += ["", "E/F denominators under each rule: 90 per block, 30 per domain, 30 per block and K, 10 per domain × K. The independently grouped values are stored in group_counts in the JSON file.",
              "No E-versus-BFCL association, test of the difference between the two rules, p-value, additional control, or family statistic is computed; no model is filtered and no final scoring rule is designated."]
    if "group_checks" in report:
        lines += ["", "| Grouped artifact | Passed | Rows | Maximum numeric difference |", "| --- | --- | ---: | ---: |"]
        for name, check in report["group_checks"].items():
            lines.append(f"| {name} | {check['passed']} | {check['rows_compared']} | {check['maximum_absolute_numeric_difference']} |")
            lines += ["- " + json.dumps(item, ensure_ascii=False) for item in check["discrepancies"]]
    (OUT / "INDEPENDENT_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "INDEPENDENT_CHECK.json"), "passed": report["passed"],
                      "point_check": report.get("point_check"), "bootstrap_check": report.get("bootstrap_check")}, ensure_ascii=False, indent=2))


def main():
    start = time.monotonic()
    models, tables, vectors = aggregate_inputs()
    entries = list(vectors)
    associations = [{"rule": rule, "comparison": comparison, "n_models": len(models),
                     "tau_b": finite(tau(*vectors[(rule, comparison)]))} for rule, comparison in entries]
    report = {"meta": {"source_scored": str(SOURCE), "source_bank": str(BANK), "source_panel": str(INPUT),
                       "models_in_input_order": models, "rule_columns": RULES, "A_column": "correct_strict",
                       "statistic": "scipy.stats.kendalltau(..., variant='b').statistic",
                       "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                                     "draws": "choice(18, size=18, replace=True) once per sample, shared across both rules and all seven comparisons",
                                     "interval": "numpy.percentile [2.5,97.5] on defined draws"},
                       "no_main_kernel_import": True, "comparison_tolerance": 1e-12,
                       "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}},
              "group_counts": tables, "associations": associations}
    if REFERENCE.exists():
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        print("POINT_CHECK " + json.dumps(compare_associations(reference, associations, ("n_models", "tau_b")), ensure_ascii=False), flush=True)
    else:
        print("POINTS_READY: 14 independent associations; main output pending", flush=True)
    rng = np.random.RandomState(SEED)
    draws = np.full((N_BOOT, len(entries)), np.nan)
    for i in range(N_BOOT):
        indices = rng.choice(len(models), size=len(models), replace=True)
        for j, key in enumerate(entries):
            x, y = vectors[key]
            draws[i, j] = tau(x[indices], y[indices])
        if (i + 1) % 2500 == 0:
            print(f"BOOTSTRAP {i + 1}/10000 shared samples, {time.monotonic() - start:.1f}s", flush=True)
    for j, row in enumerate(associations):
        row.update(summarize(draws[:, j]))
    report["meta"]["elapsed_seconds"] = time.monotonic() - start
    finish_comparison(report)


if __name__ == "__main__":
    main()
