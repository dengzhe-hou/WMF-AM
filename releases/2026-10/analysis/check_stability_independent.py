#!/usr/bin/env python3
"""Independently verify the seed, wording, and depth descriptions.

Reads frozen items and item-level scores directly. Kendall tau-b is computed
by counting unordered model pairs, without any main-analysis numerical kernel.
"""

import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data/item_bank/item_bank.jsonl"
SCORES = ROOT / "data/scored/scored_items.csv"
PANEL = ROOT / "analysis/results/BFCL_INPUTS_20261004.csv"
OUT = ROOT / "analysis/results/stability_20261004"
SEEDS = (2026, 100, 200, 300)
TEMPLATES = ("original", "formal", "casual", "minimal", "verbose")
DEPTHS = (3, 5, 7, 10, 15, 20, 30, 50, 75, 100)


def aggregate(rows):
    n = len(rows)
    correct = sum(int(row["correct_strict"]) for row in rows)
    compliant = sum(int(row["n_integers"]) == 1 for row in rows)
    bare = sum(row["format"] == "bare" for row in rows)
    truncated = sum(int(row["truncated"]) for row in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "one_integer": compliant, "compliance_rate": compliant / n,
            "bare": bare, "bare_rate": bare / n,
            "truncated": truncated, "truncation_rate": truncated / n}


def counted_tau(x, y):
    numerator = not_tied_x = not_tied_y = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        sx = int(x[i] > x[j]) - int(x[i] < x[j])
        sy = int(y[i] > y[j]) - int(y[i] < y[j])
        numerator += sx * sy
        not_tied_x += sx != 0
        not_tied_y += sy != 0
    denominator = math.sqrt(not_tied_x * not_tied_y)
    return numerator / denominator if denominator else None


def build_expected():
    bank = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines() if line]
    items = {row["item_id"]: row for row in bank if row["block"] in {"A", "B", "G"}}
    assert len(items) == 350
    panel = list(csv.DictReader(PANEL.open(encoding="utf-8", newline="")))
    models = [row["run_label"] for row in panel]
    assert len(models) == len(set(models)) == 18
    score_rows = [row for row in csv.DictReader(SCORES.open(encoding="utf-8", newline=""))
                  if row["block"] in {"A", "B", "G"}]
    assert len(score_rows) == 18 * 350
    keyed = {(row["model"], row["item_id"]): row for row in score_rows}
    assert len(keyed) == len(score_rows)
    assert set(keyed) == {(model, item_id) for model in models for item_id in items}
    for row in score_rows:
        item = items[row["item_id"]]
        assert row["status"] == "scored" and int(row["synthetic"]) == 0
        assert row["block"] == item["block"]
        assert int(row["k"]) == item["k"] and int(row["seed"]) == item["seed"]
        assert json.loads(row["gold"]) == item["gold"]
        assert row["template"] == item.get("template", "")

    # G uses its own 30 base tasks. Match all five versions by actual bank
    # metadata, not by assuming A is the original wording of G.
    bases = defaultdict(list)
    for item in items.values():
        if item["block"] == "G":
            bases[(item["k"], item["seed"], item["probe_idx"])].append(item)
    assert set(bases) == {(k, 42, probe) for k in (3, 5, 7) for probe in range(10)}
    for variants in bases.values():
        assert len(variants) == 5 and {item["template"] for item in variants} == set(TEMPLATES)
        assert len({item["gold"] for item in variants}) == 1
        for field in ("query_entity", "initial_value", "n_relevant_ops"):
            assert len({item[field] for item in variants}) == 1
    for seed in SEEDS:
        group = [item for item in items.values() if item["block"] == "A" and item["seed"] == seed]
        assert len(group) == 15
        assert {(item["k"], item["probe_idx_original"]) for item in group} == {(k, probe) for k in (3, 5, 7) for probe in range(5)}
    for k in DEPTHS:
        group = [item for item in items.values() if item["block"] in {"A", "B"} and item["k"] == k]
        assert len(group) == 20
        assert {(item["seed"], item["probe_idx_original"]) for item in group} == {(seed, probe) for seed in SEEDS for probe in range(5)}

    by_seed, by_template, by_depth = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in score_rows:
        model = row["model"]
        item = items[row["item_id"]]
        if item["block"] == "A":
            by_seed[(model, item["seed"])].append(row)
        if item["block"] == "G":
            by_template[(model, item["template"])].append(row)
        if item["block"] in {"A", "B"}:
            by_depth[(model, item["k"])].append(row)
    seed_rows = [{"model": model, "seed": seed, **aggregate(by_seed[(model, seed)])}
                 for model in models for seed in SEEDS]
    prompt_rows = [{"model": model, "template": template, **aggregate(by_template[(model, template)])}
                   for model in models for template in TEMPLATES]
    assert all(row["n"] == 15 for row in seed_rows)
    assert all(row["n"] == 30 for row in prompt_rows)
    seed_scores = {(row["model"], row["seed"]): row["accuracy"] for row in seed_rows}
    prompt_scores = {(row["model"], row["template"]): row["accuracy"] for row in prompt_rows}
    spread = []
    for model in models:
        values = [seed_scores[(model, seed)] for seed in SEEDS]
        spread.append({"model": model, "mean_accuracy": statistics.mean(values),
                       "sd_accuracy": statistics.stdev(values), "min_accuracy": min(values),
                       "max_accuracy": max(values), "range_accuracy": max(values) - min(values)})
    seed_pairs = [{"seed_1": left, "seed_2": right, "n_models": 18,
                   "tau_b": counted_tau([seed_scores[(model, left)] for model in models],
                                        [seed_scores[(model, right)] for model in models])}
                  for left, right in itertools.combinations(SEEDS, 2)]
    prompt_pairs = [{"template_1": left, "template_2": right, "n_models": 18,
                     "tau_b": counted_tau([prompt_scores[(model, left)] for model in models],
                                          [prompt_scores[(model, right)] for model in models])}
                    for left, right in itertools.combinations(TEMPLATES, 2)]
    depth_rows = []
    for model in models:
        for k in DEPTHS:
            group = by_depth[(model, k)]
            total = aggregate(group)
            assert total["n"] == 20
            depth_rows.append({"model": model, "block": "A" if k in (3, 5, 7) else "B", "k": k,
                               "n_expected": 20, "n_scored": total["n"], "n_missing": 0,
                               "n_correct": total["correct"], "accuracy": total["accuracy"],
                               "n_one_integer": total["one_integer"], "compliance_rate": total["compliance_rate"],
                               "n_bare": total["bare"], "bare_rate": total["bare_rate"],
                               "n_truncated": total["truncated"], "truncation_rate": total["truncation_rate"],
                               "n_compliant_wrong": sum(int(row["n_integers"]) == 1 and int(row["correct_strict"]) == 0 for row in group),
                               "n_noncompliant": sum(int(row["n_integers"]) != 1 for row in group)})
    pooled = []
    for k in DEPTHS:
        at_depth = [row for row in depth_rows if row["k"] == k]
        row = {"block": "A" if k in (3, 5, 7) else "B", "k": k, "n_models": 18, "n_unique_items": 20}
        for field in ("n_expected", "n_scored", "n_missing", "n_correct", "n_one_integer", "n_bare",
                      "n_truncated", "n_compliant_wrong", "n_noncompliant"):
            row[field] = sum(entry[field] for entry in at_depth)
        for rate, count in (("accuracy", "n_correct"), ("compliance_rate", "n_one_integer"),
                            ("bare_rate", "n_bare"), ("truncation_rate", "n_truncated")):
            row[rate] = row[count] / row["n_scored"]
        pooled.append(row)
    return {
        "meta": {"bank": str(BANK), "scores": str(SCORES), "panel": str(PANEL),
                 "models_in_order": models, "tau_method": "Explicit unordered-pair signs and tau-b tie denominator; no SciPy tau or main kernel",
                 "seed_sd": "statistics.stdev on four seed accuracies, ddof=1", "comparison_tolerance": 1e-12},
        "input_checks": {"unique_complete_selected_rows": len(keyed), "models": 18,
                         "A_seed_groups": 72, "G_template_groups": 90, "G_base_tasks": 30,
                         "G_five_versions_share_gold_K_seed_probe": True,
                         "G_shared_query_initial_value_relevant_ops": True,
                         "G_original_is_G30_not_A60": True,
                         "A_each_seed": 15, "G_each_template": 30, "depth_groups": 180,
                         "each_depth_each_model": 20, "pooled_each_depth": 360,
                         "depth_each_seed_probe_grid_matches_bank": True},
        "expected": {"seed_by_model.csv": seed_rows, "seed_spread_by_model.csv": spread,
                     "seed_pairs.csv": seed_pairs, "prompt_by_model.csv": prompt_rows,
                     "prompt_pairs.csv": prompt_pairs, "depth_by_model.csv": depth_rows,
                     "depth_pooled.csv": pooled},
    }


def compare_rows(expected, actual, key_fields):
    def key(row):
        return tuple(str(row[field]) for field in key_fields)
    lookup = {key(row): row for row in actual}
    differences = []
    expected_keys = {key(row) for row in expected}
    if len(lookup) != len(actual) or set(lookup) != expected_keys:
        return {"passed": False, "rows_expected": len(expected), "rows_actual": len(actual),
                "discrepancies": [{"reason": "row key mismatch", "missing": sorted(expected_keys - set(lookup)),
                                   "unexpected": sorted(set(lookup) - expected_keys)}]}
    maximum = 0.0
    compared = 0
    for row in expected:
        actual_row = lookup[key(row)]
        for field, value in row.items():
            found = actual_row.get(field)
            compared += 1
            if value is None:
                equal = found is None or found == ""
            elif isinstance(value, str):
                equal = value == found
            elif found is None or found == "":
                equal = False
            else:
                delta = abs(float(found) - value)
                maximum = max(maximum, delta)
                equal = delta == 0 if isinstance(value, int) else delta <= 1e-12
            if not equal:
                differences.append({"key": key(row), "field": field, "independent": value, "main": found})
    return {"passed": not differences, "rows_expected": len(expected), "rows_actual": len(actual),
            "values_compared": compared, "maximum_absolute_numeric_difference": maximum,
            "discrepancies": differences}


def finish_comparisons(report):
    for prefix, field, groups in (("seed", "seed", SEEDS), ("prompt", "template", TEMPLATES)):
        pooled = []
        for group in groups:
            rows = [row for row in report["expected"][prefix + "_by_model.csv"] if row[field] == group]
            row = {field: group}
            for count in ("n", "correct", "one_integer", "bare", "truncated"):
                row[count] = sum(entry[count] for entry in rows)
            for rate, count in (("accuracy", "correct"), ("compliance_rate", "one_integer"),
                                ("bare_rate", "bare"), ("truncation_rate", "truncated")):
                row[rate] = row[count] / row["n"]
            pooled.append(row)
        report["expected"][prefix + "_pooled.csv"] = pooled
    expected_summary = {}
    for prefix in ("seed", "prompt"):
        values = [row["tau_b"] for row in report["expected"][prefix + "_pairs.csv"] if row["tau_b"] is not None]
        expected_summary[prefix + "_pairs"] = {"n_pairs": len(report["expected"][prefix + "_pairs.csv"]),
                                                "defined_pairs": len(values), "mean_tau_b": statistics.mean(values),
                                                "min_tau_b": min(values), "max_tau_b": max(values)}
    spread = report["expected"]["seed_spread_by_model.csv"]
    expected_summary.update({"mean_seed_sd_accuracy": statistics.mean(row["sd_accuracy"] for row in spread),
                             "max_seed_sd_accuracy": max(row["sd_accuracy"] for row in spread),
                             "max_seed_range_accuracy": max(row["range_accuracy"] for row in spread)})
    report["expected_summary"] = expected_summary
    keys = {"seed_by_model.csv": ("model", "seed"), "seed_spread_by_model.csv": ("model",),
            "seed_pairs.csv": ("seed_1", "seed_2"), "prompt_by_model.csv": ("model", "template"),
            "prompt_pairs.csv": ("template_1", "template_2"), "depth_by_model.csv": ("model", "k"),
            "depth_pooled.csv": ("k",), "seed_pooled.csv": ("seed",), "prompt_pooled.csv": ("template",)}
    checks, pending = {}, []
    for filename, expected in report["expected"].items():
        path = OUT / filename
        if not path.exists():
            pending.append(filename)
            continue
        actual = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        checks[filename] = compare_rows(expected, actual, keys[filename])
    depth_json = OUT / "depth_analysis.json"
    if depth_json.exists():
        actual = json.loads(depth_json.read_text(encoding="utf-8"))
        checks["depth_analysis.json/by_model"] = compare_rows(report["expected"]["depth_by_model.csv"], actual["by_model"], ("model", "k"))
        checks["depth_analysis.json/pooled"] = compare_rows(report["expected"]["depth_pooled.csv"], actual["pooled"], ("k",))
    else:
        pending.append("depth_analysis.json")
    seed_prompt_json = OUT / "seed_prompt_analysis.json"
    if seed_prompt_json.exists():
        actual = json.loads(seed_prompt_json.read_text(encoding="utf-8"))
        for stem in ("seed_by_model", "seed_spread_by_model", "seed_pairs", "seed_pooled",
                     "prompt_by_model", "prompt_pairs", "prompt_pooled"):
            checks["seed_prompt_analysis.json/" + stem] = compare_rows(report["expected"][stem + ".csv"], actual[stem], keys[stem + ".csv"])
        def flatten(summary):
            result = []
            for name, value in summary.items():
                if isinstance(value, dict):
                    result.extend({"metric": name + "/" + key, "value": item} for key, item in value.items())
                else:
                    result.append({"metric": name, "value": value})
            return result
        checks["seed_prompt_analysis.json/summary"] = compare_rows(flatten(expected_summary), flatten(actual["summary"]), ("metric",))
    else:
        pending.append("seed_prompt_analysis.json")
    report["checks"] = checks
    report["pending"] = pending
    report["passed"] = not pending and all(check["passed"] for check in checks.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEPENDENT_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Independent check of the descriptive seed, wording, and depth statistics", "",
             "The frozen bank and item-level scores are read directly, and the 18 models are matched through the run_label of BFCL_INPUTS; no main summary script or statistical kernel is called.",
             "Each of the four seeds has 15 items, each of the five wordings 30 items, and each of the ten depths 20 items. The five G templates are matched to the same 30 base items by K, seed, and probe_idx, and gold, query_entity, initial_value, and n_relevant_ops are checked. G original does not use the 60 items of A.",
             "The 16 Kendall τ-b values are counted directly from the signs of the 153 unordered pairs of 18 models; the four-seed sample standard deviation uses ddof=1.", "",
             f"Passed: {report['passed']}; pending artifacts: {', '.join(pending) or 'none'}.", "",
             "| Artifact | Passed | Rows | Maximum numeric difference |", "| --- | --- | ---: | ---: |"]
    for name, check in checks.items():
        lines.append(f"| {name} | {check['passed']} | {check['rows_actual']} | {check.get('maximum_absolute_numeric_difference', '')} |")
        for difference in check["discrepancies"]:
            lines += ["", "- " + json.dumps(difference, ensure_ascii=False)]
    lines += ["", "No p-value, interval, BFCL association, model filter, or manuscript conclusion is added. The independent groupings and full-precision data are stored in INDEPENDENT_CHECK.json."]
    (OUT / "INDEPENDENT_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "pending": pending, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    finish_comparisons(build_expected())
