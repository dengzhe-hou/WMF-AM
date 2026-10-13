#!/usr/bin/env python3
"""Describe all 18 models' observed A/B depth curves without new scoring.

    python3 analysis/report_depth_stability.py

Reads the frozen bank and scored_items.csv, verifies A/B against their shared
bank generator, then summarizes saved correct_strict, integer count, format,
and truncation columns. No model calls, fitting, BFCL statistics, p-values,
confidence intervals, model exclusions, or changes to the answer rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPTHS = (3, 5, 7, 10, 15, 20, 30, 50, 75, 100)
SEEDS = (2026, 100, 200, 300)
LIMITATIONS = [
    "A and B share the frozen generator and the same prompt wrapper; K enters the random seed, so items at different K are not nested extensions of the same item.",
    "Initial state, operation sequence, queried entity, and answer vary across instances, and items may be redrawn after excluding those whose queried entity is never updated or has zero net change; the curve cannot be read as the pure causal effect of adding steps.",
    "K is the total number of operations shown; the number of operations involving the queried entity need not equal K; this report does not change the definition of the horizontal axis.",
    "All conditions have an output limit of 16 tokens and use the exactly-one-integer rule; errors include both noncompliant responses and compliant but wrong responses, and truncation can overlap with these categories.",
    "Each model has only 20 items per K; the pooled denominator 360 = 18 models × 20 items does not represent 360 independent base items. The four seeds generate different items; they are not repeated decodings.",
    "Only these observed depth points are described; no sigmoid/Kcrit fit, BFCL association, significance test, or confidence interval.",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_bank(path: Path) -> tuple[dict, dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bank = {item["item_id"]: item for item in items if item["block"] in ("A", "B")}
    assert len(bank) == 200
    builder_path = ROOT / "code" / "generator" / "build_item_bank.py"
    builder = runpy.run_path(str(builder_path))
    # These functions only construct in memory; they never write the bank.
    expected = []
    for block, depths in (("A", DEPTHS[:3]), ("B", DEPTHS[3:])):
        expected += builder["main_items"](block, builder["main_problems"](depths))
    assert {item["item_id"] for item in expected} == set(bank)
    for generated in expected:
        frozen = bank[generated["item_id"]]
        assert all(frozen.get(key) == value for key, value in generated.items() if key != "max_tokens"), generated["item_id"]
        assert frozen["max_tokens"] == 16 and frozen["answer_type"] == "int" and not frozen.get("system")
    cells = []
    for k in DEPTHS:
        group = [item for item in bank.values() if item["k"] == k]
        assert len(group) == 20 and Counter(item["seed"] for item in group) == Counter({seed: 5 for seed in SEEDS})
        assert {item["block"] for item in group} == {"A" if k <= 7 else "B"}
        cells.append({"k": k, "block": group[0]["block"], "n_items": len(group),
                      "seed_counts": dict(Counter(item["seed"] for item in group)),
                      "redrawn": sum(item["redrawn"] for item in group),
                      "n_relevant_ops_min": min(item["n_relevant_ops"] for item in group),
                      "n_relevant_ops_max": max(item["n_relevant_ops"] for item in group)})
    return bank, {
        "depths": list(DEPTHS), "seeds": list(SEEDS), "cells": cells,
        "generator": str(builder_path.resolve()), "generator_verified_items": len(expected),
        "same_generator_and_prompt": True,
        "rng_seed_formula": "seed + k * 1000 + probe_idx",
        "redraw_rule": "n_relevant_ops == 0 or gold == initial_value; next unused probe_idx within (K,seed)",
        "max_tokens": 16, "current_scoring_rule": "one_integer",
        "legacy_scoring_metadata": {block: sorted({item["legacy_scoring"] for item in bank.values() if item["block"] == block})
                                    for block in ("A", "B")},
    }


def counts(rows: list[dict], expected: int) -> dict:
    done = [row for row in rows if row["status"] == "scored"]
    assert len(done) == len(rows) == expected, "This completed panel must have no missing A/B rows"
    correct = sum(int(row["correct_strict"]) for row in done)
    one = sum(int(row["n_integers"]) == 1 for row in done)
    bare = sum(row["format"] == "bare" for row in done)
    truncated = sum(int(row["truncated"]) for row in done)
    compliant_wrong = sum(int(row["n_integers"]) == 1 and row["correct_strict"] == "0" for row in done)
    noncompliant = sum(int(row["n_integers"]) != 1 for row in done)
    assert correct + compliant_wrong + noncompliant == expected
    assert correct <= one and bare <= one
    return {
        "n_expected": expected, "n_scored": len(done), "n_missing": 0,
        "n_correct": correct, "accuracy": correct / expected,
        "n_one_integer": one, "compliance_rate": one / expected,
        "n_bare": bare, "bare_rate": bare / expected,
        "n_truncated": truncated, "truncation_rate": truncated / expected,
        "n_compliant_wrong": compliant_wrong, "n_noncompliant": noncompliant,
    }


def summarize(scored: list[dict], bank: dict, models: list[str]) -> tuple[list[dict], list[dict]]:
    assert len(models) == len(set(models)) == 18 and set(models) == {row["model"] for row in scored}
    rows = [row for row in scored if row["block"] in ("A", "B")]
    assert len(rows) == 18 * 200
    indexed = {(row["model"], row["item_id"]): row for row in rows}
    assert len(indexed) == len(rows)
    for model in models:
        assert {row["item_id"] for row in rows if row["model"] == model} == set(bank)
    for row in rows:
        item = bank[row["item_id"]]
        assert row["synthetic"] == "0" and row["status"] == "scored"
        assert row["block"] == item["block"] and int(row["k"]) == item["k"] and int(row["seed"]) == item["seed"]
        assert json.loads(row["gold"]) == item["gold"]
        assert row["answer_type"] == "int" and row["rule_strict"] == "one_integer"
        assert row["format"] in {"bare", "single_in_text", "noncompliant"}
        assert (int(row["n_integers"]) == 1) == (row["format"] != "noncompliant")
        assert int(row["truncated"]) == (row["finish_reason"] == "length")
    by_model = []
    for model in models:
        for k in DEPTHS:
            group = [indexed[(model, item_id)] for item_id, item in bank.items() if item["k"] == k]
            by_model.append({"model": model, "block": "A" if k <= 7 else "B", "k": k, **counts(group, 20)})
    pooled = []
    for k in DEPTHS:
        group = [row for row in rows if int(row["k"]) == k]
        result = {"block": "A" if k <= 7 else "B", "k": k,
                  "n_models": len(models), "n_unique_items": 20, **counts(group, len(models) * 20)}
        for key in ("n_correct", "n_one_integer", "n_bare", "n_truncated", "n_compliant_wrong", "n_noncompliant"):
            assert result[key] == sum(row[key] for row in by_model if row["k"] == k)
        pooled.append(result)
    return by_model, pooled


def markdown(report: dict) -> str:
    cells = report["bank_design"]["cells"]
    pooled = report["pooled"]
    lines = ["# Descriptive results for ten depth levels (2026-10-04)", "",
             "All 18 completed models and all A/B items are retained. A covers K=3/5/7 and B covers K=10/15/20/30/50/75/100. "
             "Each model has 20 items per K, 5 from each of the four item seeds 2026/100/200/300; in total 18×200=3600 completed responses, none missing.", "",
             f"Sources: [frozen item bank]({report['source_bank']}); [item-level scores]({report['source_scored']}).", "",
             "In the current frozen implementation A and B call the same `main_problems → build_problem → main_items/main_prompt`. "
             "The 200 items are rebuilt in memory here; prompt, gold, seed, K, probe_idx, queried entity, and redraw record all equal the frozen bank. "
             "Both blocks currently use 16 tokens, the same user wrapper, and no additional system message, and both are scored with `correct_strict=one_integer`. "
             "Their `legacy_scoring` metadata differ, which only records that the earlier source scripts differ; this report does not use the earlier scoring.", "",
             "The generator puts K into the random seed `seed + K*1000 + probe_idx`, so the initial state, operations, and queried entity are regenerated for every K. "
             "When the queried entity is never updated, or the answer equals the initial value, the item is redrawn with the next unused probe_idx within that (K, seed). "
             "The curve therefore does not add operations step by step to the same base items and cannot be read as the pure causal effect of adding steps. "
             "The four seeds generate different items; they are not repeated decodings. K is the total number of operations; the number of operations involving the queried entity need not equal K.", "",
             "| K | Block | Items per model | Items per seed | Redrawn items in the frozen bank | Range of operations involving the queried entity |",
             "| ---: | --- | ---: | --- | ---: | --- |"]
    for cell in cells:
        lines.append(f"| {cell['k']} | {cell['block']} | 20 | 5 each for 2026/100/200/300 | {cell['redrawn']} | "
                     f"{cell['n_relevant_ops_min']}–{cell['n_relevant_ops_max']} |")
    lines.extend(["", "Every pooled count has the denominator 360 = 18 models × 20 items per K; there are only 20 base items per K. "
                  "Correct counts and format counts are read directly from the stored score columns; answers are not parsed again. Exactly one integer allows surrounding text; "
                  "bare number follows the existing normalized `format=bare` class and does not mean that the raw text consists of digits only.", "",
                  "| K | Correct/360 | Accuracy | One integer/360 | Bare number/360 | Truncated/360 | Compliant but wrong/360 | Noncompliant/360 |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in pooled:
        lines.append(f"| {row['k']} | {row['n_correct']}/360 | {row['accuracy']:.1%} | {row['n_one_integer']}/360 | "
                     f"{row['n_bare']}/360 | {row['n_truncated']}/360 | {row['n_compliant_wrong']}/360 | {row['n_noncompliant']}/360 |")
    first, last = pooled[0], pooled[-1]
    lines.extend(["", f"The observed correct counts at the two end points are {first['n_correct']}/360 ({first['accuracy']:.1%}) at K=3 and "
                  f"{last['n_correct']}/360 ({last['accuracy']:.1%}) at K=100. "
                  "The per-depth results are kept as they are in the table below; local rebounds or plateaus are not smoothed away.", "",
                  "Correct, compliant but wrong, and noncompliant are three mutually exclusive classes that sum to 360 at every K. Truncated means `finish_reason=length`; "
                  "it can overlap with the classes above and must not be added to their sum.", ""])
    models = report["model_order"]
    indexed = {(row["model"], row["k"]): row for row in report["by_model"]}
    for title, key in (("Correct counts by model (denominator 20 per cell)", "n_correct"),
                       ("Exactly-one-integer counts by model (denominator 20 per cell)", "n_one_integer"),
                       ("Truncated counts by model (denominator 20 per cell)", "n_truncated")):
        lines.extend([f"## {title}", "", "| Model | " + " | ".join(f"K={k}" for k in DEPTHS) + " |",
                      "| --- | " + " | ".join("---:" for _ in DEPTHS) + " |"])
        for model in models:
            lines.append("| " + model + " | " + " | ".join(f"{indexed[(model,k)][key]}/20" for k in DEPTHS) + " |")
        lines.append("")
    lines.extend(["Per-model accuracy is the correct count above divided by 20, so the smallest step is 5 percentage points. The CSV files also store accuracy, compliance rate, "
                  "bare-number rate, truncation rate, and the mutually exclusive error classes; the JSON file stores the generator check.", "",
                  "This report fits no sigmoid or Kcrit, gives no p-value or confidence interval, computes no BFCL association, filters no model, "
                  "changes nothing in E/F, and forms no manuscript claim. The observed change with depth involves both item content and output behaviour and cannot be attributed directly to an internal working-memory mechanism.", "",
                  "Reproduce: from the package root run `python3 analysis/report_depth_stability.py`. "
                  "The outputs are `depth_by_model.csv`, `depth_pooled.csv`, `depth_analysis.json`, and this report in the same directory.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, default=ROOT / "data/item_bank/item_bank.jsonl")
    parser.add_argument("--scored", type=Path, default=ROOT / "data/scored/scored_items.csv")
    parser.add_argument("--panel", type=Path, default=ROOT / "analysis/results/BFCL_INPUTS_20261004.csv",
                        help="Only run_label is used, to share the panel's model ordering; no BFCL scores are analyzed.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis/results/stability_20261004")
    args = parser.parse_args()
    bank, design = verify_bank(args.bank)
    models = [row["run_label"] for row in read_csv(args.panel)]
    by_model, pooled = summarize(read_csv(args.scored), bank, models)
    report = {"source_bank": str(args.bank.resolve()), "source_scored": str(args.scored.resolve()),
              "model_order": models, "model_order_source": str(args.panel.resolve()),
              "bank_design": design, "by_model": by_model, "pooled": pooled, "limitations": LIMITATIONS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "depth_by_model.csv", by_model)
    write_csv(args.out_dir / "depth_pooled.csv", pooled)
    (args.out_dir / "depth_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "DEPTH_REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(f"Verified 200 frozen A/B items and 3600 scored responses; wrote {len(by_model)} model-depth and {len(pooled)} pooled rows.")
    print(args.out_dir / "DEPTH_REPORT.md")


if __name__ == "__main__":
    main()
