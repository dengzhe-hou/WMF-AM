#!/usr/bin/env python3
"""Descriptive stability analysis of existing A seeds and G prompts.

Run: python3 analysis/report_seed_prompt_stability.py
No model inference, rescoring, bootstrap, p values, or automatic claim threshold.
"""

import csv
import json
import platform
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
OUT = PANEL / "stability_20261004"
SEEDS = (2026, 100, 200, 300)
TEMPLATES = ("original", "formal", "casual", "minimal", "verbose")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(rows):
    n = len(rows)
    correct = sum(int(row["correct_strict"]) for row in rows)
    compliant = sum(int(row["n_integers"]) == 1 for row in rows)
    bare = sum(row["format"] == "bare" for row in rows)
    truncated = sum(int(row["truncated"]) for row in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "one_integer": compliant, "compliance_rate": compliant / n,
            "bare": bare, "bare_rate": bare / n,
            "truncated": truncated, "truncation_rate": truncated / n}


def pairwise(groups, category, order, models):
    indexed = {(row["model"], row[category]): row for row in groups}
    result = []
    for first, second in combinations(order, 2):
        # Every model has the same denominator within each condition, so the
        # integer numerator preserves ranks and ties without rounding.
        left = [indexed[(model, first)]["correct"] for model in models]
        right = [indexed[(model, second)]["correct"] for model in models]
        tau = float(kendalltau(left, right, variant="b").statistic)
        result.append({category + "_1": first, category + "_2": second,
                       "n_models": len(models), "tau_b": tau if np.isfinite(tau) else None})
    return result


def pair_summary(rows):
    valid = [row["tau_b"] for row in rows if row["tau_b"] is not None]
    return {"n_pairs": len(rows), "defined_pairs": len(valid),
            "mean_tau_b": float(np.mean(valid)) if valid else None,
            "min_tau_b": min(valid) if valid else None,
            "max_tau_b": max(valid) if valid else None}


def main():
    scored_source = ROOT / "data/scored/scored_items.csv"
    bank_source = ROOT / "data/item_bank/item_bank.jsonl"
    bank = [json.loads(line) for line in bank_source.read_text().splitlines()]
    bank = {item["item_id"]: item for item in bank if item["block"] in ("A", "G")}
    assert Counter(item["block"] for item in bank.values()) == {"A": 60, "G": 150}
    mapping = read_csv(PANEL / "BFCL_INPUTS_20261004.csv")
    models = [row["run_label"] for row in mapping]
    assert len(models) == len(set(models)) == 18
    scored = [row for row in read_csv(scored_source) if row["block"] in ("A", "G")]
    assert len(scored) == len({(row["model"], row["item_id"]) for row in scored}) == 18 * 210
    assert {row["model"] for row in scored} == set(models)
    for model in models:
        assert {row["item_id"] for row in scored if row["model"] == model} == set(bank)
    for row in scored:
        item = bank[row["item_id"]]
        assert row["status"] == "scored" and row["synthetic"] == "0"
        assert row["rule_strict"] == "one_integer" and row["answer_type"] == "int"
        assert item["max_tokens"] == 16 and item["answer_type"] == "int"
        assert int(row["k"]) == item["k"] and int(row["seed"]) == item["seed"]
        assert json.loads(row["gold"]) == item["gold"]
        if row["block"] == "G":
            assert row["template"] == item["template"]
    # G's original is its own 30-task condition, never A's 60-task score.
    bases = {}
    for item in bank.values():
        if item["block"] != "G":
            continue
        key = (item["seed"], item["k"], item["probe_idx"])
        assert item["template"] not in bases.setdefault(key, {})
        bases[key][item["template"]] = item
    assert len(bases) == 30
    for group in bases.values():
        assert set(group) == set(TEMPLATES)
        for key in ("gold", "query_entity", "initial_value", "n_relevant_ops"):
            assert len({item[key] for item in group.values()}) == 1
    assert Counter(item["seed"] for item in bank.values() if item["block"] == "A") == {seed: 15 for seed in SEEDS}
    seed_rows, template_rows, spread = [], [], []
    for model in models:
        current = [row for row in scored if row["model"] == model]
        for seed in SEEDS:
            group = [row for row in current if row["block"] == "A" and int(row["seed"]) == seed]
            assert Counter(int(row["k"]) for row in group) == {3: 5, 5: 5, 7: 5}
            seed_rows.append({"model": model, "seed": seed, **counts(group)})
        accuracies = [row["accuracy"] for row in seed_rows if row["model"] == model]
        spread.append({"model": model, "mean_accuracy": float(np.mean(accuracies)),
                       "sd_accuracy": float(np.std(accuracies, ddof=1)),
                       "min_accuracy": min(accuracies), "max_accuracy": max(accuracies),
                       "range_accuracy": max(accuracies) - min(accuracies)})
        for template in TEMPLATES:
            group = [row for row in current if row["block"] == "G" and row["template"] == template]
            assert Counter(int(row["k"]) for row in group) == {3: 10, 5: 10, 7: 10}
            template_rows.append({"model": model, "template": template, **counts(group)})
    seed_pairs = pairwise(seed_rows, "seed", SEEDS, models)
    prompt_pairs = pairwise(template_rows, "template", TEMPLATES, models)
    seed_pooled = [{"seed": seed, **counts([row for row in scored
                    if row["block"] == "A" and int(row["seed"]) == seed])} for seed in SEEDS]
    prompt_pooled = [{"template": template, **counts([row for row in scored
                      if row["block"] == "G" and row["template"] == template])} for template in TEMPLATES]
    summary = {"seed_pairs": pair_summary(seed_pairs), "prompt_pairs": pair_summary(prompt_pairs),
               "mean_seed_sd_accuracy": float(np.mean([row["sd_accuracy"] for row in spread])),
               "max_seed_sd_accuracy": max(row["sd_accuracy"] for row in spread),
               "max_seed_range_accuracy": max(row["range_accuracy"] for row in spread)}
    tables = {"seed_by_model": seed_rows, "seed_spread_by_model": spread,
              "seed_pairs": seed_pairs, "seed_pooled": seed_pooled,
              "prompt_by_model": template_rows, "prompt_pairs": prompt_pairs, "prompt_pooled": prompt_pooled}
    metadata = {
                "scored_source": str(scored_source), "bank_source": str(bank_source),
                "models_in_order": models, "seeds": list(SEEDS), "templates": list(TEMPLATES),
                "scoring": "Existing correct_strict: exactly one integer matching gold; no rescoring",
                "seed_design": "4 item-generation seeds, 15 tasks per model/seed, 5 at each K=3/5/7",
                "prompt_design": "G's own 30 base tasks, 10 at each K=3/5/7, seed=42, repeated in five templates; original is not A",
                "G_option_A": "minimal's 30 prompts append Respond with ONLY the final number.; all G max_tokens=16",
                "pairing_check": "30 G base keys (seed,k,probe_idx), five templates per key, matching gold/query/initial queried value/relevant-op count",
                "statistics": "Kendall tau-b of 18-model accuracy rankings for each pair; unweighted arithmetic mean of defined pair coefficients; per-model four-seed sample SD ddof=1",
                "format": "one_integer = saved n_integers==1, allows surrounding text; bare follows saved scorer normalization; truncated=finish_reason length",
                "limits": ["Descriptive, selected after results were visible; no p, CI, significance count, or automatic claim rule",
                           "Item-seed variation is not repeated model-generation reliability",
                           "Rank correlation does not imply invariant scores or per-item predictions",
                           "Shared models/tasks mean pair coefficients are dependent; their mean is descriptive",
                           "Floor, ceiling and ties retained; no model or template exclusion",
                           "Prompt/compliance/truncation differences remain; no causal mechanism inference"],
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    OUT.mkdir(exist_ok=True)
    for name, rows in tables.items():
        write_csv(OUT / (name + ".csv"), rows)
    (OUT / "seed_prompt_analysis.json").write_text(json.dumps({"meta": metadata, "summary": summary, **tables},
        ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Seed and wording stability (18 models, 2026-10-04)", "",
             "Only the existing strict scores are used and all 18 models are retained. This report gives descriptive point values, with no p-value, interval, count of significant pairs, or automatic conclusion threshold.", "",
             "## Four item seeds", "",
             "Each seed has 15 items, 5 at each of K=3/5/7. The table gives correct counts out of 15 and the sample SD over the four seeds (ddof=1, in percentage points). "
             "The seeds generate different items; they are not repeated generations of the same item.", "",
             "| Model | 2026 | 100 | 200 | 300 | SD (percentage points) | Range (percentage points) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {(row["model"], row["seed"]): row for row in seed_rows}
    for row in spread:
        lines.append("| " + row["model"] + " | " + " | ".join(str(lookup[(row["model"], seed)]["correct"]) for seed in SEEDS)
                     + f" | {100 * row['sd_accuracy']:.2f} | {100 * row['range_accuracy']:.2f} |")
    lines += ["", f"The arithmetic mean of the 18 sample SDs is {100 * summary['mean_seed_sd_accuracy']:.2f} percentage points; it is not a standard error of the mean or a confidence interval.", "",
              "| Seed 1 | Seed 2 | Models | τ-b |", "| --- | --- | ---: | ---: |"]
    for row in seed_pairs:
        lines.append(f"| {row['seed_1']} | {row['seed_2']} | 18 | {row['tau_b']:.6f} |")
    lines += ["", "## Five wordings", "",
              "G uses its own 30 base items, matched one-to-one across the five templates; original is taken from G itself, not from the 60 items of A. "
              "minimal carries the appended number-only sentence (option A); all five templates use 16 tokens and strict one-integer scoring. Each cell below is a correct count out of 30.", "",
              "| Model | original | formal | casual | minimal | verbose |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {(row["model"], row["template"]): row for row in template_rows}
    for model in models:
        lines.append("| " + model + " | " + " | ".join(str(lookup[(model, template)]["correct"]) for template in TEMPLATES) + " |")
    lines += ["", "| Template 1 | Template 2 | Models | τ-b |", "| --- | --- | ---: | ---: |"]
    for row in prompt_pairs:
        lines.append(f"| {row['template_1']} | {row['template_2']} | 18 | {row['tau_b']:.6f} |")
    lines += ["", "## Summary and format by condition", "",
              "Compliant means the response contains exactly one integer, with surrounding text allowed; it is not the same as bare number. Truncated can overlap with correct/compliant and is not a mutually exclusive class. "
              "All models have the same denominator, so the pooled rate equals the arithmetic mean of the model rates, but the responses are not an i.i.d. sample and no interval is inferred from this table.", "",
              "| Condition | Responses | Correct | Compliant | Bare number | Truncated |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in [*seed_pooled, *prompt_pooled]:
        label = f"seed={row['seed']}" if "seed" in row else row["template"]
        lines.append(f"| {label} | {row['n']} | {row['correct']} | {row['one_integer']} | {row['bare']} | {row['truncated']} |")
    lines += ["", "## Limits of interpretation and reproduction", "",
              "All point values of the rank correlations are in the tables above; a high correlation is no substitute for score stability or item-level agreement. Seed/template pairs share the same models, "
              "so the 6 or 10 pairs cannot be treated as independent replications. Floor, ceiling, and ties are kept as they are, and no model is removed for its score. "
              "G and A differ in item set and generator, so their absolute score difference is not interpreted as an effect of wording alone.", "",
              "The per-model condition tables also store compliant/bare-number/truncated counts and rates, see seed_by_model.csv and prompt_by_model.csv; "
              "full precision and sources are in seed_prompt_analysis.json. Reproduce: `python3 analysis/report_seed_prompt_stability.py`. "
              "For depth see DEPTH_REPORT.md. This analysis involves no new model call, scoring change, manuscript change, or conclusion threshold."]
    (OUT / "SEED_PROMPT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
