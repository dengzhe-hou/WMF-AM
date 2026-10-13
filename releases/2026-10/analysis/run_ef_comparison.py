#!/usr/bin/env python3
"""E/F dual-rule analysis, no primary scoring rule selected.

Reproduce: python3 analysis/run_ef_comparison.py
Reuses stored labels; each rule gets the same seven rank correlations and the
same 10,000 model-bootstrap draws. Does not run analyze.main() or any model.
"""

import csv
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from analyze import clean, kendall, model_weights, sign_matrix, taub
from run_bfcl_comparison import CRITERIA, N_BOOT, SEED, fmt, interval, summary, write_csv

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
OUT = PANEL / "ef_comparison_20261004"
RULES = {"submitted": "correct_submitted", "candidate": "correct_strict"}
DOMAINS = {"E": ("color", "location", "status"), "F": ("schedule", "permissions", "inventory")}
DEPTHS = (3, 5, 7)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_group(rows, column, expected):
    assert len(rows) == expected
    assert all(row["status"] == "scored" and row[column] in ("0", "1") for row in rows)
    correct = sum(int(row[column]) for row in rows)
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "truncated": sum(int(row["truncated"]) for row in rows)}


def main():
    source = ROOT / "data/scored/scored_items.csv"
    panel_source = PANEL / "BFCL_INPUTS_20261004.csv"
    all_scored, panel = read_csv(source), read_csv(panel_source)
    names = [row["run_label"] for row in panel]
    assert len(names) == len(set(names)) == 18
    scored = [row for row in all_scored if row["block"] in DOMAINS]
    assert len(scored) == len({(row["model"], row["item_id"]) for row in scored}) == 3240
    assert {row["model"] for row in scored} == set(names)
    bank_source = ROOT / "data/item_bank/item_bank.jsonl"
    bank = {item["item_id"]: item for item in map(json.loads, bank_source.read_text().splitlines())
            if item["block"] in DOMAINS}
    for row in scored:
        item = bank[row["item_id"]]
        assert row["synthetic"] == "0" and row["status"] == "scored"
        assert (row["block"], row["domain"], int(row["k"])) == (item["block"], item["domain"], item["k"])
        assert row["answer_type"] == item["answer_type"] and json.loads(row["gold"]) == item["gold"]
    for model in names:
        assert {row["item_id"] for row in scored if row["model"] == model} == set(bank)
    by_model, by_domain, by_k, cells, pooled = [], [], [], [], []
    for rule, column in RULES.items():
        for model in names:
            current = [row for row in scored if row["model"] == model]
            for block, domains in DOMAINS.items():
                group = [row for row in current if row["block"] == block]
                prefix = {"rule": rule, "model": model, "block": block}
                by_model.append({**prefix, **summarize_group(group, column, 90)})
                for domain in domains:
                    selected = [row for row in group if row["domain"] == domain]
                    by_domain.append({**prefix, "domain": domain, **summarize_group(selected, column, 30)})
                    for k in DEPTHS:
                        cell = [row for row in selected if int(row["k"]) == k]
                        cells.append({**prefix, "domain": domain, "k": k, **summarize_group(cell, column, 10)})
                for k in DEPTHS:
                    selected = [row for row in group if int(row["k"]) == k]
                    by_k.append({**prefix, "k": k, **summarize_group(selected, column, 30)})
        for block, domains in DOMAINS.items():
            block_rows = [row for row in scored if row["block"] == block]
            for domain in ("ALL", *domains):
                domain_rows = block_rows if domain == "ALL" else [row for row in block_rows if row["domain"] == domain]
                for depth in ("ALL", *DEPTHS):
                    group = domain_rows if depth == "ALL" else [row for row in domain_rows if int(row["k"]) == depth]
                    expected = 18 * (90 if domain == "ALL" else 30) // (1 if depth == "ALL" else 3)
                    pooled.append({"rule": rule, "block": block, "domain": domain, "k": depth,
                                   **summarize_group(group, column, expected)})
    a = np.array([int(row["A_correct"]) for row in panel])
    for i, model in enumerate(names):
        a_rows = [row for row in all_scored if row["model"] == model and row["block"] == "A"]
        assert len(a_rows) == 60 and sum(int(row["correct_strict"]) for row in a_rows) == a[i]
    model_lookup = {(row["rule"], row["model"], row["block"]): row["correct"] for row in by_model}
    domain_lookup = {(row["rule"], row["model"], row["block"], row["domain"]): row["correct"] for row in by_domain}
    weights = model_weights(18, N_BOOT, SEED)
    associations = []
    for rule in RULES:
        f = np.array([model_lookup[(rule, model, "F")] for model in names])
        comparisons = [("F_vs_A", f, a)]
        comparisons += [("F_vs_bfcl_" + criterion, f,
                         np.array([float(row[f"bfcl_{criterion}_1216"]) for row in panel])) for criterion in CRITERIA]
        comparisons += [(f"F_{domain}_vs_A", np.array([domain_lookup[(rule, model, "F", domain)] for model in names]), a)
                        for domain in DOMAINS["F"]]
        for comparison, x, y in comparisons:
            bootstrap = taub(sign_matrix(x)[None], sign_matrix(y)[None], weights)
            associations.append({"rule": rule, "comparison": comparison, "n_models": 18,
                                 "tau_b": clean(kendall(x, y)), **summary(bootstrap)})
    assert len(associations) == 14
    meta = {
        "scoring_choice": "Both submitted and candidate; no primary or final scoring rule selected",
        "status": "exploratory, specified after existing model scores and prior A/C/D analyses were visible",
        "source": str(source), "panel_source": str(panel_source), "bank_source": str(bank_source),
        "models_in_input_order": names, "rules": RULES, "A_rule": "correct_strict",
        "task_correction": "E is direct assignment; F contains 60 set-update items and 30 +/-1 meeting-count items per model, not wholly non-arithmetic",
        "denominators_per_model": {"block": 90, "domain": 30, "block_k": 30, "domain_k": 10},
        "statistic": "Kendall tau-b with ties retained; fixed denominators allow exact integer correct-count rankings",
        "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                      "pairing": "Identical 18-model resamples across all comparisons and both scoring rules",
                      "ci": "Pointwise 95% percentile of defined draws; counts retained, no multiplicity adjustment"},
        "limitations": ["Model resampling only; does not include item sampling or remove provider dependence",
                        "Positive cross-task correlation is not proof of a common working-memory mechanism",
                        "E/F differ in entities, answer types and some token budgets, not an isolated arithmetic manipulation",
                        "No association-difference test, added controls, p value, family resampling, or model screening"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    tables = {"accuracy_by_model": by_model, "accuracy_by_domain": by_domain,
              "accuracy_by_k": by_k, "accuracy_cells": cells, "pooled_accuracy": pooled,
              "associations": associations}
    OUT.mkdir(exist_ok=True)
    for name, rows in tables.items():
        write_csv(OUT / (name + ".csv"), rows)
    (OUT / "analysis.json").write_text(json.dumps({"meta": meta, **tables}, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# E/F analysis under two rules (18 models, 2026-10-04)", "",
             "Both rule sets are reported side by side and neither is designated as primary: the submitted rule and the candidate rule are reported in full, with no primary score. "
             "Existing labels are read directly, with no new model call or rescoring. All 18 models are retained and the 3,240 E/F responses are complete.", "",
             "E is direct assignment of colors/locations/statuses; F contains 60 permission/item set-update items and 30 meeting-count items with increments and decrements. "
             "This corrects the earlier description of the whole F block as non-arithmetic; all three domains are retained. E/F differ in entity count, answer type, and some output limits, "
             "so their score difference cannot be read as the causal effect of changing only the arithmetic/accumulation demand.", "",
             "## Accuracy by block and domain", "",
             "Each model has 90 items per block and 30 per domain; the table pools the 18 models.", "",
             "| Block | Domain | Submitted rule | Candidate rule |", "| --- | --- | ---: | ---: |"]
    lookup = {(row["rule"], row["block"], row["domain"], row["k"]): row for row in pooled}
    for block, domains in DOMAINS.items():
        for domain in ("ALL", *domains):
            entries = [lookup[(rule, block, domain, "ALL")] for rule in RULES]
            lines.append(f"| {block} | {domain} | " + " | ".join(f"{row['correct']}/{row['n']} ({row['accuracy']:.2%})" for row in entries) + " |")
    lines += ["", "## 14 associations and intervals", "",
              "The same 7 associations under each rule. τ is Kendall τ-b, and 95% intervals come from 10,000 shared model bootstrap draws, seed 42. "
              "The three BFCL columns remain in parallel; this is an exploratory analysis, and intervals are not adjusted for multiple comparisons.", "",
              "| Relation | Submitted τ (95% interval) | Candidate τ (95% interval) | Valid/undefined (submitted; candidate) |",
              "| --- | --- | --- | --- |"]
    indexed = {(row["rule"], row["comparison"]): row for row in associations}
    for row in associations[:7]:
        comparison = row["comparison"]
        entries = [indexed[(rule, comparison)] for rule in RULES]
        lines.append(f"| {comparison} | " + " | ".join(f"{fmt(item['tau_b'])} {interval(item)}" for item in entries)
                     + " | " + "; ".join(f"{item['boot_valid']}/{item['boot_undefined']}" for item in entries) + " |")
    lines += ["", "## Accuracy by number of steps", "",
              "Each model has 30 items per block and K, and 10 per domain and K. ALL rows overlap with the domain rows and are not added to them.", "",
              "| Block | Domain | K | Submitted correct/items | Candidate correct/items |", "| --- | --- | ---: | ---: | ---: |"]
    for block, domains in DOMAINS.items():
        for domain in ("ALL", *domains):
            for k in DEPTHS:
                entries = [lookup[(rule, block, domain, k)] for rule in RULES]
                lines.append(f"| {block} | {domain} | {k} | " + " | ".join(f"{item['correct']}/{item['n']}" for item in entries) + " |")
    lines += ["", "## Block scores of all 18 models", "",
              "| Model | E submitted | E candidate | F submitted | F candidate |", "| --- | ---: | ---: | ---: | ---: |"]
    for model in names:
        lines.append(f"| {model} | " + " | ".join(f"{model_lookup[(rule, model, block)]}/90" for block in ("E", "F") for rule in RULES) + " |")
    lines += ["", "## Sources, limitations, and reproduction", "",
              "The specific rules are in analysis/score_bank.py, and all 7 differences are in EF_SCORING_DIFFERENCES_20261004.csv in the parent directory. "
              "The columns used for E/F are correct_submitted and the candidate correct_strict; A always uses correct_strict. "
              "Both rules are deterministic automatic parsers; some differences are output-format issues, so the 7 cases cannot all be called semantic errors.", "",
              "The model bootstrap reflects model resampling within this fixed convenience panel only; it does not include item sampling or remove same-provider dependence. "
              "A positive association does not prove a shared internal mechanism; spanning 0 does not prove zero association. The associations here are not tested directly against the earlier A/C/D associations, "
              "so F cannot be claimed to be better than the other probes by comparing point estimates or intervals alone.", "",
              "Raw counts, denominators, accuracies, and truncation counts by model, domain, K, and their cross-classifications are all in the CSV files and analysis.json, "
              "not only summaries or favourable domains. Reproduce: `python3 analysis/run_ef_comparison.py`. "
              "There is no p-value, family-wise interval, additional control, model filter, designation of a primary score, or manuscript change."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "associations": associations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
