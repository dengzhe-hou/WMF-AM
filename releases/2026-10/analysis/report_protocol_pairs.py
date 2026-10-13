#!/usr/bin/env python3
"""Describe the existing 18-model A/H/I protocol comparison; no new inference.

Run from the package root:
    python3 analysis/report_protocol_pairs.py

Pairs H/I to A using the bank's source_item_id, checks the underlying tasks,
and reuses correct_strict from score_bank.py. Integer/"bare" diagnostics use
that scorer's existing parser; they do not change any answer's correctness.
No model filtering, association tests, or causal/working-memory claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from score_bank import int_format, integers

ROOT = Path(__file__).resolve().parent.parent
DATE = "2026-10-04"
A_PREFIX = ("You will track a sequence of point updates. "
            "You cannot refer back to the initial state after reading it once.\n\n")
A_END = "Respond with ONLY the final number."
H_PREFIX = "Track the following point updates carefully.\n\n"
H_END = "Answer with ONLY the number."
H_SYSTEM = ("You are a precise arithmetic assistant. You track numerical state changes "
            "and report final values. Always respond with only the requested number, no explanation.")
I_END = "Think step by step, then give your final answer as a single number on the last line."
TASK = re.compile(
    r"Initial state:\n(.*?)\n\nOperations \(apply in order\):\n(.*?)\n\n"
    r"(?:Question: )?After all operations, how many points does (.*?) have\?", re.DOTALL)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bank_pairs(bank: list[dict]) -> tuple[dict, dict]:
    by_id = {item["item_id"]: item for item in bank}
    assert len(by_id) == len(bank) == 840, "Bank should contain 840 unique items"
    a_items = {key: item for key, item in by_id.items() if item["block"] == "A"}
    assert len(a_items) == 60
    pairs = {key: {"A": key} for key in a_items}
    for block in "HI":
        variants = [item for item in bank if item["block"] == block]
        assert len(variants) == 60
        assert Counter(item["source_item_id"] for item in variants) == Counter({key: 1 for key in a_items})
        for item in variants:
            source_id = item["source_item_id"]
            a = a_items[source_id]
            assert TASK.search(a["prompt"]) and TASK.search(item["prompt"])
            assert TASK.search(a["prompt"]).groups() == TASK.search(item["prompt"]).groups(), source_id
            for key in ("gold", "k", "seed", "probe_idx", "query_entity", "n_relevant_ops", "initial_value"):
                assert a[key] == item[key], (source_id, block, key)
            assert a.get("system") is None and a["max_tokens"] == 16 and a["answer_type"] == "int"
            assert a["prompt"].startswith(A_PREFIX) and a["prompt"].endswith(A_END)
            if block == "H":
                expected = a["prompt"].replace(A_PREFIX, H_PREFIX, 1)
                expected = expected.replace("\n\nAfter all operations,", "\n\nQuestion: After all operations,", 1)
                expected = expected.removesuffix(A_END) + H_END
                assert item["prompt"] == expected and item.get("system") == H_SYSTEM, source_id
                assert item["max_tokens"] == 16 and item["answer_type"] == "int"
            else:
                assert item["prompt"] == a["prompt"].removesuffix(A_END) + I_END, source_id
                assert item.get("system") is None and item["max_tokens"] == 512 and item["answer_type"] == "int_last"
            pairs[source_id][block] = item["item_id"]
    return by_id, pairs


def summarize(scored: list[dict], bank: dict, pairs: dict) -> tuple[list[dict], list[dict], list[dict]]:
    models = sorted({row["model"] for row in scored})
    assert len(models) == 18, f"Expected all 18 models, got {len(models)}"
    assert all(row["synthetic"] == "0" for row in scored), "Synthetic rows are not model results"
    indexed = {(row["model"], row["item_id"]): row for row in scored}
    assert len(indexed) == len(scored), "Duplicate scored model/item rows"
    conditions, paired, summaries = [], [], []
    for model in models:
        model_rows = [row for row in scored if row["model"] == model]
        assert {row["item_id"] for row in model_rows} == set(bank), model
        for block in "AHI":
            group = [indexed[(model, ids[block])] for ids in pairs.values()]
            assert len(group) == 60 and all(row["status"] == "scored" for row in group), (model, block)
            for row in group:
                item = bank[row["item_id"]]
                assert row["block"] == block and row["answer_type"] == item["answer_type"]
                assert json.loads(row["gold"]) == item["gold"]
                assert row["rule_strict"] == ("last_integer" if block == "I" else "one_integer")
                assert int(row["n_integers"]) == len(integers(row["raw_response"]))
                if block != "I":
                    assert row["format"] == int_format(row["raw_response"])
                assert int(row["truncated"]) == int(row["finish_reason"] == "length")
            conditions.append({
                "model": model, "block": block, "n_expected": 60, "n_scored": len(group), "n_missing": 0,
                "correct": sum(int(row["correct_strict"]) for row in group),
                "exactly_one_integer": sum(int(row["n_integers"]) == 1 for row in group),
                "bare_normalized": sum(int_format(row["raw_response"]) == "bare" for row in group),
                "truncated": sum(int(row["truncated"]) for row in group),
                "max_tokens": 512 if block == "I" else 16,
                "correctness_rule": "last_integer" if block == "I" else "one_integer",
            })
        for block in "HI":
            counts = Counter()
            for a_id, ids in sorted(pairs.items()):
                a, other = indexed[(model, a_id)], indexed[(model, ids[block])]
                ca, cb = int(a["correct_strict"]), int(other["correct_strict"])
                outcome = "both_correct" if ca and cb else "loss" if ca else "gain" if cb else "both_wrong"
                counts[outcome] += 1
                paired.append({
                    "model": model, "comparison": f"A_to_{block}", "source_item_id": a_id,
                    "variant_item_id": ids[block], "gold": a["gold"], "a_correct": ca,
                    "variant_correct": cb, "outcome": outcome,
                })
            summary = {"model": model, "comparison": f"A_to_{block}", "n_expected_pairs": 60, "n_paired": sum(counts.values()),
                       **{key: counts[key] for key in ("gain", "loss", "both_correct", "both_wrong")},
                       "net_change": counts["gain"] - counts["loss"]}
            assert summary["n_paired"] == 60
            current = {row["block"]: row["correct"] for row in conditions if row["model"] == model}
            assert summary["gain"] + summary["both_correct"] == current[block]
            assert summary["loss"] + summary["both_correct"] == current["A"]
            assert summary["net_change"] == current[block] - current["A"]
            summaries.append(summary)
    return conditions, paired, summaries


def report(conditions: list[dict], summaries: list[dict], bank_path: Path, csv_path: Path) -> str:
    models = sorted({row["model"] for row in conditions})
    by_condition = {(row["model"], row["block"]): row for row in conditions}
    by_comparison = {(row["model"], row["comparison"]): row for row in summaries}
    lines = [f"# Descriptive check of the existing A/H/I protocol comparison for 18 models ({DATE})", "",
             "Existing item-level scores are used; there is no new model call, rescoring, BFCL association, or significance test. All 18 models are included. "
             "Every model completed the 60 items of each of A/H/I, 3240 responses in total; no item is missing.", "",
             f"Inputs: [frozen item bank]({bank_path.resolve()}); [item-level scores]({csv_path.resolve()}).", "",
             "H/I are linked to A through the `source_item_id` of the bank, independent of the row order of items or of the CSV file. "
             "For all 60 groups the initial state, operation sequence, queried entity, gold, K, seed, probe_idx, number of relevant operations, and initial queried value agree item by item. "
             "H and I each correspond one-to-one with A, with no duplicate or unmatched item.", "",
             "| Condition | system | Difference of the user prompt from A | Limit | Scoring for correct |",
             "| --- | --- | --- | ---: | --- |",
             "| A | no additional system | original wrapper; the ending asks for the final number only | 16 | exactly one integer, equal to gold |",
             "| H | adds an instruction for precise arithmetic, state tracking, and number-only answers | opening changed to Track the following point updates carefully.; Question: added before the query; ending changed to Answer with ONLY the number. | 16 | same as A |",
             "| I | no additional system, as in A | only the ending is replaced by Think step by step, then give your final answer as a single number on the last line. | 512 | last integer equal to gold |", "",
             "Full system message of H: `" + H_SYSTEM + "`. The original opening of A/I contains `You cannot refer back to the initial state after reading it once.`; "
             "the replacement opening of H does not. These differences were verified for all 60 groups.", "",
             "All counts in the table below have the denominator 60. Correct counts are read directly from `correct_strict`; for A/H, 'one integer' is the current format-compliance criterion, "
             "which allows text around the integer. For I, 'one integer' and 'bare number' are descriptive diagnostics only, not criteria for compliance or correctness. "
             "'Bare number' follows the `bare` class of the scorer: only one integer remains after removing the supported dollar sign, thousands commas, asterisks/backticks, and a final period. "
             "Truncated means `finish_reason=length`.", "",
             "| Model | A correct | A one integer | A bare number | A truncated | H correct | H one integer | H bare number | H truncated | I correct | I one integer | I bare number | I truncated |",
             "| --- | " + " | ".join("---:" for _ in range(12)) + " |"]
    for model in models:
        values = [by_condition[(model, block)][key] for block in "AHI"
                  for key in ("correct", "exactly_one_integer", "bare_normalized", "truncated")]
        lines.append("| " + model + " | " + " | ".join(map(str, values)) + " |")
    lines.extend(["", "Paired gains and losses all use the same 60 base items as the denominator. 'Gain' means A wrong and the other condition correct; 'loss' means A correct and the other condition wrong; "
                  "'both correct/both wrong' cover the remaining pairs. Within a row the four entries of one comparison sum to 60, and net change is gain minus loss.", "",
                  "| Model | A→H gain | Loss | Both correct | Both wrong | Net change | A→I gain | Loss | Both correct | Both wrong | Net change |",
                  "| --- | " + " | ".join("---:" for _ in range(10)) + " |"])
    for model in models:
        values = [by_comparison[(model, f"A_to_{block}")][key] for block in "HI"
                  for key in ("gain", "loss", "both_correct", "both_wrong", "net_change")]
        lines.append("| " + model + " | " + " | ".join(map(str, values)) + " |")
    totals = {block: {key: sum(row[key] for row in conditions if row["block"] == block)
                     for key in ("correct", "exactly_one_integer", "bare_normalized", "truncated")} for block in "AHI"}
    lines.extend(["", "As a count check only, summed over 18 models × 60 items:", "",
                  "| Condition | Correct / 1080 | One integer / 1080 | Bare number / 1080 | Truncated / 1080 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for block in "AHI":
        lines.append("| " + block + " | " + " | ".join(str(totals[block][key]) for key in
                     ("correct", "exactly_one_integer", "bare_normalized", "truncated")) + " |")
    for block in "HI":
        group = [row for row in summaries if row["comparison"] == f"A_to_{block}"]
        lines.append(f"\nA→{block}: total gain {sum(row['gain'] for row in group)}/1080, "
                     f"loss {sum(row['loss'] for row in group)}/1080; "
                     f"correct counts increase for {sum(row['net_change'] > 0 for row in group)}/18 models, "
                     f"are unchanged for {sum(row['net_change'] == 0 for row in group)}/18, "
                     f"and decrease for {sum(row['net_change'] < 0 for row in group)}/18.")
    lines.extend(["", "These totals are not a new overall score, and the 1080 model × item observations are not 1080 independent base items. "
                  "H changes both the user wrapper and the system message; I changes scratchpad permission, output budget, and the scoring rule at once. "
                  "The differences therefore cannot be read as the causal effect of format alone, and this comparison alone cannot establish a working-memory mechanism. "
                  "No model is filtered, the main A score is not replaced, scoring is not modified, and no manuscript conclusion is fixed on this basis.", "",
                  "Reproduce: `python3 analysis/report_protocol_pairs.py` (from the package root). "
                  "In the same directory, `protocol_conditions.csv` stores the counts per condition, `protocol_pair_summary.csv` stores the pair summary, "
                  "and `protocol_pairs.csv` stores the 2160 rows of pair detail by model, base item, and comparison.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, default=ROOT / "data" / "item_bank" / "item_bank.jsonl")
    parser.add_argument("--scored", type=Path, default=ROOT / "data" / "scored" / "scored_items.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "results")
    args = parser.parse_args()
    bank_rows = [json.loads(line) for line in args.bank.read_text(encoding="utf-8").splitlines() if line.strip()]
    bank, pairs = bank_pairs(bank_rows)
    conditions, paired, summaries = summarize(read_csv(args.scored), bank, pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "protocol_conditions.csv", conditions)
    write_csv(args.out_dir / "protocol_pairs.csv", paired)
    write_csv(args.out_dir / "protocol_pair_summary.csv", summaries)
    path = args.out_dir / "PROTOCOL_CHECK_20261004.md"
    path.write_text(report(conditions, summaries, args.bank, args.scored), encoding="utf-8")
    print(f"Verified 60 A/H/I source-item groups; 18 models, {len(conditions)} conditions, {len(paired)} paired comparisons.")
    print(path)


if __name__ == "__main__":
    main()
