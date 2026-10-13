#!/usr/bin/env python3
"""BFCL Overall >= 50 subset (selected 2026-10-04); existing responses only.

Retains the established A/C/D comparison, three BFCL columns, 10,000 paired
model bootstraps and seed 42. No p values, fitted controls, model exclusion by
response quality, inference, or analyze.main(). Run from any directory.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze import clean, kendall, model_weights, sign_matrix, taub
from run_bfcl_comparison import BLOCKS, CRITERIA, SEED, N_BOOT, fmt, interval, summary, write_csv

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
INPUT = PANEL / "BFCL_INPUTS_20261004.csv"
OUT = PANEL / "bfcl_high_subset_20261004"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    all_rows = read_csv(INPUT)
    assert len(all_rows) == 18
    rows = [r for r in all_rows if float(r["bfcl_overall_1216"]) >= 50]
    assert {r["model"] for r in rows} == {
        "GPT-5.2", "GPT-4.1", "GPT-4.1-mini", "Claude Haiku 4.5",
        "Claude Sonnet 4.5", "Claude Opus 4.5",
    }
    n = len(rows)
    x = {b: np.array([int(r[f"{b}_correct"]) for r in rows]) for b in BLOCKS}
    y = {c: np.array([float(r[f"bfcl_{c}_1216"]) for r in rows]) for c in CRITERIA}
    assert all(int(r[f"{b}_n"]) == expected for r in rows
               for b, expected in (("A", 60), ("C", 90), ("D", 100)))
    weights = model_weights(n, N_BOOT, SEED)
    associations, differences = [], []
    for criterion in CRITERIA:
        points, boots = {}, {}
        sy = sign_matrix(y[criterion])
        for block in BLOCKS:
            sx = sign_matrix(x[block])
            points[block] = clean(kendall(x[block], y[criterion]))
            boots[block] = taub(sx[None], sy[None], weights)
            pairs = (sx * sy)[np.triu_indices(n, k=1)]
            associations.append({
                "criterion": criterion, "block": block, "n_models": n,
                "tau_b": points[block], **summary(boots[block]),
                "x_unique": len(np.unique(x[block])),
                "y_unique": len(np.unique(y[criterion])),
                "concordant_pairs": int((pairs > 0).sum()),
                "discordant_pairs": int((pairs < 0).sum()),
                "pairs_tied_in_either": int((pairs == 0).sum()),
            })
        for control in ("C", "D"):
            delta = (points["A"] - points[control]
                     if points["A"] is not None and points[control] is not None else None)
            differences.append({
                "criterion": criterion, "comparison": f"A-{control}",
                "n_models": n, "delta_tau_b": delta,
                **summary(boots["A"] - boots[control]),
            })

    protocol = {(r["model"], r["block"]): r for r in read_csv(PANEL / "protocol_conditions.csv")}
    member_rows = []
    for r in rows:
        member = {k: r[k] for k in ("model", "run_label", "family")}
        for b in BLOCKS:
            member.update({f"{b}_{k}": int(r[f"{b}_{k}"])
                           for k in ("n", "correct", "one_integer", "truncated")})
        for b in ("H", "I"):
            pr = protocol[r["run_label"], b]
            assert int(pr["n_scored"]) == 60 and int(pr["n_missing"]) == 0
            member.update({f"{b}_correct": int(pr["correct"]),
                           f"{b}_truncated": int(pr["truncated"])})
        member.update({f"bfcl_{c}": float(r[f"bfcl_{c}_1216"]) for c in CRITERIA})
        member_rows.append(member)

    # Reuse completed full-panel estimates, without re-estimating them.
    baseline = json.loads((PANEL / "bfcl_comparison_20261004/analysis.json").read_text())
    full_a = [r for r in baseline["associations"] if r["block"] == "A"]
    meta = {
        "subset_name": "high-BFCL subgroup", "criterion": "bfcl_overall_1216 >= 50",
        "threshold_origin": "Proposed in the analysis plan on 2026-10-02; selected after current scores were visible.",
        "status": "Exploratory post hoc subgroup analysis; not a universal frontier definition.",
        "input": str(INPUT), "criterion_snapshot": "2025-12-16, FC",
        "models_in_input_order": [r["model"] for r in rows],
        "n_models": n, "family_counts": dict(Counter(r["family"] for r in rows)),
        "statistic": "Kendall tau-b, all ties retained; correct counts preserve accuracy ordering.",
        "criteria_parallel": list(CRITERIA), "blocks": list(BLOCKS),
        "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                      "sampling": "six model indices with replacement; paired across blocks/criteria",
                      "interval": "pointwise 95% percentile over defined draws; no multiplicity correction",
                      "undefined": "counts retained; no imputation and no model deletion"},
        "full_panel_reference": "Existing n=18 results copied; no test of subgroup-vs-full difference.",
        "protocol": "Existing A/H/I counts shown descriptively; no new H/I association or filtering on compliance.",
        "limitations": [
            "The grouping variable is BFCL Overall itself, which restricts the range of the criterion; the subgroup association is not an unbiased substitute for the full-panel association.",
            "Only 6 models from two providers; model resampling does not remove provider dependence, and the result cannot be generalized to all frontier models.",
            "With so few models the resampling distribution is discrete; the intervals are for exploratory description only and are not reliable population bounds.",
            "No item resampling, held-out prediction, identification of internal mechanisms, or additional p-values.",
            "A retains the effects of format and truncation; H changes both the system and the user text, and I additionally changes scratchpad permission, output budget, and scoring.",
            "No model is excluded for low scores, truncation, or noncompliance; H/I do not replace the original A score.",
        ],
    }
    OUT.mkdir(exist_ok=True)
    result = {"meta": meta, "members": member_rows, "associations": associations,
              "paired_differences": differences, "full18_A_reference": full_a}
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv(OUT / "members.csv", member_rows)
    write_csv(OUT / "associations.csv", associations)
    write_csv(OUT / "paired_differences.csv", differences)
    lines = ["# High-BFCL subgroup analysis", "",
             "2026-10-04. The threshold BFCL Overall ≥ 50 retains 6 models, called the high-BFCL subgroup. "
             "The threshold was proposed in the analysis plan of 2 October; it was selected after the current scores were visible, so the analysis is exploratory. "
             "It is not a recognized frontier list.", "",
             "## Core block A and the full 18-model reference", "",
             "Kendall τ-b; 95% intervals come from 10000 model resamples, seed 42. The full 18-model results are reused as they are. "
             "The difference between the two correlation coefficients is not tested; the two groups are nested, so no significant difference can be claimed from this.", "",
             "| BFCL column | Full 18 models τ [95% interval] | High-BFCL 6 models τ [95% interval] | 6-model valid/undefined resamples | Concordant/discordant/tied model pairs |",
             "|---|---|---|---|---|"]
    for c in CRITERIA:
        full = next(r for r in full_a if r["criterion"] == c)
        sub = next(r for r in associations if r["criterion"] == c and r["block"] == "A")
        lines.append(f"| {c} | {fmt(full['tau_b'])} {interval(full)} | {fmt(sub['tau_b'])} {interval(sub)} | "
                     f"{sub['boot_valid']}/{sub['boot_undefined']} | "
                     f"{sub['concordant_pairs']}/{sub['discordant_pairs']}/{sub['pairs_tied_in_either']} |")
    lines += ["", "## Raw scores and output behaviour of the six models", "",
              "A/H/I are three protocols over the same 60 base items and do not replace the main A result. BFCL scores are on a 0–100 scale.", "",
              "| Model | A correct/60 | A exactly one integer/60 | A truncated/60 | H correct/60 | I correct/60 | Overall | Multi-Turn | Agentic |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in member_rows:
        keys = ("A_correct", "A_one_integer", "A_truncated", "H_correct", "I_correct", "bfcl_overall", "bfcl_multi_turn", "bfcl_agentic")
        lines.append(f"| {r['model']} | " + " | ".join(str(r[k]) for k in keys) + " |")
    lines += ["", "## Controls and paired association differences", "",
              "C is single-step arithmetic with 90 items and D is the cancellation control with 100 items. When all scores in a group are equal, τ is undefined and is not filled with 0.", "",
              "| BFCL column | Task | τ [95% interval] | Valid/undefined resamples |",
              "|---|---|---|---|"]
    for r in associations:
        if r["block"] != "A":
            lines.append(f"| {r['criterion']} | {r['block']} | {fmt(r['tau_b'])} {interval(r)} | {r['boot_valid']}/{r['boot_undefined']} |")
    lines += ["", "| BFCL column | Difference | Δτ [95% interval] | Valid/undefined resamples |", "|---|---|---|---|"]
    for r in differences:
        lines.append(f"| {r['criterion']} | {r['comparison']} | {fmt(r['delta_tau_b'])} {interval(r)} | {r['boot_valid']}/{r['boot_undefined']} |")
    lines += ["", "## Scope of interpretation", ""] + [f"- {v}" for v in meta["limitations"]]
    lines += ["", "An interval that spans zero does not prove no association; no Claude model was removed, H/I was not substituted, and no other threshold was chosen on the basis of these results.", "",
              "This analysis added no inference or paid requests and changed no item bank, scoring, or manuscript text.", "",
              "Reproduce: `python analysis/report_bfcl_high_subset.py`; all scores and intervals are in analysis.json at full precision."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"A": [r for r in associations if r["block"] == "A"],
                      "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
