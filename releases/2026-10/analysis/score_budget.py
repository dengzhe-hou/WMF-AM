#!/usr/bin/env python3
"""Score the budget block (A's 60 items at 512 output tokens) and compare it with A, H, and I.

Rules fixed before any budget response was read (2026-10-05):
  one-integer  A's rule: exactly one integer in the response and it equals gold (score_bank.strict_int).
  last-integer I's rule: the last integer equals gold (score_bank.strict_int_last).
Both rules are reported for A (16 tokens), the budget block (512 tokens), H, and I. A request that ended
with an error is a missing trial. Truncation is finish_reason == "length".
BFCL associations (Kendall tau-b, the same 10,000 model resamples as Table 2) are computed for the budget
block under both rules and reported regardless of their values.

Usage: python analysis/score_budget.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import score_bank  # noqa: E402
from analyze import clean, kendall, model_weights, percentile_ci, sign_matrix, taub  # noqa: E402

PANEL = ROOT / "analysis/results"
RES = ROOT / "data/responses/budget"
BANK = ROOT / "data/item_bank/budget_bank.jsonl"
CRITERIA = ("overall", "multi_turn", "agentic")


def records_for(label: str) -> list[dict]:
    paths = [RES / "local" / f"{label}.jsonl", RES / "api" / f"{label}.jsonl",
             RES / "hosted70b" / f"{label}.jsonl"]
    found = [p for p in paths if p.exists()]
    assert len(found) == 1, (label, found)
    final = {}
    for line in found[0].open(encoding="utf-8"):
        rec = json.loads(line)
        final[rec["item_id"]] = rec          # the runner appends; the last record of an item is final
    return list(final.values())


def main() -> None:
    bank = {json.loads(l)["item_id"]: json.loads(l) for l in BANK.open(encoding="utf-8")}
    panel = list(csv.DictReader((PANEL / "BFCL_INPUTS_20261004.csv").open(encoding="utf-8")))
    labels = [r["run_label"] for r in panel]
    scored = list(csv.DictReader((ROOT / "data/scored/scored_items.csv").open(encoding="utf-8")))

    rows, counts = [], defaultdict(lambda: defaultdict(int))
    for label in labels:
        recs = records_for(label)
        assert {r["item_id"] for r in recs} == set(bank), label
        for r in recs:
            gold = int(bank[r["item_id"]]["gold"])
            resp = r.get("raw_response")
            missing = r.get("error") is not None or resp is None
            one = (not missing) and score_bank.strict_int(resp, gold)
            last = (not missing) and score_bank.strict_int_last(resp, gold)
            trunc = r.get("finish_reason") == "length"
            rows.append({"model": label, "item_id": r["item_id"], "gold": gold, "missing": int(missing),
                         "finish_reason": r.get("finish_reason"), "truncated": int(trunc),
                         "n_integers": len(score_bank.integers(resp)) if resp else 0,
                         "correct_one_integer": int(one), "correct_last_integer": int(last),
                         "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                         "raw_response": resp})
            c = counts[label]
            c["A512_one"] += one; c["A512_last"] += last; c["A512_trunc"] += trunc; c["A512_missing"] += missing
    for s in scored:
        if s["block"] in ("A", "H", "I") and s["model"] in counts:
            g, resp = int(s["gold"]), s["raw_response"]
            c = counts[s["model"]]
            c[f"{s['block']}_one"] += score_bank.strict_int(resp, g)
            c[f"{s['block']}_last"] += score_bank.strict_int_last(resp, g)
            c[f"{s['block']}_trunc"] += int(s["truncated"])

    RES.mkdir(exist_ok=True)
    out_dir = ROOT / "analysis/results/budget"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "scored_items.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = ["A_one", "A_last", "A_trunc", "A512_one", "A512_last", "A512_trunc", "A512_missing",
            "H_one", "H_last", "H_trunc", "I_one", "I_last", "I_trunc"]
    per_model = [{"model": m, **{k: counts[m][k] for k in keys}} for m in labels]
    with (out_dir / "by_model.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["model"] + keys); w.writeheader(); w.writerows(per_model)
    totals = {k: sum(r[k] for r in per_model) for k in keys}
    assert totals["A_one"] == 342 and totals["H_one"] == 430 and totals["I_last"] == 950, totals

    y = {c: np.array([float(r[f"bfcl_{c}_1216"]) for r in panel]) for c in CRITERIA}
    weights = model_weights(len(panel), 10_000, 42)
    stored = {(r["criterion"], r["block"]): r for r in csv.DictReader((PANEL / "bfcl_comparison_20261004/associations.csv").open())}
    assoc = []
    for col in CRITERIA:
        a = np.array([counts[m]["A_one"] for m in labels])
        assert abs(clean(kendall(a, y[col])) - float(stored[col, "A"]["tau_b"])) < 1e-12
        for key in ("A512_one", "A512_last"):
            x = np.array([counts[m][key] for m in labels])
            boot = taub(sign_matrix(x)[None], sign_matrix(y[col])[None], weights)
            ci, valid = percentile_ci(boot)
            assoc.append({"criterion": col, "term": key, "tau_b": clean(kendall(x, y[col])),
                          "ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
                          "boot_valid": valid, "boot_undefined": 10_000 - valid})
    with (out_dir / "bfcl_associations.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(assoc[0])); w.writeheader(); w.writerows(assoc)
    (out_dir / "summary.json").write_text(json.dumps({"totals": totals, "per_model": per_model,
        "bfcl_associations": assoc}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=1))
    for r in per_model:
        print(f"{r['model'][:28]:28s} A {r['A_one']:2d}/{r['A_last']:2d} t{r['A_trunc']:2d} | A512 {r['A512_one']:2d}/{r['A512_last']:2d} t{r['A512_trunc']:2d} | H {r['H_one']:2d}/{r['H_last']:2d} | I {r['I_one']:2d}/{r['I_last']:2d}")
    for r in assoc:
        print(r)


if __name__ == "__main__":
    main()
