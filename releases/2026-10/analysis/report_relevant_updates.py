#!/usr/bin/env python3
"""Core accuracy by the number of updates to the queried entity, and that number by depth (2026-10-11).

    python analysis/report_relevant_updates.py

Reads the frozen bank and scored_items.csv. Pools the 18 models' core (block A) responses by the bank field
n_relevant_ops, with four or more updates in one bin, and gives the median of that field per K over blocks A
and B. Descriptive only: no resampling, intervals or model exclusions.
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
SCORED = ROOT / "data/scored/scored_items.csv"
BANK = ROOT / "data/item_bank/item_bank.jsonl"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with SCORED.open(encoding="utf-8", newline="") as handle:
        core = [r for r in csv.DictReader(handle) if r["block"] == "A"]
    assert len(core) == 1080 and len({r["model"] for r in core}) == 18
    pooled = defaultdict(lambda: [0, 0])
    for r in core:
        n = int(r["n_relevant_ops"])
        cell = pooled["4+" if n >= 4 else str(n)]
        cell[0] += int(r["correct_strict"])
        cell[1] += 1
    accuracy = [{"block": "A", "updates_to_queried_entity": key, "n_responses": n, "n_correct": c,
                 "accuracy": round(c / n, 6)} for key, (c, n) in sorted(pooled.items())]
    by_k = defaultdict(list)
    with BANK.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["block"] in ("A", "B"):
                by_k[item["k"]].append(item["n_relevant_ops"])
    depth = [{"k": k, "n_items": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
             for k, v in sorted(by_k.items())]
    write_csv(PANEL / "relevant_updates_accuracy.csv", accuracy)
    write_csv(PANEL / "relevant_updates_by_k.csv", depth)
    for r in accuracy:
        print(r["updates_to_queried_entity"], f"{r['n_correct']}/{r['n_responses']}", f"{100 * r['accuracy']:.1f}%")
    print("median updates to the queried entity by K:", {r["k"]: r["median"] for r in depth})


if __name__ == "__main__":
    main()
