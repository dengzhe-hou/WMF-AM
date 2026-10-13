#!/usr/bin/env python3
"""Check 1. Recompute every answer from the prompt text alone and compare with gold.

  python -B check1_solver.py [bank.jsonl]

Blocks A-I (int, int_last, set, and also E's word answers) are solved from the
prompt text by textsolve.py.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsolve import DEFAULT_BANK, load_bank, solve  # noqa: E402

bank = load_bank(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BANK)
mismatch: list[str] = []
solved: Counter = Counter()
kinds: dict[str, Counter] = defaultdict(Counter)

for item in bank:
    sol = solve(item["prompt"])
    kinds[item["block"]][sol["kind"]] += 1
    solved[item["block"]] += 1
    if sol["answer"] != item["gold"] or type(sol["answer"]) is not type(item["gold"]):
        mismatch.append(f"{item['item_id']}: gold {item['gold']!r}, recomputed {sol['answer']!r}")

print("block  items  solved_from_text  prompt shapes")
for block in sorted(kinds):
    n = sum(kinds[block].values())
    print(f"{block:<6}{n:>6}{solved[block]:>18}  {dict(kinds[block])}")

print(f"\nmismatches: {len(mismatch)}")
for line in mismatch:
    print("  " + line)
print("CHECK 1:", "PASS" if not mismatch else "FAIL")
sys.exit(1 if mismatch else 0)
