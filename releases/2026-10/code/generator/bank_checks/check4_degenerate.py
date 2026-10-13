#!/usr/bin/env python3
"""Check 4. Degenerate items under the Q10 definition, counted from the prompt text.

  python -B check4_degenerate.py [bank.jsonl]

Q10 definition: the queried entity is changed by no operation ("untouched"), or
the answer equals its initial value. Both are read from the prompt text by
textsolve.py. PASS means block A has none. The other blocks are only reported;
nothing is changed. The bank's own n_relevant_ops / initial_value fields are
compared with the text-derived values at the end.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsolve import DEFAULT_BANK, is_degenerate, load_bank, solve  # noqa: E402

bank = load_bank(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BANK)
rows: dict[str, list] = defaultdict(list)
field_mismatch = []
for item in bank:
    sol = solve(item["prompt"])
    untouched, same = is_degenerate(sol)
    rows[item["block"]].append((item, sol, untouched, same))
    if item["n_relevant_ops"] != sol["n_touch"] or item["initial_value"] != sol["initial"]:
        field_mismatch.append(f"{item['item_id']}: bank n_relevant_ops={item['n_relevant_ops']} initial_value={item['initial_value']!r}; "
                              f"text gives {sol['n_touch']} / {sol['initial']!r}")

print("block  items  untouched  answer==initial  either")
for block in sorted(rows):
    r = rows[block]
    print(f"{block:<6}{len(r):>6}{sum(u for *_, u, _ in r):>11}{sum(s for *_, s in r):>17}{sum(u or s for *_, u, s in r):>8}")

print("\nby sub-cell, blocks other than A (only cells with at least one such item):")
for block in sorted(rows):
    if block == "A":
        continue
    cells: dict[tuple, list] = defaultdict(list)
    for item, sol, u, s in rows[block]:
        cells[(sol["kind"], sol["n_ops"])].append((item["item_id"], u, s))
    for (kind, n_ops), members in sorted(cells.items()):
        n_u, n_s = sum(u for _, u, _ in members), sum(s for _, _, s in members)
        if n_u or n_s:
            print(f"  {block} {kind} K={n_ops}: {len(members)} items, untouched {n_u}, answer==initial {n_s}")

a_bad = [item["item_id"] for item, _, u, s in rows["A"] if u or s]
print(f"\nblock A degenerate items: {len(a_bad)} {a_bad}")
print(f"bank fields n_relevant_ops / initial_value that disagree with the prompt text: {len(field_mismatch)}")
for line in field_mismatch[:20]:
    print("  " + line)
ok = not a_bad and not field_mismatch
print("CHECK 4:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
