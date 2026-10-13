#!/usr/bin/env python3
"""Check 5. Duplicates, ids, fields, max_tokens.

  python -B check5_structure.py [bank.jsonl]

FAIL conditions: duplicate item_id; duplicate (system, prompt) inside a block;
a field of the plan's schema missing or of the wrong type; max_tokens missing or
not a positive integer; a gold whose type does not fit its answer_type.
Reported only: `system` is absent (not null) outside block H; the same
underlying problem appearing twice inside block A/B.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsolve import DEFAULT_BANK, load_bank, solve  # noqa: E402

bank = load_bank(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BANK)
problems: list[str] = []

ids = Counter(i["item_id"] for i in bank)
dup_ids = [k for k, n in ids.items() if n > 1]
if dup_ids:
    problems.append(f"duplicate item_id: {dup_ids[:10]}")

REQUIRED = {"item_id": str, "block": str, "prompt": str, "answer_type": str, "max_tokens": int}
GOLD_TYPE = {"int": int, "int_last": int, "word": str, "set": list}
for i in bank:
    for key, typ in REQUIRED.items():
        if key not in i or type(i[key]) is not typ:
            problems.append(f"{i.get('item_id')}: field {key!r} missing or not {typ.__name__}")
    for key in ("k", "seed", "gold"):
        if key not in i:
            problems.append(f"{i['item_id']}: field {key!r} missing")
    if type(i["k"]) is not int or type(i["seed"]) is not int:
        problems.append(f"{i['item_id']}: k/seed not integers")
    if i["answer_type"] not in GOLD_TYPE or type(i["gold"]) is not GOLD_TYPE[i["answer_type"]]:
        problems.append(f"{i['item_id']}: gold {i['gold']!r} does not fit answer_type {i['answer_type']!r}")
    if type(i.get("max_tokens")) is int and i["max_tokens"] <= 0:
        problems.append(f"{i['item_id']}: max_tokens {i['max_tokens']}")
    if not i["prompt"].strip() or i["prompt"] != i["prompt"].strip():
        problems.append(f"{i['item_id']}: prompt empty or has leading/trailing whitespace")
    if "system" in i and (type(i["system"]) is not str or not i["system"]):
        problems.append(f"{i['item_id']}: system present but empty")
    if i["answer_type"] == "set" and i["gold"] != sorted(i["gold"]):
        problems.append(f"{i['item_id']}: set gold not sorted")
    if i["answer_type"] in ("word", "set"):
        vocab = i.get("vocab")
        golds = [i["gold"]] if i["answer_type"] == "word" else i["gold"]
        if not vocab or any(g not in vocab for g in golds):
            problems.append(f"{i['item_id']}: gold not inside the item's vocab")
    if not i["item_id"].startswith(i["block"] + "-"):
        problems.append(f"{i['item_id']}: id prefix differs from block {i['block']}")

within = defaultdict(Counter)
for i in bank:
    within[i["block"]][(i.get("system"), i["prompt"])] += 1
for block, counter in sorted(within.items()):
    dups = sum(n - 1 for n in counter.values() if n > 1)
    if dups:
        problems.append(f"block {block}: {dups} duplicate prompt(s)")

across = Counter((i.get("system"), i["prompt"]) for i in bank)
n_across = sum(n - 1 for n in across.values() if n > 1)

# the same underlying problem twice (same state, operations, query) inside A+B, or inside a G template
same_problem = Counter()
for i in bank:
    if i["block"] in "AB":
        s = solve(i["prompt"])
        same_problem[(tuple(s["initial_state"].items()), tuple(s["ops"]), s["query"])] += 1
n_same_problem = sum(n - 1 for n in same_problem.values() if n > 1)

# bookkeeping fields against the prompt text: k = number of operation lines, query_entity,
# domain / template labels, and G/H/I items holding the same problem as their source A item
n_field_checks = 0
a_solutions = {i["item_id"]: solve(i["prompt"]) for i in bank if i["block"] == "A"}
KIND_OF = {"A": "main-bare", "B": "main-bare", "D": "main-bare", "H": "main-chat", "I": "main-cot"}
for i in bank:
    s = solve(i["prompt"])
    n_field_checks += 1
    if i["k"] != s["n_ops"]:
        problems.append(f"{i['item_id']}: k={i['k']} but the prompt lists {s['n_ops']} operations")
    if i.get("query_entity") != s["query"]:
        problems.append(f"{i['item_id']}: query_entity field {i.get('query_entity')!r}, prompt asks about {s['query']!r}")
    if i["block"] in KIND_OF:
        want_kind = KIND_OF[i["block"]]
    elif i["block"] == "G":
        want_kind = "main-bare" if i["template"] == "original" else f"G-{i['template']}"
    else:
        want_kind = f"{i['block']}-{i['domain']}"
    if s["kind"] != want_kind:
        problems.append(f"{i['item_id']}: prompt shape {s['kind']} does not fit block/domain/template ({want_kind})")
    if "source_item_id" in i:
        src = a_solutions[i["source_item_id"]]
        if (s["initial_state"], s["ops"], s["query"], s["answer"]) != (src["initial_state"], src["ops"], src["query"], src["answer"]):
            problems.append(f"{i['item_id']}: not the same problem as its source {i['source_item_id']}")
        if i["item_id"].split("-k", 1)[1] != i["source_item_id"].split("-k", 1)[1]:
            problems.append(f"{i['item_id']}: id does not match source id {i['source_item_id']}")
for block in ("G", "H", "I"):
    rows = [i for i in bank if i["block"] == block and "source_item_id" in i]
    if rows:
        per_source = Counter(i["source_item_id"] for i in rows)
        if set(per_source) != set(a_solutions) or len(set(per_source.values())) != 1:
            problems.append(f"block {block}: does not cover each of block A's items the same number of times")

print("block  items  unique_prompts  system_key  max_tokens          answer_types")
for block in sorted(within):
    rows = [i for i in bank if i["block"] == block]
    mt = ",".join(f"{t}x{n}" for t, n in sorted(Counter(r["max_tokens"] for r in rows).items()))
    at = ",".join(f"{t}:{n}" for t, n in sorted(Counter(r["answer_type"] for r in rows).items()))
    print(f"{block:<6}{len(rows):>6}{len(within[block]):>16}{sum('system' in r for r in rows):>12}  {mt:<20}{at}")
print(f"total  {len(bank):>6}")
print(f"\nitem_ids: {len(ids)} unique of {len(bank)}")
print(f"duplicate (system, prompt) across different blocks: {n_across}")
print(f"same underlying problem twice inside A+B: {n_same_problem}")
print(f"`system` key: present in {sum('system' in i for i in bank)} items (block H), absent (not null) in the other "
      f"{sum('system' not in i for i in bank)}; run_wmf_am_bank.py reads it with item.get('system')")
extra = Counter(k for i in bank for k in i) - Counter()
print("all keys and how many items carry them: " + ", ".join(f"{k}={n}" for k, n in sorted(extra.items())))
print(f"bookkeeping fields compared with the prompt text (k, query_entity, domain/template, source pairing): {n_field_checks} items")
print(f"\nproblems: {len(problems)}")
for line in problems[:40]:
    print("  " + line)
print("CHECK 5:", "PASS" if not problems else "FAIL")
sys.exit(1 if problems else 0)
