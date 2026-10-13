#!/usr/bin/env python3
"""Check 5a: file format, and the frozen runner's view of the bank (no request is sent).

Declared caps: K max_tokens 16, L max_tokens 48.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
BANK = Path(__file__).resolve().parents[4] / "data" / "item_bank" / "ext_bank.jsonl"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "runner"))
import run_wmf_am_bank as runner  # noqa: E402

failures: list[str] = []
REQUIRED = ("item_id", "block", "prompt", "gold", "answer_type", "max_tokens")
K_MAX_TOKENS, L_MAX_TOKENS = 16, 48        # declared state of 2026-10-04 (L was 32 in v0)

text = BANK.read_text(encoding="utf-8")
lines = text.split("\n")
if lines[-1] != "":
    failures.append("file does not end with a newline")
lines = lines[:-1]
items = [json.loads(line) for line in lines]
ids = [item.get("item_id") for item in items]
print(f"lines: {len(lines)}   unique item_id: {len(set(ids))}   blocks: {dict(Counter(i.get('block') for i in items))}")
if len(items) != 180 or len(set(ids)) != 180:
    failures.append("not 180 unique item_ids")

combos = Counter()
for item in items:
    iid = item.get("item_id")
    missing = [key for key in REQUIRED if key not in item]
    if missing:
        failures.append(f"{iid}: missing fields {missing}")
        continue
    gold = item["gold"]
    combos[(item["block"], item["answer_type"], item["max_tokens"], type(gold).__name__)] += 1
    if not isinstance(item["prompt"], str) or not isinstance(iid, str):
        failures.append(f"{iid}: item_id/prompt not strings")
    if item["block"] == "K":
        good = item["answer_type"] == "int" and item["max_tokens"] == K_MAX_TOKENS and type(gold) is int
    elif item["block"] == "L":
        good = (item["answer_type"] == "int_list" and item["max_tokens"] == L_MAX_TOKENS and type(gold) is list
                and all(type(v) is int for v in gold) and len(gold) == item.get("target_count") and len(gold) in (1, 2, 4))
    else:
        good = False
    if not good:
        failures.append(f"{iid}: block/answer_type/max_tokens/gold type combination is wrong")
    if "system" in item:
        failures.append(f"{iid}: has a system turn (the source requests had only a user turn)")
    prompt = item["prompt"]
    if not prompt.isascii() or "\r" in prompt or "\t" in prompt or prompt != prompt.strip() or any(
            line != line.rstrip() for line in prompt.split("\n")):
        failures.append(f"{iid}: non-ASCII character, tab, CR or stray whitespace in the prompt")
for combo, count in sorted(combos.items()):
    print(f"  block {combo[0]}: answer_type={combo[1]!r} max_tokens={combo[2]} gold type={combo[3]}  x{count}")
print(f"distinct prompts: {len({item['prompt'] for item in items})}")
if len({item["prompt"] for item in items}) != 180:
    failures.append("two items share one prompt")
k_gold = [item["gold"] for item in items if item["block"] == "K"]
l_gold = [v for item in items if item["block"] == "L" for v in item["gold"]]
print(f"gold range: K {min(k_gold)}..{max(k_gold)}   L {min(l_gold)}..{max(l_gold)}   "
      f"L values with 2 digits: {sum(v < 100 for v in l_gold)}/{len(l_gold)}   negative: {sum(v < 0 for v in k_gold + l_gold)}")

# the frozen runner: load_bank, the request body it would build, and what it stores for each answer type
loaded = runner.load_bank(BANK)
print(f"runner.load_bank: {len(loaded)} items, answer types {dict(Counter(i['answer_type'] for i in loaded))}")
if loaded != items:
    failures.append("runner.load_bank returns something different from the file's JSON objects")
args = argparse.Namespace(model="dummy", max_tokens=16, temperature=0.0, extra_body={})
bodies = Counter()
for item in loaded:
    body = runner.build_body(item, args)
    bodies[(item["block"], body["max_tokens"], len(body["messages"]), body["messages"][0]["role"])] += 1
    if body["messages"] != [{"role": "user", "content": item["prompt"]}] or body["max_tokens"] != item["max_tokens"]:
        failures.append(f"{item['item_id']}: runner request body does not carry the prompt / per-item max_tokens")
for combo, count in sorted(bodies.items()):
    print(f"  runner.build_body block {combo[0]}: max_tokens={combo[1]} messages={combo[2]} ({combo[3]} turn)  x{count}")
print("  runner.score('104', 104, 'int')            ->", runner.score("104", 104, "int"))
print("  runner.score('190, 167', [190,167], 'int_list') ->", runner.score("190, 167", [190, 167], "int_list"),
      "(raw response stored only; scored offline)")

print(f"\nformat problems: {len(failures)}")
for line in failures:
    print("  ", line)
print("CHECK 5a:", "PASS" if not failures else "FAIL")
sys.exit(0 if not failures else 1)
