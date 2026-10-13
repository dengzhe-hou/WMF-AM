#!/usr/bin/env python3
"""Budget block (2026-10-05, author decision): A's 60 core items with A's prompt, output limit 512 tokens.

Isolates the output budget: same items, same user message ("Respond with ONLY the final number."),
same parser options as A; only max_tokens changes from 16 to 512. Item ids get the prefix "A512-".
"""
import json
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[1] / "frozen/item_bank.jsonl"
OUT = Path(__file__).resolve().parent / "budget_bank.jsonl"

rows = [json.loads(line) for line in FROZEN.open(encoding="utf-8")]
core = [r for r in rows if r["block"] == "A"]
assert len(core) == 60 and all(r["max_tokens"] == 16 for r in core)
with OUT.open("w", encoding="utf-8") as handle:
    for r in core:
        item = dict(r)
        item["item_id"] = "A512-" + r["item_id"][2:] if r["item_id"].startswith("A-") else "A512-" + r["item_id"]
        item["block"] = "A512"
        item["max_tokens"] = 512
        item["source_item_id"] = r["item_id"]
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
check = [json.loads(line) for line in OUT.open(encoding="utf-8")]
assert len(check) == 60 and len({c["item_id"] for c in check}) == 60
src = {r["item_id"]: r for r in core}
assert all(c["prompt"] == src[c["source_item_id"]]["prompt"] and c["gold"] == src[c["source_item_id"]]["gold"] for c in check)
print(f"wrote {OUT} with {len(check)} items, max_tokens 512, prompts and gold identical to block A")
