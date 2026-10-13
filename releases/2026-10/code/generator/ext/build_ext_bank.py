#!/usr/bin/env python3
"""Build the extension bank: block K (intermediate-state queries) and block L
(multi-target tracking with a matched readout control).

The item designs are the two frozen source banks, rebuilt by importing their
generators and checked against the bank fingerprints in their manifests:
  checkpoint_query_v1.py        60 items  (20 trajectories x 3 query positions)
  multitarget_tracking_v2.py   120 items  (20 trajectories x M in {1,2,4} x 2 conditions)

Only the answer instruction changes, from the original JSON answer schema to the plain
text protocol of the current evaluation:
  K  "What was E0 immediately after slot NN (the q-th active update to E0)? Return only the required JSON object."
     -> "What was E0 immediately after slot NN? Respond with ONLY the number."
     The parenthesis is removed (decided 2026-10-04): it stated which update the slot was, and since all
     updates to E0 are unit steps of one sign, the answer could be computed as initial value +/- q without
     reading the sequence. Scored like block A: exactly one integer in the response, equal to gold. max_tokens 16.
  L  "... in exactly this order as the JSON answers array. Return only the required JSON object."
     -> "... in exactly this order. Respond with ONLY the requested values, separated by commas."
     Scored as: exactly M integers in the response, equal to gold in order. max_tokens 48, the allowance of
     the set items in block F (decided 2026-10-04; 32 would truncate "E0 = a, E1 = b, ..." for M = 4).
     For M = 1 (40 items) the tail is singular instead:
     "Return the final values for [E0] in exactly this order. Respond with ONLY the requested values, separated by commas."
     -> "Return the final value for [E0]. Respond with ONLY the number."
     The plural tail was used in the first runs of three models (falcon3-1b, llama-3.2-3b, falcon3-3b; bank kept as
     ext_bank_v1_m1_plural.jsonl); two of them returned several numbers on most M = 1 items in both conditions.
     The change was made after seeing those responses and before any other model was run. Scoring is unchanged.
Nothing else in the prompts differs; the script asserts this.

Output: ext_bank.jsonl in the format run_wmf_am_bank.py reads (standard library only, deterministic).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))    # the generator modules are next to this file
checkpoint = importlib.import_module("checkpoint_query_v1")
multitarget = importlib.import_module("multitarget_tracking_v2")

K_OLD_TAIL = " Return only the required JSON object."
K_NEW_TAIL = " Respond with ONLY the number."
L_OLD_TAIL = " in exactly this order as the JSON answers array. Return only the required JSON object."
L_NEW_TAIL = " in exactly this order. Respond with ONLY the requested values, separated by commas."
L_M1_OLD_HEAD = "Return the final values for ["
L_M1_NEW_HEAD = "Return the final value for ["
L_M1_NEW_TAIL = ". Respond with ONLY the number."
K_MAX_TOKENS = 16
L_MAX_TOKENS = 48


def frozen_bank(module) -> list[dict]:
    bank = module.build_bank()
    module.validate_bank(bank)
    expected = module.load_manifest()["design"]["bank_sha256"]
    found = module.bank_fingerprint(bank)
    if found != expected:
        raise SystemExit(f"{module.__name__}: bank fingerprint {found} != manifest {expected}")
    return bank


def k_item(source: dict) -> dict:
    prompt = source["prompt"]
    q = int(source["query_after"])
    hint = f" (the {q}th active update to {source['query_entity']})?"
    if prompt.count(hint) != 1 or not prompt.endswith(hint + K_OLD_TAIL):
        raise SystemExit(f"{source['item_id']}: unexpected source prompt")
    prompt = prompt[: -len(hint + K_OLD_TAIL)] + "?" + K_NEW_TAIL
    return {
        "item_id": f"K-r{source['replicate_id']:02d}-q{q}",
        "block": "K",
        "prompt": prompt,
        "gold": int(source["correct_answer"]),
        "answer_type": "int",
        "max_tokens": K_MAX_TOKENS,
        "replicate_id": source["replicate_id"],
        "query_after": q,
        "checkpoint_slot": source["checkpoint_slot"],
        "initial_value": source["initial_answer"],
        "final_value": source["final_answer"],
        "previous_checkpoint_value": source["previous_checkpoint_answer"],
        "next_checkpoint_value": source["next_checkpoint_answer"],
        "same_sign": source["same_sign"],
        "source_item_id": source["item_id"],
    }


def l_item(source: dict) -> dict:
    prompt = source["prompt"]
    if not prompt.endswith(L_OLD_TAIL):
        raise SystemExit(f"{source['item_id']}: unexpected source prompt")
    m = int(source["target_count"])
    if m == 1:
        if prompt.count(L_M1_OLD_HEAD) != 1:
            raise SystemExit(f"{source['item_id']}: unexpected source prompt")
        prompt = prompt[: -len(L_OLD_TAIL)].replace(L_M1_OLD_HEAD, L_M1_NEW_HEAD) + L_M1_NEW_TAIL
    else:
        prompt = prompt[: -len(L_OLD_TAIL)] + L_NEW_TAIL
    condition = {"parallel_tracking": "tracking", "readout_control": "readout"}[source["condition"]]
    return {
        "item_id": f"L-r{source['replicate_id']:02d}-m{m}-{condition}",
        "block": "L",
        "prompt": prompt,
        "gold": [int(v) for v in source["correct_answer"]],
        "answer_type": "int_list",
        "max_tokens": L_MAX_TOKENS,
        "replicate_id": source["replicate_id"],
        "target_count": m,
        "condition": condition,
        "display_slots": source["display_slots"],
        "initial_values": [int(source["initial_state"][e]) for e in source["query_entities"]],
        "source_item_id": source["item_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("ext_bank.jsonl"))
    args = parser.parse_args()

    items = [k_item(s) for s in frozen_bank(checkpoint)] + [l_item(s) for s in frozen_bank(multitarget)]
    ids = [item["item_id"] for item in items]
    assert len(ids) == len(set(ids)) == 180, len(ids)
    assert sum(item["block"] == "K" for item in items) == 60
    assert sum(item["block"] == "L" for item in items) == 120
    args.out.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")

    print(f"wrote {args.out} ({len(items)} items)")
    for block in ("K", "L"):
        rows = [item for item in items if item["block"] == block]
        cells: dict[str, int] = {}
        for item in rows:
            key = f"q{item['query_after']}" if block == "K" else f"M{item['target_count']}/{item['condition']}"
            cells[key] = cells.get(key, 0) + 1
        print(block, len(rows), "items; max_tokens", sorted({item["max_tokens"] for item in rows}), cells)


if __name__ == "__main__":
    main()
