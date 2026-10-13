#!/usr/bin/env python3
"""Check 1: independent solver. Parses only the prompt text, simulates, compares with gold.

Declared state of 2026-10-04: the K question names the slot only
("What was E0 immediately after slot NN?"). The answer is the state of E0 after
that slot. This script also confirms that the named slot is an active update to
E0, that the slot number picks out one slot and one answer, and that the bank's
bookkeeping fields agree with what the prompt says.
"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.dont_write_bytecode = True
from promptparse import load_items, parse_k, parse_l  # noqa: E402

items = load_items()
gold_mismatch: list[str] = []
not_an_update: list[str] = []          # K: the named slot is not an active update to the queried entity
trajectory_problem: list[str] = []     # K: the three slot numbers of a trajectory do not select three distinct updates
field_mismatch: list[str] = []
solved = {"K": 0, "L": 0}
k_by_replicate: dict[str, list[tuple[int, int, int]]] = defaultdict(list)   # replicate -> (named slot, rank, answer)

for item in items:
    iid = item["item_id"]
    if item["block"] == "K":
        p = parse_k(item["prompt"])                           # raises if the old parenthesis is still present
        entity, named = p["entity"], p["named_slot"]
        answer = p["history"][named][entity]                  # state immediately after the named slot
        before = p["history"][named - 1][entity]
        updates = p["active_updates"]
        named_line = p["slots"][named - 1]
        # the named slot must be an ACTIVE update that touches the queried entity (so the value changes at that slot)
        if not (named in updates and named_line["active"] == 1 and entity in named_line["deltas"] and answer != before
                and [s["slot"] for s in p["slots"]].count(named) == 1):
            not_an_update.append(f"{iid}: slot {named:02d} is not exactly one active update to {entity}; "
                                 f"active updates are slots {updates}")
        if answer != item["gold"]:
            gold_mismatch.append(f"{iid}: solver {answer} != gold {item['gold']}")
        q = p["rank"]                                          # 1, 2 or 3: derived, no longer stated in the question
        checkpoints = [p["history"][s][entity] for s in updates]
        deltas = {s["deltas"][entity] for s in p["slots"] if s["active"] == 1 and entity in s["deltas"]}
        derived = {
            "query_after": q,
            "checkpoint_slot": named,
            "initial_value": p["initial"][entity],
            "final_value": p["history"][-1][entity],
            "previous_checkpoint_value": checkpoints[q - 2] if q >= 2 else None,
            "next_checkpoint_value": checkpoints[q] if q < len(checkpoints) else None,
            "same_sign": (1 if deltas == {1} else -1 if deltas == {-1} else "mixed"),
            "replicate_id": int(iid.split("-")[1][1:]),
        }
        if iid != f"K-r{derived['replicate_id']:02d}-q{q}":
            field_mismatch.append(f"{iid}: item_id does not encode the rank {q} of the named slot")
        k_by_replicate[iid.rsplit("-", 1)[0]].append((named, q, answer))
        solved["K"] += 1
    else:
        p = parse_l(item["prompt"])
        answer = [p["history"][-1][t] for t in p["targets"]]
        if answer != item["gold"]:
            gold_mismatch.append(f"{iid}: solver {answer} != gold {item['gold']}")
        touched = {e for s in p["slots"] for e in s["deltas"]}
        condition = "tracking" if set(p["targets"]) & touched else "readout"
        _, rep, m_tag, cond_tag = iid.split("-")
        derived = {
            "target_count": len(p["targets"]),
            "condition": condition,
            "display_slots": len(p["slots"]),
            "initial_values": [p["initial"][t] for t in p["targets"]],
            "replicate_id": int(rep[1:]),
        }
        if (m_tag, cond_tag) != (f"m{len(p['targets'])}", condition):
            field_mismatch.append(f"{iid}: item_id does not encode M={len(p['targets'])}/{condition}")
        solved["L"] += 1
    for key, value in derived.items():
        if item.get(key, "<absent>") != value:
            field_mismatch.append(f"{iid}: field {key}={item.get(key, '<absent>')!r}, prompt gives {value!r}")

# Within a trajectory the three questions differ only in the slot number. The slot number must then select
# three different slots, the 1st, 2nd and 3rd update, and three different answers.
for rep, rows in sorted(k_by_replicate.items()):
    slots, ranks, answers = zip(*sorted(rows))
    if len(rows) != 3 or len(set(slots)) != 3 or ranks != (1, 2, 3) or len(set(answers)) != 3:
        trajectory_problem.append(f"{rep}: named slots {slots}, ranks {ranks}, answers {answers}")

print(f"items solved from prompt text: K={solved['K']}  L={solved['L']}  total={sum(solved.values())}")
print(f"gold mismatches: {len(gold_mismatch)}")
for line in gold_mismatch:
    print("  ", line)
print(f"K: named slot is an active update to E0 and the value changes at that slot: {solved['K'] - len(not_an_update)}/60")
for line in not_an_update:
    print("  ", line)
print("K: per trajectory the three slot numbers are distinct, are the 1st/2nd/3rd update and give three different answers: "
      f"{len(k_by_replicate) - len(trajectory_problem)}/20")
for line in trajectory_problem:
    print("  ", line)
print(f"bookkeeping fields that disagree with the prompt: {len(field_mismatch)}")
for line in field_mismatch:
    print("  ", line)
ok = (not gold_mismatch and not not_an_update and not trajectory_problem and not field_mismatch
      and solved == {"K": 60, "L": 120} and len(k_by_replicate) == 20)
print("CHECK 1:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
