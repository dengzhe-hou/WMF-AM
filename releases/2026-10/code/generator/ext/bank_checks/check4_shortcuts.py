#!/usr/bin/env python3
"""Check 4 / item (c): shortcut audit for the declared state of 2026-10-04.

Facts only, computed from the prompt text; nothing is changed. A "strategy" is a
way to produce an answer without the intended computation. For each one: on how
many items does it give the gold answer.

K question now: "What was E0 immediately after slot NN?" (no update index).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from math import comb

sys.dont_write_bytecode = True
from promptparse import load_items, parse_k, parse_l  # noqa: E402

items = load_items()

# ── K ────────────────────────────────────────────────────────────────────────
strategies: dict[str, Counter] = {}
facts = Counter()
rank_by_slot: dict[int, Counter] = defaultdict(Counter)       # named slot number -> rank (1, 2, 3) -> items
field_disagree = 0
for item in items:
    if item["block"] != "K":
        continue
    p = parse_k(item["prompt"])
    gold, entity, named, q = item["gold"], p["entity"], p["named_slot"], p["rank"]
    initial, final = p["initial"][entity], p["history"][-1][entity]
    e0_lines = [s for s in p["slots"] if entity in s["deltas"]]                 # every line that mentions E0
    active_e0 = [s for s in e0_lines if s["active"] == 1]
    sign = active_e0[0]["deltas"][entity]                                      # +1 or -1 (check 3: one sign, size 1)
    upto = sum(s["slot"] <= named for s in active_e0)                          # active E0 updates up to the named slot
    checkpoints = [p["history"][s["slot"]][entity] for s in active_e0]
    previous = checkpoints[q - 2] if q >= 2 else None
    following = checkpoints[q] if q < 3 else None
    # the trajectory with the active flags ignored (every displayed slot applied)
    state = dict(p["initial"])
    flags_ignored = {}
    for s in p["slots"]:
        for name, delta in s["deltas"].items():
            state[name] += delta
        flags_ignored[s["slot"]] = state[entity]
    active_slots_upto = sum(s["active"] == 1 and s["slot"] <= named for s in p["slots"])
    answers = {
        "intended: initial + sign * (active E0 updates up to the named slot)": initial + sign * upto,
        "1 report the initial value": initial,
        "2 report the final value": final,
        "3 apply only the first E0 update": initial + sign,
        "4 count ALL E0 updates, whatever the slot": initial + sign * len(e0_lines),
        "5 use the slot number itself as the count": initial + sign * named,
        "x value just BEFORE the named slot (off by one)": p["history"][named - 1][entity],
        "x count every active slot up to the named slot, whatever the entity": initial + sign * active_slots_upto,
        "x apply slots 01..NN with the active flags ignored": flags_ignored[named],
    }
    for name, value in answers.items():
        strategies.setdefault(name, Counter())[q] += value == gold
    rank_by_slot[named][q] += 1
    facts["E0 lines that are inactive"] += len(e0_lines) - len(active_e0)
    facts["inactive lines that mention E0"] += sum(s["active"] == 0 and entity in s["deltas"] for s in p["slots"])
    facts["gold == previous checkpoint value (q>=2)"] += previous is not None and gold == previous
    facts["gold == next checkpoint value (q<=2)"] += following is not None and gold == following
    facts["gold equals another entity's initial value printed in the prompt"] += gold in [v for n, v in p["initial"].items() if n != entity]
    field_disagree += (item["initial_value"], item["final_value"], item["previous_checkpoint_value"],
                       item["next_checkpoint_value"], item["query_after"]) != (initial, final, previous, following, q)

print("K: 60 items, 20 per rank q of the named slot (q = 1st, 2nd, 3rd active E0 update; not stated in the question)")
print("  items answered correctly by each strategy   [q1, q2, q3]")
for name, by_q in strategies.items():
    print(f"  {sum(by_q.values()):3d}/60  {name}   [{by_q[1]}, {by_q[2]}, {by_q[3]}]")
for name, count in facts.items():
    print(f"  {count:3d}      {name}")
print(f"  bank fields initial/final/previous/next/query_after that disagree with the prompt: {field_disagree}")

print("\n  Does the slot number alone fix the answer (without reading which slots update E0)?")
print("  named slot -> how many of the 60 items have rank 1 / 2 / 3")
majority = 0
single_rank_slots = []
for slot in sorted(rank_by_slot):
    counts = rank_by_slot[slot]
    majority += max(counts.values())
    if len(counts) == 1:
        single_rank_slots.append(slot)
    print(f"    slot {slot:02d}: {counts[1]:2d} / {counts[2]:2d} / {counts[3]:2d}")
n01, n09 = sum(rank_by_slot[1].values()), sum(rank_by_slot[9].values())
print(f"  logically forced: slot 01 can only be the 1st update -> {n01} items (answer = initial +/- 1, the sign read from that one line).")
print(f"  slot 09 is forced to be the 3rd update only if one already knows there are exactly three E0 updates -> {n09} items.")
print(f"  every other slot number (02..08) occurs with more than one rank, except in this bank: "
      f"{[f'{s:02d}' for s in single_rank_slots if s not in (1, 9)] or 'none'}")
print(f"  best possible rule that looks at the slot number only (per-slot majority, tuned on this bank): {majority}/60")
ceil3 = sum(counts[-(-slot // 3)] for slot, counts in rank_by_slot.items())
print(f"  rule 'rank = ceil(slot/3)' (position in thirds): {ceil3}/60")
# reference: three update slots drawn uniformly among nine (84 placements); best slot-only guess per slot
placements = {slot: [comb(slot - 1, rank - 1) * comb(9 - slot, 3 - rank) for rank in (1, 2, 3)] for slot in range(1, 10)}
best = sum(max(row) for row in placements.values())
print(f"  reference, three updates placed uniformly among nine slots: best slot-only rule = {best}/{3 * comb(9, 3)} "
      f"= {best / (3 * comb(9, 3)):.3f}; it is rank = ceil(slot/3): "
      f"{all(row.index(max(row)) + 1 == -(-slot // 3) for slot, row in placements.items())}")
print("  any constant count (always 1, always 2 or always 3) gives 20/60.")

# ── L (unchanged by the rebuild; recomputed on the new file) ─────────────────
l = Counter()
n_items = Counter()
n_values = Counter()
for item in items:
    if item["block"] != "L":
        continue
    p = parse_l(item["prompt"])
    targets, gold = p["targets"], item["gold"]
    condition = item["item_id"].rsplit("-", 1)[1]
    initial = [p["initial"][t] for t in targets]
    n_items[condition] += 1
    n_values[condition] += len(targets)
    l[f"{condition}: target values equal to their initial value"] += sum(g == i for g, i in zip(gold, initial))
    l[f"{condition}: items answered by copying the initial values"] += gold == initial
    if condition == "tracking":
        updates = {t: [s["deltas"][t] for s in p["slots"] if t in s["deltas"]] for t in targets}
        first_only = [p["initial"][t] + updates[t][0] for t in targets]
        last_only = [p["initial"][t] + updates[t][-1] for t in targets]
        l["tracking: items answered by applying only the FIRST update of each target"] += first_only == gold
        l["tracking: items answered by applying only the LAST update of each target"] += last_only == gold
        l["tracking: target values answered by first-update-only"] += sum(a == g for a, g in zip(first_only, gold))
        l["tracking: target values answered by last-update-only"] += sum(a == g for a, g in zip(last_only, gold))
    else:
        confused = dict(p["initial"])
        for s in p["slots"]:
            for name, delta in s["deltas"].items():
                if name[0] == "U":
                    confused["E" + name[1:]] += delta
        l["readout: items where applying the U<i> updates to E<i> still gives gold"] += [confused[t] for t in targets] == gold
        first_m = list(p["initial"].values())[: len(targets)]
        l["readout: gold is the first M numbers of the initial-state line, in order"] += first_m == gold

print(f"\nL: tracking {n_items['tracking']} items / {n_values['tracking']} target values; "
      f"readout {n_items['readout']} items / {n_values['readout']} target values")
for name in l:
    unit = n_values if "target values" in name else n_items
    print(f"  {l[name]:3d}/{unit[name.split(':')[0]]}  {name}")

intended = sum(strategies["intended: initial + sign * (active E0 updates up to the named slot)"].values())
zero_ok = (l["tracking: target values equal to their initial value"] == 0
           and sum(strategies["1 report the initial value"].values()) == 0)
print("\nCHECK 4: facts only. Hard conditions: intended computation gives gold on 60/60, initial-value copies give 0, fields agree.")
ok = intended == 60 and zero_ok and field_disagree == 0
print("  hard conditions:", "hold" if ok else "VIOLATED")
sys.exit(0 if ok else 1)
