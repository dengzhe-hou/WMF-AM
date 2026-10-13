#!/usr/bin/env python3
"""Check 3: design properties, derived from the prompt text (declared state of 2026-10-04).

K questions name the slot only; the rank of that slot among the active E0 updates
(1, 2 or 3) is derived here from the sequence.

Grouping uses the item_id (K-rNN-qQ, L-rNN-mM-condition); check 1 already showed
that the ids encode what the prompts say.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
from promptparse import load_items, parse_k, parse_l  # noqa: E402

items = load_items()
failures: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


# ── K ────────────────────────────────────────────────────────────────────────
k_groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
k_tally: Counter = Counter()
for item in items:
    if item["block"] != "K":
        continue
    iid = item["item_id"]
    p = parse_k(item["prompt"])
    k_groups[iid.rsplit("-", 1)[0]].append((item, p))
    slots, entity = p["slots"], p["entity"]
    require(entity == "E0", f"{iid}: queried entity is {entity}, not E0")
    require(len(slots) == 9 and p["declared_slots"] == 9, f"{iid}: {len(slots)} slots, header says {p['declared_slots']}")
    require(list(p["initial"]) == ["E0", "E1", "E2", "E3", "E4"], f"{iid}: initial state lists {list(p['initial'])}")
    active_e0 = [s for s in slots if s["active"] == 1 and "E0" in s["deltas"]]
    active_other = [s for s in slots if s["active"] == 1 and "E0" not in s["deltas"]]
    inactive = [s for s in slots if s["active"] == 0]
    require(len(active_e0) == 3, f"{iid}: {len(active_e0)} active updates to E0")
    require(all(s["kind"] in ("gain", "loss") for s in active_e0), f"{iid}: an E0 update is a transfer")
    signs = {s["deltas"]["E0"] for s in active_e0}
    require(signs in ({1}, {-1}), f"{iid}: E0 update deltas are {sorted(signs)} (need one sign, size 1)")
    require(len(active_other) == 4, f"{iid}: {len(active_other)} active slots that do not touch E0")
    require(len(inactive) == 2, f"{iid}: {len(inactive)} inactive slots")
    require(all(s["amount"] == 1 for s in slots), f"{iid}: an amount other than 1")
    k_tally["inactive slots that mention E0"] += sum("E0" in s["deltas"] for s in inactive)
    k_tally["sign +1 items" if signs == {1} else "sign -1 items"] += 1
    for s in active_other:
        k_tally[f"active distractor kind: {s['kind']}"] += 1
    k_tally[f"query slot position {p['named_slot']:02d}"] += 1

require(len(k_groups) == 20, f"K has {len(k_groups)} replicates")
bodies = set()
for rep, rows in sorted(k_groups.items()):
    require(sorted(p["rank"] for _, p in rows) == [1, 2, 3], f"{rep}: ranks of the named slots {[p['rank'] for _, p in rows]}")
    require(sorted(p["named_slot"] for _, p in rows) == rows[0][1]["active_updates"],
            f"{rep}: the three named slots are not the three active E0 update slots")
    # the three prompts of a trajectory differ only in the two digits of the slot number
    require(len({item["prompt"].replace(f"after slot {p['named_slot']:02d}?", "after slot NN?") for item, p in rows}) == 1,
            f"{rep}: the three prompts differ in more than the slot number")
    require(len({p["body"] for _, p in rows}) == 1, f"{rep}: prompts differ before the question sentence")
    require(len({item["gold"] for item, _ in rows}) == len(rows), f"{rep}: gold repeats across q: {[i['gold'] for i, _ in rows]}")
    require(len({len(item["prompt"]) for item, _ in rows}) == 1, f"{rep}: the three prompts differ in length")
    bodies.add(rows[0][1]["body"])
require(len(bodies) == 20, f"only {len(bodies)} distinct K trajectories")

print("K (60 items, 20 replicates x q in 1,2,3)")
print("  per item: 9 slots; exactly 3 active E0 updates, one sign, size 1; exactly 4 active non-E0 slots; exactly 2 inactive slots")
print("  per replicate: identical prompt up to the question sentence; the questions differ only in the slot number;")
print("                 the three named slots are the three active E0 updates; three different golds; 20 distinct trajectories")
print("  no K prompt mentions an update index: " + str(sum("active update" in i["prompt"] for i in items if i["block"] == "K")) + " prompts contain 'active update'")
for key in sorted(k_tally):
    print(f"  {key}: {k_tally[key]}")

# ── L ────────────────────────────────────────────────────────────────────────
l_cells: dict[tuple[str, int], dict[str, tuple[dict, dict]]] = defaultdict(dict)
l_tally: Counter = Counter()
for item in items:
    if item["block"] != "L":
        continue
    iid = item["item_id"]
    _, rep, m_tag, condition = iid.split("-")
    m = int(m_tag[1:])
    p = parse_l(item["prompt"])
    l_cells[(rep, m)][condition] = (item, p)
    slots, targets = p["slots"], p["targets"]
    require(targets == [f"E{i}" for i in range(m)], f"{iid}: targets {targets}")
    require(len(slots) == {1: 6, 2: 8, 4: 12}[m] and p["declared_slots"] == len(slots),
            f"{iid}: {len(slots)} slots, header says {p['declared_slots']}")
    require(all(s["kind"] in ("gain", "loss") for s in slots), f"{iid}: a transfer in block L")
    require(list(p["initial"]) == [f"{c}{i}" for c in "EUV" for i in range(4)], f"{iid}: initial state lists {list(p['initial'])}")
    final = p["history"][-1]
    on_targets = [s for s in slots if set(s["deltas"]) & set(targets)]
    on_v = [s for s in slots if next(iter(s["deltas"]))[0] == "V"]
    on_u = [s for s in slots if next(iter(s["deltas"]))[0] == "U"]
    on_other_e = [s for s in slots if next(iter(s["deltas"]))[0] == "E" and not set(s["deltas"]) & set(targets)]
    require(len(on_v) == 4, f"{iid}: {len(on_v)} operations on V entities")
    require(len({next(iter(s["deltas"])) for s in on_v}) == 4, f"{iid}: a V entity is updated twice")
    require(not on_other_e, f"{iid}: an operation touches an E entity that is not listed")
    if condition == "tracking":
        per_target = Counter(next(iter(s["deltas"])) for s in on_targets)
        require(all(per_target[t] == 2 for t in targets), f"{iid}: updates per target {dict(per_target)}")
        require(all(final[t] != p["initial"][t] for t in targets), f"{iid}: a target ends at its initial value")
        require(len(slots) - len(on_targets) == 4, f"{iid}: {len(slots) - len(on_targets)} operations off the targets")
        require(not on_u, f"{iid}: tracking item updates a U entity")
        for t in targets:
            first, second = [s["deltas"][t] for s in on_targets if t in s["deltas"]]
            l_tally["target lanes: both updates same sign" if first * second > 0 else "target lanes: opposite signs"] += 1
    elif condition == "readout":
        require(not on_targets, f"{iid}: readout item updates a listed target")
        require([final[t] for t in targets] == [p["initial"][t] for t in targets] == item["gold"],
                f"{iid}: readout gold is not the initial values")
        require(len(on_u) == 2 * m, f"{iid}: {len(on_u)} redirected updates, expected {2 * m}")
    else:
        require(False, f"{iid}: unknown condition {condition}")
    l_tally[f"items M={m} {condition}, slots={len(slots)}"] += 1

require(len(l_cells) == 60, f"L has {len(l_cells)} replicate x M cells")
for (rep, m), pair in sorted(l_cells.items()):
    if set(pair) != {"tracking", "readout"}:
        require(False, f"L-{rep}-m{m}: conditions present {sorted(pair)}")
        continue
    (t_item, tp), (r_item, rp) = pair["tracking"], pair["readout"]
    t_lines, r_lines = t_item["prompt"].split("\n"), r_item["prompt"].split("\n")
    require(len(t_lines) == len(r_lines), f"L-{rep}-m{m}: line counts differ")
    differing = [(a, b) for a, b in zip(t_lines, r_lines) if a != b]
    identical_ops = sum(a["line"] == b["line"] for a, b in zip(tp["slots"], rp["slots"]))
    require(len(differing) == 2 * m, f"L-{rep}-m{m}: {len(differing)} differing lines, expected {2 * m}")
    require(identical_ops == 4, f"L-{rep}-m{m}: {identical_ops} operation lines shared by both conditions")
    for a, b in zip(tp["slots"], rp["slots"]):
        if a["line"] == b["line"]:
            continue
        (ta, da), (rb, db) = next(iter(a["deltas"].items())), next(iter(b["deltas"].items()))
        # same slot, same verb and amount; E<i> in tracking, U<i> in readout; nothing else on the line differs
        same = da == db and ta[0] == "E" and rb == "U" + ta[1:] and b["line"] == a["line"].replace(ta, rb, 1)
        require(same, f"L-{rep}-m{m}: slot {a['slot']:02d} differs by more than the entity name: {a['line']!r} / {b['line']!r}")
    require(tp["initial"] == rp["initial"] and tp["question"] == rp["question"] and tp["header"] == rp["header"],
            f"L-{rep}-m{m}: header, initial state or question differ between conditions")
    require(len(t_item["prompt"]) == len(r_item["prompt"]), f"L-{rep}-m{m}: prompt lengths differ")
    # applying the readout item's updates to the E entities instead would give the tracking gold
    require(t_item["gold"] != r_item["gold"], f"L-{rep}-m{m}: tracking and readout gold coincide")

# nesting across M within a replicate: same initial state; the M=1 and M=2 operation lists are the M=4 list
# with the lines of the dropped targets removed (order kept)
for rep in sorted({rep for rep, _ in l_cells}):
    full = l_cells[(rep, 4)]["tracking"][1]
    for m in (1, 2):
        part = l_cells[(rep, m)]["tracking"][1]
        keep = [s["text"] for s in full["slots"] if next(iter(s["deltas"])) not in {f"E{i}" for i in range(m, 4)}]
        require([s["text"] for s in part["slots"]] == keep and part["initial"] == full["initial"],
                f"L-{rep}: the M={m} item is not the M=4 trajectory with lanes >= {m} removed")

print("L (120 items, 20 replicates x M in 1,2,4 x tracking/readout)")
print("  tracking: each listed target gets exactly 2 updates and ends away from its initial value; 4 other operations")
print("  readout : no listed target is touched; gold = initial values; 2M redirected updates on U<i>")
print("  pair    : same replicate and M -> prompts identical except E<i> -> U<i> on the 2M redirected lines; 4 shared lines")
print("  every item: exactly 4 distractor operations (one on each of V0..V3); no operation on an unlisted E entity")
print("  across M  : M=1 and M=2 items are the M=4 trajectory with the dropped targets' lines removed")
for key in sorted(l_tally):
    print(f"  {key}: {l_tally[key]}")

print(f"\nproperty violations: {len(failures)}")
for line in failures:
    print("  ", line)
print("CHECK 3:", "PASS" if not failures else "FAIL")
sys.exit(0 if not failures else 1)
