#!/usr/bin/env python3
"""Build the CANDIDATE unified WMF-AM item bank (blocks A-I).

DRAFT. Nothing written by this script is frozen. Standard library only,
deterministic, no network. Output is JSONL in the format run_wmf_am_bank.py reads.

Prompt templates below are copied from the submitted-version scripts in
/path/to/earlier_repo/code/ ; the source file and line are cited next to
each one. Explicitly enabled new text is declared per item. The verifier
check3_templates.py compares prompts with the original source functions and
reports declared additions separately from undeclared differences.

Configuration choices are switches, all required (no silent default):
  --include-cot | --no-cot                 block I            (Q4)
  --paraphrase-mode rewrite_A | original   block G            (Q17)
  --minimal-final-number | --no-minimal-final-number
                                           append the number-only
                                           instruction to original G/minimal
  --max-tokens-int N                       every "int" item   (Q9)
  --max-tokens-block BLOCK=N | none        per-block override of max_tokens, applied
                                           after the two rules above; repeatable
Scoring (Q9, Q17) is not decided here. Each item carries `answer_type` (which
strict rule applies) and `legacy_scoring` (the submitted script's rule), so
both can be computed later from the stored raw responses.

Built in: Q10. In blocks A and B an item is redrawn
when the queried entity is changed by no operation or the answer equals its
initial value.

Candidate command:
  python build_item_bank.py --include-cot \
      --paraphrase-mode rewrite_A --no-minimal-final-number \
      --max-tokens-int 16 --max-tokens-block none

Frozen bank v2 (2026-10-03: block G keeps the submitted cap of
1024 tokens, config.py MODELS entries, all "max_tokens": 1024):
  python build_item_bank.py --include-cot --paraphrase-mode original \
      --no-minimal-final-number \
      --max-tokens-int 16 --max-tokens-block G=1024 --out ../frozen/item_bank.jsonl
Frozen bank v1 (kept as ../frozen/item_bank_v1_g16_20261003.jsonl) is the same
command with --max-tokens-block none.
Frozen bank v3 (2026-10-03; option A: 30 minimal prompts gain the
same number-only instruction as the main experiment; all G items use 16 tokens):
  python build_item_bank.py --include-cot --paraphrase-mode original \
      --minimal-final-number --max-tokens-int 16 --max-tokens-block none \
      --out ../frozen/item_bank.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path(__file__).resolve().parent / "item_bank_CANDIDATE.jsonl"

# max_tokens for the non-"int" answer types (plan section 4.1 table).
# "int" items take --max-tokens-int.
FIXED_MAX_TOKENS = {"int_last": 512, "word": 16, "set": 48}


def make_item(item_id: str, block: str, k: int | None, seed: int | None, probe_idx: int,
              prompt: str, gold: Any, answer_type: str, n_relevant_ops: int | None,
              initial_value: Any, legacy_scoring: str, system: str | None = None,
              **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"item_id": item_id, "block": block, "k": k, "seed": seed, "probe_idx": probe_idx}
    if system is not None:
        item["system"] = system
    item.update(prompt=prompt, gold=gold, answer_type=answer_type, max_tokens=None,
                n_relevant_ops=n_relevant_ops, initial_value=initial_value,
                legacy_scoring=legacy_scoring, new_text=False)
    item.update(extra)
    return item


# ── Blocks A and B: main probe ───────────────────────────────────────────────

# wmf_am_frozen_v1.py L49-52
ENTITY_NAMES = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
    "Henry", "Iris", "James", "Kate", "Leo", "Mia", "Noah",
)
A_DEPTHS = (3, 5, 7)                        # wmf_am_frozen_v1.py L39
B_DEPTHS = (10, 15, 20, 30, 50, 75, 100)    # wmf_am_extended_k.py L27, minus 3/5/7 (block A)
CORE_SEEDS = (2026, 100, 200, 300)          # wmf_am_frozen_v1.py L40; wmf_am_extended_k.py L29
PROBES_PER_SEED = 5                         # wmf_am_frozen_v1.py L41; wmf_am_extended_k.py L30


def main_prompt(initial_state: dict[str, int], op_texts: list[str], query_entity: str) -> str:
    # wmf_am_frozen_v1.py L92-101. Same text in wmf_am_extended_k.py L50-59,
    # wmf_am_yoked_control.py L184-198, wmf_am_paraphrase.py L82-93 ("original"),
    # wmf_am_template_harmonization.py L55-67 ("bare").
    state_text = ", ".join(f"{entity}: {value} points" for entity, value in initial_state.items())
    operations_text = "\n".join(f"  {idx + 1}. {operation}" for idx, operation in enumerate(op_texts))
    return (
        "You will track a sequence of point updates. "
        "You cannot refer back to the initial state after reading it once.\n\n"
        f"Initial state:\n{state_text}\n\n"
        f"Operations (apply in order):\n{operations_text}\n\n"
        f"After all operations, how many points does {query_entity} have?\n\n"
        "Respond with ONLY the final number."
    )


def build_problem(k: int, seed: int, probe_idx: int) -> dict[str, Any]:
    """wmf_am_frozen_v1.py L55-112: the same RNG calls in the same order.

    Each operation is kept as (text, entities whose value it changes, kind);
    the extra bookkeeping draws nothing from the RNG.
    """
    rng = random.Random(seed + k * 1000 + probe_idx)
    entities = rng.sample(list(ENTITY_NAMES), 3)
    state = {entity: rng.randint(5, 20) for entity in entities}
    initial_state = dict(state)
    ops: list[tuple[str, tuple[str, ...], str]] = []

    for _ in range(k):
        op_type = rng.choice(["add", "subtract", "transfer"])
        if op_type == "add":
            entity = rng.choice(entities)
            amount = rng.randint(1, 10)
            state[entity] += amount
            ops.append((f"{entity} gains {amount} points.", (entity,), "gain"))          # L71
        elif op_type == "subtract":
            entity = rng.choice(entities)
            amount = min(rng.randint(1, 5), state[entity] - 1)
            if amount > 0:
                state[entity] -= amount
                ops.append((f"{entity} loses {amount} points.", (entity,), "lose"))      # L77
            else:
                state[entity] += 1
                ops.append((f"{entity} gains 1 point.", (entity,), "gain_one"))          # L80
        else:
            giver, receiver = rng.sample(entities, 2)
            amount = min(rng.randint(1, 3), state[giver] - 1)
            if amount > 0:
                state[giver] -= amount
                state[receiver] += amount
                ops.append((f"{giver} gives {amount} points to {receiver}.", (giver, receiver), "transfer"))  # L87
            else:
                ops.append(("No transfer occurs this round.", (), "no_transfer"))        # L89

    query_entity = rng.choice(entities)
    return {
        "k": k, "seed": seed, "probe_idx": probe_idx,
        "initial_state": initial_state, "ops": ops, "query_entity": query_entity,
        "gold": state[query_entity],
        "n_relevant_ops": sum(query_entity in changed for _, changed, _ in ops),
    }


def is_degenerate(problem: dict[str, Any]) -> bool:
    """Q10: no operation changes the queried entity, or the answer equals its initial value."""
    return (problem["n_relevant_ops"] == 0
            or problem["gold"] == problem["initial_state"][problem["query_entity"]])


def main_problems(depths: tuple[int, ...]) -> list[dict[str, Any]]:
    """K x seed x 5 slots. A degenerate slot takes the next unused probe_idx (5, 6, ...)
    of its (K, seed) cell until a non-degenerate problem appears."""
    problems = []
    for k in depths:
        for seed in CORE_SEEDS:
            next_unused = PROBES_PER_SEED
            for slot in range(PROBES_PER_SEED):
                problem = build_problem(k, seed, slot)
                why = None
                if is_degenerate(problem):
                    why = "untouched" if problem["n_relevant_ops"] == 0 else "net_zero"
                while is_degenerate(problem):
                    problem = build_problem(k, seed, next_unused)
                    next_unused += 1
                problem.update(probe_idx_original=slot, redraw_reason=why)
                problems.append(problem)
    return problems


# Submitted scoring rule per block. The K-sweep script took the last integer
# (wmf_am_extended_k.py L63-65); it has no \boxed / "Answer:" priority.
MAIN_LEGACY_SCORING = {"A": "wmf_am_frozen_v1.parse_answer", "B": "last_int (wmf_am_extended_k.py L63-65)"}


def main_items(block: str, problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        make_item(
            f"{block}-k{p['k']}-s{p['seed']}-p{p['probe_idx']}", block, p["k"], p["seed"], p["probe_idx"],
            main_prompt(p["initial_state"], [text for text, _, _ in p["ops"]], p["query_entity"]),
            p["gold"], "int", p["n_relevant_ops"], p["initial_state"][p["query_entity"]],
            MAIN_LEGACY_SCORING[block],
            probe_idx_original=p["probe_idx_original"], redrawn=p["redraw_reason"] is not None,
            redraw_reason=p["redraw_reason"], query_entity=p["query_entity"],
        )
        for p in problems
    ]


# ── Block C: K=1 single-step arithmetic (wmf_am_last_op.py) ──────────────────

# wmf_am_last_op.py L51-88 (only the fields the prompt uses)
K1_ENTITIES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "James"]
K1_DOMAINS = {
    "points": {
        "init_template": "{entity} starts with {val} points.",
        "op_gain": "{entity} gains {delta} points.",
        "op_lose": "{entity} loses {delta} points.",
        "question": "How many points does {entity} have now?",
        "parse_hint": "Respond with ONLY the final number.",
    },
    "inventory": {
        "init_template": "{entity} has {val} items in their warehouse.",
        "op_gain": "{entity} adds {delta} items to their warehouse.",
        "op_lose": "{entity} removes {delta} items from their warehouse.",
        "question": "How many items does {entity} have in their warehouse now?",
        "parse_hint": "Respond with ONLY the final number.",
    },
    "accounts": {
        "init_template": "{entity}'s account balance is ${val}.",
        "op_gain": "{entity} deposits ${delta} into their account.",
        "op_lose": "{entity} withdraws ${delta} from their account.",
        "question": "What is {entity}'s account balance now?",
        "parse_hint": "Respond with ONLY the final number (no $ sign).",
    },
}
K1_RANGES = {"small": (1, 20), "medium": (20, 100), "large": (100, 1000)}   # L91-95
K1_SEED = 42            # L278
K1_PER_CELL = 10        # L274


def block_c() -> list[dict[str, Any]]:
    rng = random.Random(K1_SEED)                               # L208: one stream for all 90 items
    items = []
    for domain_name, domain in K1_DOMAINS.items():             # L211-213 loop order
        for range_name, (lo, hi) in K1_RANGES.items():
            for prob_idx in range(K1_PER_CELL):
                # build_last_op_problem, L126-150
                entity = rng.choice(K1_ENTITIES)
                init_val = rng.randint(lo, hi)
                delta = rng.randint(lo // 2 + 1, hi)
                is_gain = rng.choice([True, False])
                if is_gain:
                    correct = init_val + delta
                    op_line = domain["op_gain"].format(entity=entity, delta=delta)
                else:
                    if delta > init_val:
                        delta = rng.randint(1, init_val)
                    correct = init_val - delta
                    op_line = domain["op_lose"].format(entity=entity, delta=delta)
                init_line = domain["init_template"].format(entity=entity, val=init_val)
                question = domain["question"].format(entity=entity)
                prompt = f"{init_line}\n{op_line}\n{question}\n{domain['parse_hint']}"   # L147-150
                items.append(make_item(
                    f"C-{domain_name}-{range_name}-{prob_idx:02d}", "C", 1, K1_SEED, prob_idx,
                    prompt, correct, "int", 1, init_val, "wmf_am_last_op.parse_number",
                    domain=domain_name, numeric_range=range_name, query_entity=entity,
                ))
    return items


# ── Block D: yoked cancellation control (wmf_am_yoked_control.py) ────────────

YOKED_DEPTHS = (2, 4, 6, 8, 12)     # config.py L338 WMF_OPERATION_DEPTHS
YOKED_SEED = 42                     # wmf_am_yoked_control.py L245
YOKED_PER_DEPTH = 20                # L241


def block_d() -> list[dict[str, Any]]:
    rng = random.Random(YOKED_SEED)                            # L177: one stream for all 100 items
    items = []
    for k in YOKED_DEPTHS:
        # The odd-K branch (L121-166) is not copied: every submitted depth is even.
        assert k % 2 == 0
        for prob_idx in range(YOKED_PER_DEPTH):
            # build_yoked_control_problem, L108-170. Entity pool L47-62 has the
            # same 14 names in the same order as ENTITY_NAMES.
            entities = rng.sample(list(ENTITY_NAMES), 3)
            state = {e: rng.randint(5, 20) for e in entities}
            ops: list[tuple[str, tuple[str, ...]]] = []
            for _ in range(k // 2):
                # _generate_cancelling_pair, L70-92
                op_type = rng.choice(["gain_loss", "loss_gain", "transfer"])
                if op_type == "gain_loss":
                    e = rng.choice(entities)
                    amount = rng.randint(1, 10)
                    ops += [(f"{e} gains {amount} points.", (e,)), (f"{e} loses {amount} points.", (e,))]
                elif op_type == "loss_gain":
                    e = rng.choice(entities)
                    amount = rng.randint(1, 5)
                    ops += [(f"{e} loses {amount} points.", (e,)), (f"{e} gains {amount} points.", (e,))]
                else:
                    giver, receiver = rng.sample(entities, 2)
                    amount = rng.randint(1, 3)
                    ops += [(f"{giver} gives {amount} points to {receiver}.", (giver, receiver)),
                            (f"{receiver} gives {amount} points to {giver}.", (giver, receiver))]
            query_entity = rng.choice(entities)
            items.append(make_item(
                f"D-k{k}-{prob_idx:02d}", "D", k, YOKED_SEED, prob_idx,
                main_prompt(state, [text for text, _ in ops], query_entity),
                state[query_entity], "int",
                sum(query_entity in changed for _, changed in ops), state[query_entity],
                "last_int (wmf_am_yoked_control.py L207-209)", query_entity=query_entity,
            ))
    return items


# ── Block E: non-arithmetic direct assignment (wmf_am_nonarithmetic.py) ──────

# L40-48
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink",
          "white", "black", "gray", "brown", "cyan"]
LOCATIONS = ["kitchen", "garden", "library", "garage", "bedroom", "office",
             "basement", "attic", "patio", "workshop", "hallway", "studio"]
STATUSES = ["active", "paused", "reviewing", "completed", "pending",
            "archived", "drafting", "approved", "rejected", "waiting",
            "processing", "finalized"]
# L50-69
NONARITH_DOMAINS = {
    "color": {"values": COLORS, "verb": "changes color to",
              "question": "What color is {entity} now?", "attribute": "color"},
    "location": {"values": LOCATIONS, "verb": "moves to the",
                 "question": "Where is {entity} now?", "attribute": "location"},
    "status": {"values": STATUSES, "verb": "changes status to",
               "question": "What is {entity}'s current status?", "attribute": "status"},
}
# L71-72
NONARITH_ENTITIES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank",
                     "Grace", "Henry", "Iris", "James", "Kate", "Leo"]
NONARITH_SEED = 42      # L215
NONARITH_DEPTHS = (3, 5, 7)   # L169
NONARITH_PER_CELL = 10  # L211


def block_e() -> list[dict[str, Any]]:
    rng = random.Random(NONARITH_SEED)                         # L167: one stream for all 90 items
    items = []
    for domain_name, domain in NONARITH_DOMAINS.items():       # L171-173 loop order
        values = domain["values"]
        for k in NONARITH_DEPTHS:
            for prob_idx in range(NONARITH_PER_CELL):
                # build_nonarith_problem, L85-127
                entities = rng.sample(NONARITH_ENTITIES, 3)
                initial_vals = rng.sample(values, 3)
                state = dict(zip(entities, initial_vals))
                initial_state = dict(state)
                operations = []
                updated = []
                for _ in range(k):
                    entity = rng.choice(entities)
                    current = state[entity]
                    candidates = [v for v in values if v != current]
                    new_val = rng.choice(candidates)
                    state[entity] = new_val
                    operations.append(f"{entity} {domain['verb']} {new_val}.")
                    updated.append(entity)
                query_entity = rng.choice(entities)
                correct = state[query_entity]

                if domain_name == "color":
                    init_str = ", ".join(f"{e} is {v}" for e, v in zip(entities, initial_vals))
                elif domain_name == "location":
                    init_str = ", ".join(f"{e} is in the {v}" for e, v in zip(entities, initial_vals))
                else:
                    init_str = ", ".join(f"{e}'s status is {v}" for e, v in zip(entities, initial_vals))
                ops_str = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(operations))
                question = domain["question"].format(entity=query_entity)
                # L117-127
                prompt = f"""You will track a sequence of {domain['attribute']} changes. Pay careful attention to each update.

Initial state:
{init_str}

Updates (apply in order):
{ops_str}

{question}

Respond with ONLY the final {domain['attribute']} (one word)."""
                items.append(make_item(
                    f"E-{domain_name}-k{k}-{prob_idx:02d}", "E", k, NONARITH_SEED, prob_idx,
                    prompt, correct, "word", updated.count(query_entity), initial_state[query_entity],
                    "wmf_am_nonarithmetic.evaluate_response",
                    domain=domain_name, query_entity=query_entity, vocab=values,
                ))
    return items


# ── Block F: non-arithmetic cumulative probe (wmf_am_cumulative_logical.py) ──

LOGICAL_ENTITIES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]          # L42
PERMISSIONS = ["read", "write", "execute", "admin"]                             # L45
MEETING_TYPES = ["morning standup", "design review", "client call",
                 "team lunch", "project sync", "budget review"]                 # L48-49
INVENTORY_ITEMS = ["key", "map", "torch", "rope", "compass", "shield",
                   "potion", "scroll", "gem", "ring"]                           # L52-53
LOGICAL_SEED = 42           # L237
LOGICAL_DEPTHS = (3, 5, 7)  # L238
LOGICAL_PER_CELL = 10       # L236


def build_permission_problem(k_ops: int, rng: random.Random) -> dict[str, Any]:
    # L58-79
    entity = rng.choice(LOGICAL_ENTITIES)
    start_perms = sorted(rng.sample(PERMISSIONS, rng.randint(0, 2)))
    current = set(start_perms)
    ops = []
    n_changing = 0
    for _ in range(k_ops):
        action = rng.choice(["grant", "revoke"])
        perm = rng.choice(PERMISSIONS)
        if action == "grant":
            ops.append(f"{entity} is granted {perm} access.")
            n_changing += perm not in current
            current.add(perm)
        else:
            ops.append(f"{entity}'s {perm} access is revoked.")
            n_changing += perm in current
            current.discard(perm)

    start_str = ", ".join(start_perms) if start_perms else "no permissions"
    correct = ", ".join(sorted(current)) if current else "no permissions"

    prompt = f"{entity} starts with {start_str}.\n"
    for op in ops:
        prompt += f"{op}\n"
    prompt += f"\nWhat permissions does {entity} currently have? List them in alphabetical order, separated by commas. If none, say 'no permissions'."
    prompt += "\n\nRespond with ONLY the answer."
    return {"prompt": prompt, "gold": sorted(current), "gold_text": correct, "answer_type": "set",
            "entity": entity, "initial_value": start_perms, "n_relevant_ops": n_changing, "vocab": PERMISSIONS}


def build_schedule_problem(k_ops: int, rng: random.Random) -> dict[str, Any]:
    entity = rng.choice(LOGICAL_ENTITIES)                      # L86
    # L87-101 build a first version that the script then discards (L104-107).
    # Its draws are repeated here only so the RNG stream stays the same.
    rng.randint(1, 3)
    for _ in range(k_ops):
        rng.choice(["add", "cancel"])
        rng.choice(MEETING_TYPES)
    # L105-126: the version actually used
    start_count = rng.randint(2, 5)
    current = start_count
    ops = []
    for _ in range(k_ops):
        action = rng.choice(["add", "cancel"])
        meeting = rng.choice(MEETING_TYPES)
        if action == "add":
            ops.append(f"A {meeting} is added to {entity}'s schedule.")
            current += 1
        else:
            if current > 0:
                ops.append(f"The {meeting} is cancelled from {entity}'s schedule.")
                current -= 1
            else:
                ops.append(f"A {meeting} is added to {entity}'s schedule.")
                current += 1

    prompt = f"{entity} starts the day with {start_count} meetings.\n"
    for op in ops:
        prompt += f"{op}\n"
    prompt += f"\nHow many meetings does {entity} have now?\n"
    prompt += "\nRespond with ONLY the final number."
    # Every schedule operation changes the count by one, so n_relevant_ops = K.
    return {"prompt": prompt, "gold": current, "gold_text": str(current), "answer_type": "int",
            "entity": entity, "initial_value": start_count, "n_relevant_ops": k_ops, "vocab": None}


def build_inventory_problem(k_ops: int, rng: random.Random) -> dict[str, Any]:
    # L133-153
    entity = rng.choice(LOGICAL_ENTITIES)
    start_items = sorted(rng.sample(INVENTORY_ITEMS, rng.randint(1, 3)))
    current = set(start_items)
    ops = []
    n_changing = 0
    for _ in range(k_ops):
        action = rng.choice(["picks up", "drops"])
        item = rng.choice(INVENTORY_ITEMS)
        if action == "picks up":
            ops.append(f"{entity} picks up the {item}.")
            n_changing += item not in current
            current.add(item)
        else:
            ops.append(f"{entity} drops the {item}.")
            n_changing += item in current
            current.discard(item)

    correct = ", ".join(sorted(current)) if current else "nothing"

    prompt = f"{entity} starts with: {', '.join(start_items)}.\n"
    for op in ops:
        prompt += f"{op}\n"
    prompt += f"\nWhat items does {entity} currently have? List them in alphabetical order, separated by commas. If none, say 'nothing'."
    prompt += "\n\nRespond with ONLY the answer."
    return {"prompt": prompt, "gold": sorted(current), "gold_text": correct, "answer_type": "set",
            "entity": entity, "initial_value": start_items, "n_relevant_ops": n_changing, "vocab": INVENTORY_ITEMS}


# L158-162
LOGICAL_BUILDERS = {
    "permissions": build_permission_problem,
    "schedule": build_schedule_problem,
    "inventory": build_inventory_problem,
}


def block_f() -> list[dict[str, Any]]:
    items = []
    for domain_name, builder in LOGICAL_BUILDERS.items():      # L201-205 loop order
        for k in LOGICAL_DEPTHS:
            for trial in range(LOGICAL_PER_CELL):
                # L204: one RNG per (K, trial); the domain is not part of the seed.
                p = builder(k, random.Random(LOGICAL_SEED * 1000 + k * 100 + trial))
                items.append(make_item(
                    f"F-{domain_name}-k{k}-{trial:02d}", "F", k, LOGICAL_SEED, trial,
                    p["prompt"], p["gold"], p["answer_type"], p["n_relevant_ops"], p["initial_value"],
                    "wmf_am_cumulative_logical.score_response",
                    domain=domain_name, query_entity=p["entity"], gold_text=p["gold_text"], vocab=p["vocab"],
                ))
    return items


# ── Block G: paraphrase templates (wmf_am_paraphrase.py) ─────────────────────

PARAPHRASE_TEMPLATES = ("formal", "casual", "minimal", "verbose")   # L149, without "original"
PARAPHRASE_ENTITIES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]   # L37
PARAPHRASE_SEED = 42          # L209
PARAPHRASE_PER_DEPTH = 10     # L205
# Option A (2026-10-03). Existing main-prompt sentence, new to
# the submitted G/minimal template; appended only by an explicit CLI choice.
MINIMAL_FINAL_NUMBER = "Respond with ONLY the final number."

# wmf_am_paraphrase.py L130: the per-step note of the "verbose" template.
VERBOSE_NOTE = " (After this step, update {e}'s running total accordingly.)"
# NEW TEXT, used only with --paraphrase-mode rewrite_A (Q17).
# 1. A transfer line inside the paraphrase templates. The wording is block A's own
#    (wmf_am_frozen_v1.py L87, final period dropped); the submitted paraphrase script has no transfers.
TRANSFER_LINE_NEW = "{giver} gives {amount} points to {receiver}"
# 2. The verbose per-step note for a transfer, which changes two totals. Not in any submitted script.
VERBOSE_NOTE_TRANSFER_NEW = " (After this step, update {giver}'s and {receiver}'s running totals accordingly.)"


def format_paraphrase(template: str, state_str: str, ops_lines: list[str],
                      verbose_notes: list[str], query: str) -> str:
    """wmf_am_paraphrase.format_prompt, L95-143. `ops_lines` have no final period,
    as in L79; `verbose_notes` is the text appended after each step in "verbose"."""
    if template == "formal":
        ops_str = "\n".join(f"  Step {i+1}: {op}." for i, op in enumerate(ops_lines))
        # L97-107
        return f"""The following is a sequential state-tracking exercise. Please process each modification in the order given and determine the resulting value.

Starting values:
{state_str}

Modifications to apply sequentially:
{ops_str}

Question: What is the final point total for {query} after all modifications have been applied?

Please provide only the numerical answer."""

    if template == "casual":
        ops_str = "\n".join(f"- {op}" for op in ops_lines)
        # L111-120
        return f"""Hey, can you help me keep track of some scores?

So we start with: {state_str}

Then these things happen one after another:
{ops_str}

So after all that, how many points does {query} end up with?

Just tell me the number."""

    if template == "minimal":
        ops_str = "; ".join(ops_lines)
        # L124-126
        return f"""Start: {state_str}
Changes: {ops_str}
{query}'s final points = ?"""

    if template == "verbose":
        ops_str = "\n".join(
            f"  {i+1}. {op}.{note}" for i, (op, note) in enumerate(zip(ops_lines, verbose_notes))
        )
        # L133-143
        return f"""In this task, you need to carefully track point totals for multiple people as they change over time. Read the initial state, then process each operation one by one in the exact order listed. Each operation either adds points to or subtracts points from one person's total. You must keep a mental running total for each person.

Here are the initial point totals for each person:
{state_str}

Now, apply the following operations one at a time, in order. After each operation, mentally update the running total for the affected person:
{ops_str}

Now that you have processed all {len(ops_lines)} operations, please tell me: what is {query}'s final point total?

Important: respond with ONLY the final number, nothing else."""

    raise ValueError(f"Unknown template: {template}")


def block_g_rewrite(a_problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Q17 variant: block A's 60 items re-rendered in the four paraphrase templates.

    The operation wording is block A's (wmf_am_frozen_v1.py L71-87) with the final
    period dropped. The submitted paraphrase generator has gain and lose only,
    so a transfer brings text that is new inside these templates:
      TRANSFER_LINE_NEW          in all four templates
      VERBOSE_NOTE_TRANSFER_NEW  in "verbose" only
    Items containing a transfer carry new_text = true and list these strings.
    """
    items = []
    for template in PARAPHRASE_TEMPLATES:
        for p in a_problems:
            state_str = ", ".join(f"{e}: {v} points" for e, v in p["initial_state"].items())   # L78
            ops_lines, notes = [], []
            for text, changed, kind in p["ops"]:
                # Block A's 60 items contain no "No transfer occurs this round." and no
                # "gains 1 point." line, so no paraphrase wording was written for them.
                assert kind in ("gain", "lose", "transfer"), f"no paraphrase wording for {kind!r}"
                ops_lines.append(text.removesuffix("."))
                if kind == "transfer":
                    notes.append(VERBOSE_NOTE_TRANSFER_NEW.format(giver=changed[0], receiver=changed[1]))
                else:
                    notes.append(VERBOSE_NOTE.format(e=changed[0]))
            new_strings = []
            if any(kind == "transfer" for _, _, kind in p["ops"]):
                new_strings = [TRANSFER_LINE_NEW] + ([VERBOSE_NOTE_TRANSFER_NEW.strip()] if template == "verbose" else [])
            item = make_item(
                f"G-{template}-k{p['k']}-s{p['seed']}-p{p['probe_idx']}", "G", p["k"], p["seed"], p["probe_idx"],
                format_paraphrase(template, state_str, ops_lines, notes, p["query_entity"]),
                p["gold"], "int", p["n_relevant_ops"], p["initial_state"][p["query_entity"]],
                "first_int (wmf_am_paraphrase.py L175-177)",
                template=template, source_item_id=f"A-k{p['k']}-s{p['seed']}-p{p['probe_idx']}",
                query_entity=p["query_entity"], new_text_strings=new_strings,
            )
            item["new_text"] = bool(new_strings)
            items.append(item)
    return items


def block_g_original(minimal_final_number: bool = False) -> list[dict[str, Any]]:
    """Submitted paraphrase generator: its own 30 problems, each in all five templates."""
    rng = random.Random(PARAPHRASE_SEED)                       # L155: one stream for all 30 problems
    items = []
    for k in (3, 5, 7):                                        # L157
        for prob_idx in range(PARAPHRASE_PER_DEPTH):
            # build_problem_data, L42-59: gain/lose only, no floor, answers can be negative
            entities = rng.sample(PARAPHRASE_ENTITIES, 3)
            state = {e: rng.randint(5, 20) for e in entities}
            initial_state = dict(state)
            operations = []
            for _ in range(k):
                entity = rng.choice(entities)
                op_type = rng.choice(["gain", "lose"])
                amount = rng.randint(1, 8)
                if op_type == "gain":
                    state[entity] += amount
                    operations.append((entity, "gains", amount))
                else:
                    state[entity] -= amount
                    operations.append((entity, "loses", amount))
            query_entity = rng.choice(entities)

            state_str = ", ".join(f"{e}: {initial_state[e]} points" for e in entities)   # L78
            ops_lines = [f"{e} {verb} {amt} points" for e, verb, amt in operations]      # L79
            notes = [VERBOSE_NOTE.format(e=e) for e, _, _ in operations]
            for template in ("original",) + PARAPHRASE_TEMPLATES:                        # L149 order
                if template == "original":
                    prompt = main_prompt(initial_state, [f"{op}." for op in ops_lines], query_entity)  # L82-93
                else:
                    prompt = format_paraphrase(template, state_str, ops_lines, notes, query_entity)
                item = make_item(
                    f"G-orig-{template}-k{k}-{prob_idx:02d}", "G", k, PARAPHRASE_SEED, prob_idx,
                    prompt, state[query_entity], "int",
                    sum(e == query_entity for e, _, _ in operations), initial_state[query_entity],
                    "first_int (wmf_am_paraphrase.py L175-177)",
                    template=template, query_entity=query_entity,
                )
                if template == "minimal" and minimal_final_number:
                    item["prompt"] += "\n\n" + MINIMAL_FINAL_NUMBER
                    item["new_text"] = True
                    item["new_text_strings"] = [MINIMAL_FINAL_NUMBER]
                items.append(item)
    return items


# ── Blocks H and I: chat and CoT wrappers (wmf_am_template_harmonization.py) ─

# L69-73
CHAT_SYSTEM = (
    "You are a precise arithmetic assistant. "
    "You track numerical state changes and report final values. "
    "Always respond with only the requested number, no explanation."
)


def wrapper_items(block: str, a_problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for p in a_problems:
        initial_state, query_entity = p["initial_state"], p["query_entity"]
        state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())         # L55
        ops_str = "\n".join(f"  {i+1}. {op}" for i, (op, _, _) in enumerate(p["ops"]))      # L56
        if block == "H":
            system = CHAT_SYSTEM
            # L74-80
            user = (
                f"Track the following point updates carefully.\n\n"
                f"Initial state:\n{state_str}\n\n"
                f"Operations (apply in order):\n{ops_str}\n\n"
                f"Question: After all operations, how many points does {query_entity} have?\n\n"
                "Answer with ONLY the number."
            )
            answer_type, legacy = "int", "first_int_unsigned (wmf_am_template_harmonization.py L127-128)"
        else:
            system = None
            # L83-90
            user = (
                "You will track a sequence of point updates. "
                "You cannot refer back to the initial state after reading it once.\n\n"
                f"Initial state:\n{state_str}\n\n"
                f"Operations (apply in order):\n{ops_str}\n\n"
                f"After all operations, how many points does {query_entity} have?\n\n"
                "Think step by step, then give your final answer as a single number on the last line."
            )
            answer_type, legacy = "int_last", "last_int_unsigned (wmf_am_template_harmonization.py L124-125)"
        items.append(make_item(
            f"{block}-k{p['k']}-s{p['seed']}-p{p['probe_idx']}", block, p["k"], p["seed"], p["probe_idx"],
            user, p["gold"], answer_type, p["n_relevant_ops"], initial_state[query_entity], legacy,
            system=system, template="chat" if block == "H" else "cot",
            source_item_id=f"A-k{p['k']}-s{p['seed']}-p{p['probe_idx']}", query_entity=query_entity,
        ))
    return items


# ── Assemble, write, report ──────────────────────────────────────────────────

def build_bank(cot: bool, paraphrase_mode: str, max_tokens_int: int,
               max_tokens_block: dict[str, int], minimal_final_number: bool = False) -> list[dict[str, Any]]:
    if minimal_final_number and paraphrase_mode != "original":
        raise ValueError("--minimal-final-number requires --paraphrase-mode original")
    a_problems = main_problems(A_DEPTHS)
    b_problems = main_problems(B_DEPTHS)
    rng_seeds = [p["seed"] + p["k"] * 1000 + p["probe_idx"] for p in a_problems + b_problems]
    # seed + K*1000 + probe_idx is not injective in general; a clash would put the same problem in two cells.
    assert len(set(rng_seeds)) == len(rng_seeds), "two A/B items share one RNG seed"

    items = main_items("A", a_problems) + main_items("B", b_problems)
    items += block_c() + block_d() + block_e() + block_f()
    items += block_g_rewrite(a_problems) if paraphrase_mode == "rewrite_A" else block_g_original(minimal_final_number)
    items += wrapper_items("H", a_problems)
    if cot:
        items += wrapper_items("I", a_problems)

    for item in items:
        item["max_tokens"] = max_tokens_int if item["answer_type"] == "int" else FIXED_MAX_TOKENS[item["answer_type"]]
    unknown = sorted(set(max_tokens_block) - {item["block"] for item in items})
    if unknown:
        raise SystemExit(f"--max-tokens-block names block(s) not in this bank: {unknown}")
    for item in items:
        if item["block"] in max_tokens_block:
            item["max_tokens"] = max_tokens_block[item["block"]]
    ids = [item["item_id"] for item in items]
    assert len(set(ids)) == len(ids), "duplicate item_id"
    return items


def count_table(items: list[dict[str, Any]]) -> str:
    lines = [f"{'block':<6}{'items':>6}  {'answer_type':<22}{'max_tokens':<12}{'new_text':>8}"]
    for block in sorted({item["block"] for item in items}):
        rows = [item for item in items if item["block"] == block]
        types = ",".join(f"{t}:{n}" for t, n in sorted(Counter(r["answer_type"] for r in rows).items()))
        tokens = ",".join(str(t) for t in sorted({r["max_tokens"] for r in rows}))
        lines.append(f"{block:<6}{len(rows):>6}  {types:<22}{tokens:<12}{sum(r['new_text'] for r in rows):>8}")
    lines.append(f"{'total':<6}{len(items):>6}{'':<36}{sum(item['new_text'] for item in items):>8}")
    return "\n".join(lines)


def main_block_table(items: list[dict[str, Any]], block: str) -> str:
    """Per K: Q10 redraws, n_relevant_ops distribution, answer range (blocks A and B)."""
    lines = [f"block {block} | K | items | redrawn (untouched + net_zero) | highest probe_idx | "
             f"n_relevant_ops min/median/max | n_relevant_ops value:count | answer min..max"]
    for k in sorted({item["k"] for item in items if item["block"] == block}):
        rows = [item for item in items if item["block"] == block and item["k"] == k]
        reasons = Counter(r["redraw_reason"] for r in rows)
        rel = sorted(r["n_relevant_ops"] for r in rows)
        hist = " ".join(f"{value}:{count}" for value, count in sorted(Counter(rel).items()))
        golds = [r["gold"] for r in rows]
        lines.append(
            f"block {block} | {k} | {len(rows)} | {sum(r['redrawn'] for r in rows)} "
            f"({reasons['untouched']} + {reasons['net_zero']}) | {max(r['probe_idx'] for r in rows)} | "
            f"{rel[0]}/{(rel[len(rel) // 2 - 1] + rel[len(rel) // 2]) / 2:g}/{rel[-1]} | {hist} | "
            f"{min(golds)}..{max(golds)}"
        )
    return "\n".join(lines)


def unchanged_answer_table(items: list[dict[str, Any]]) -> str:
    """Blocks outside Q10: items the queried entity's answer does not move in."""
    lines = ["block | items | n_relevant_ops == 0 | gold == initial_value"]
    for block in sorted({item["block"] for item in items} - {"A", "B"}):
        rows = [item for item in items if item["block"] == block]
        lines.append(f"{block} | {len(rows)} | {sum(r['n_relevant_ops'] == 0 for r in rows)} | "
                     f"{sum(r['gold'] == r['initial_value'] for r in rows)}")
    return "\n".join(lines)


def parse_block_override(text: str) -> tuple[str, int] | None:
    """'G=1024' -> ('G', 1024); 'none' -> None."""
    if text == "none":
        return None
    block, sep, value = text.partition("=")
    if not sep or len(block) != 1 or not block.isupper() or not value.isdigit() or int(value) <= 0:
        raise argparse.ArgumentTypeError(f"expected BLOCK=N (e.g. G=1024) or none, got {text!r}")
    return block, int(value)


def block_overrides(values: list[tuple[str, int] | None]) -> dict[str, int]:
    if None in values:
        if len(values) > 1:
            raise SystemExit("--max-tokens-block none cannot be combined with BLOCK=N")
        return {}
    blocks = [block for block, _ in values]
    if len(set(blocks)) != len(blocks):
        raise SystemExit(f"--max-tokens-block gives one block twice: {blocks}")
    return dict(values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cot = parser.add_mutually_exclusive_group(required=True)
    cot.add_argument("--include-cot", dest="cot", action="store_true", help="block I (Q4)")
    cot.add_argument("--no-cot", dest="cot", action="store_false")
    parser.add_argument("--paraphrase-mode", choices=("rewrite_A", "original"), required=True, help="block G (Q17)")
    minimal = parser.add_mutually_exclusive_group(required=True)
    minimal.add_argument("--minimal-final-number", dest="minimal_final_number", action="store_true",
                         help="append the number-only instruction to original G/minimal")
    minimal.add_argument("--no-minimal-final-number", dest="minimal_final_number", action="store_false",
                         help="keep the submitted minimal template unchanged")
    parser.add_argument("--max-tokens-int", type=int, required=True, help="max_tokens of every 'int' item (Q9)")
    parser.add_argument("--max-tokens-block", type=parse_block_override, action="append", required=True,
                        metavar="BLOCK=N|none",
                        help="set max_tokens of every item in BLOCK to N, after the rules above; repeatable; "
                             "'none' = no override (must be given explicitly)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.minimal_final_number and args.paraphrase_mode != "original":
        parser.error("--minimal-final-number requires --paraphrase-mode original")
    args.max_tokens_block = block_overrides(args.max_tokens_block)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    items = build_bank(args.cot, args.paraphrase_mode, args.max_tokens_int,
                       args.max_tokens_block, args.minimal_final_number)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"CANDIDATE bank (not frozen): {args.out}")
    print(f"switches: cot={args.cot} paraphrase_mode={args.paraphrase_mode} "
          f"minimal_final_number={args.minimal_final_number} "
          f"max_tokens_int={args.max_tokens_int} max_tokens_block={args.max_tokens_block or 'none'}\n")
    print(count_table(items))
    print()
    print(main_block_table(items, "A"))
    print(main_block_table(items, "B"))
    print()
    print(unchanged_answer_table(items))

    a_rows = [item for item in items if item["block"] == "A"]
    g_rows = [item for item in items if item["block"] == "G"]
    if args.paraphrase_mode == "rewrite_A":
        print(f"\nblock A items containing a transfer (new_text in each of the four G templates): "
              f"{sum(r['new_text'] for r in g_rows if r['template'] == 'formal')} of {len(a_rows)}")

    from_int = sum(item["answer_type"] == "int" and item["block"] not in args.max_tokens_block for item in items)
    overridden = sum(item["block"] in args.max_tokens_block for item in items)
    print("\nother switch values (counts only; --max-tokens-int and --max-tokens-block change no count; "
          f"here --max-tokens-int sets max_tokens on {from_int} items, --max-tokens-block on {overridden}):")
    print(f"{'cot':<6}{'paraphrase':<12}{'G':>5}{'I':>5}{'total':>7}{'new_text':>9}")
    for cot in (True, False):
        for mode in ("rewrite_A", "original"):
            variant = build_bank(cot, mode, args.max_tokens_int, {},
                                 args.minimal_final_number if mode == "original" else False)
            n = Counter(item["block"] for item in variant)
            print(f"{'yes' if cot else 'no':<6}{mode:<12}"
                  f"{n['G']:>5}{n['I']:>5}{len(variant):>7}{sum(i['new_text'] for i in variant):>9}")


if __name__ == "__main__":
    main()
