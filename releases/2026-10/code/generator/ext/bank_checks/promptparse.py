#!/usr/bin/env python3
"""Independent prompt parser and simulator for ext_bank.jsonl (blocks K and L).

Everything here is derived from the PROMPT TEXT only. Nothing is imported from
the generators and no bank field other than `prompt` is read by parse_*().
Any line that does not match the expected grammar raises (fail fast).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BANK = Path(__file__).resolve().parents[4] / "data" / "item_bank" / "ext_bank.jsonl"

NAME = r"[A-Z]\d+"
STATE_PAIR = re.compile(rf"^({NAME})=(-?\d+)$")
GAIN_LOSS = re.compile(rf"^({NAME}) (gains|loses) (\d+) (point|points)\.$")
TRANSFER = re.compile(rf"^({NAME}) gives (\d+) (point|points) to ({NAME})\.$")
K_SLOT = re.compile(r"^  (\d{2}) \| active=([01]) \| (.+)$")
L_SLOT = re.compile(r"^  (\d{2}) \| (.+)$")
K_HEADER = re.compile(
    r"^Track integer point states through the displayed sequence\. There are exactly (\w+) slots\. "
    r"Apply each slot exactly once and in order only when active=1; ignore active=0 slots\.$"
)
L_HEADER = re.compile(
    r"^Track integer point states through the displayed sequence\. Apply all (\d+) operations "
    r"exactly once and in order\.$"
)
# Declared state after the rebuild of 2026-10-04: the question names the slot only.
# A prompt that still carries "(the q-th active update to E0)" does not match and raises.
K_QUESTION = re.compile(
    rf"^What was ({NAME}) immediately after slot (\d{{2}})\? Respond with ONLY the number\.$"
)
L_QUESTION = re.compile(
    rf"^Return the final values for \[({NAME}(?:, {NAME})*)\] in exactly this order\. "
    r"Respond with ONLY the requested values, separated by commas\.$"
)
# Final bank: the 40 single-target items ask for one value, in the singular.
L_QUESTION_M1 = re.compile(rf"^Return the final value for \[({NAME})\]\. Respond with ONLY the number\.$")
NUMBER_WORDS = {"six": 6, "eight": 8, "nine": 9, "twelve": 12}


def load_items(path: Path = BANK) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sections(prompt: str) -> tuple[str, str, list[str], str]:
    parts = prompt.split("\n\n")
    if len(parts) != 4:
        raise ValueError(f"expected 4 blank-line separated sections, found {len(parts)}")
    header, state_line, slots_block, question = parts
    slot_lines = slots_block.split("\n")
    if slot_lines[0] != "Operation slots:":
        raise ValueError(f"unexpected slots heading: {slot_lines[0]!r}")
    return header, state_line, slot_lines[1:], question


def _initial_state(state_line: str) -> dict[str, int]:
    prefix = "Initial state: "
    if not state_line.startswith(prefix):
        raise ValueError(f"unexpected state line: {state_line!r}")
    state: dict[str, int] = {}
    for pair in state_line[len(prefix):].split(", "):
        match = STATE_PAIR.match(pair)
        if not match:
            raise ValueError(f"unparsed state pair: {pair!r}")
        if match[1] in state:
            raise ValueError(f"entity listed twice: {match[1]}")
        state[match[1]] = int(match[2])
    return state


def _operation(body: str) -> dict:
    """One operation as {'deltas': {entity: signed amount}, 'kind', 'amount', 'text'}."""
    match = GAIN_LOSS.match(body)
    if match:
        entity, verb, amount, noun = match[1], match[2], int(match[3]), match[4]
        sign = 1 if verb == "gains" else -1
        op = {"kind": "gain" if sign > 0 else "loss", "amount": amount, "deltas": {entity: sign * amount}}
    else:
        match = TRANSFER.match(body)
        if not match:
            raise ValueError(f"unparsed operation: {body!r}")
        giver, amount, noun, receiver = match[1], int(match[2]), match[3], match[4]
        if giver == receiver:
            raise ValueError(f"transfer to self: {body!r}")
        op = {"kind": "transfer", "amount": amount, "deltas": {giver: -amount, receiver: amount}}
    if (amount == 1) != (noun == "point"):
        raise ValueError(f"singular/plural mismatch: {body!r}")
    op["text"] = body
    return op


def _slots(slot_lines: list[str], with_active: bool) -> list[dict]:
    slots = []
    for position, line in enumerate(slot_lines, start=1):
        if with_active:
            match = K_SLOT.match(line)
            if not match:
                raise ValueError(f"unparsed K slot line: {line!r}")
            number, active, body = int(match[1]), int(match[2]), match[3]
        else:
            match = L_SLOT.match(line)
            if not match or "active=" in line:
                raise ValueError(f"unparsed L slot line: {line!r}")
            number, active, body = int(match[1]), 1, match[2]
        if number != position:
            raise ValueError(f"slot numbers not consecutive: line {position} is numbered {number}")
        slots.append({"slot": number, "active": active, "line": line, **_operation(body)})
    return slots


def simulate(initial: dict[str, int], slots: list[dict]) -> list[dict[str, int]]:
    """State after each slot (index 0 = initial state). Inactive slots change nothing."""
    state = dict(initial)
    history = [dict(state)]
    for slot in slots:
        if slot["active"] == 1:
            for entity, delta in slot["deltas"].items():
                state[entity] += delta          # KeyError on an entity absent from the initial state
        history.append(dict(state))
    return history


def parse_k(prompt: str) -> dict:
    header, state_line, slot_lines, question = _sections(prompt)
    head = K_HEADER.match(header)
    if not head:
        raise ValueError(f"unexpected K header: {header!r}")
    quest = K_QUESTION.match(question)
    if not quest:
        raise ValueError(f"unexpected K question: {question!r}")
    initial = _initial_state(state_line)
    slots = _slots(slot_lines, with_active=True)
    entity, named = quest[1], int(quest[2])
    if not 1 <= named <= len(slots):
        raise ValueError(f"the question names slot {named:02d}, which is not displayed")
    # active updates that touch the queried entity, in order; rank = how many of them lie at or before the named slot
    active_updates = [s["slot"] for s in slots if s["active"] == 1 and entity in s["deltas"]]
    return {
        "header": header, "declared_slots": NUMBER_WORDS[head[1]], "initial": initial, "slots": slots,
        "history": simulate(initial, slots), "question": question,
        "entity": entity, "named_slot": named, "active_updates": active_updates,
        "rank": sum(slot <= named for slot in active_updates),
        "body": prompt[: len(prompt) - len(question)],        # everything up to the question sentence
    }


def parse_l(prompt: str) -> dict:
    header, state_line, slot_lines, question = _sections(prompt)
    head = L_HEADER.match(header)
    if not head:
        raise ValueError(f"unexpected L header: {header!r}")
    quest = L_QUESTION.match(question) or L_QUESTION_M1.match(question)
    if not quest:
        raise ValueError(f"unexpected L question: {question!r}")
    initial = _initial_state(state_line)
    slots = _slots(slot_lines, with_active=False)
    targets = quest[1].split(", ")
    if len(set(targets)) != len(targets):
        raise ValueError("a target is listed twice")
    return {
        "header": header, "declared_slots": int(head[1]), "initial": initial, "slots": slots,
        "history": simulate(initial, slots), "question": question, "targets": targets,
        "state_line": state_line,
    }
