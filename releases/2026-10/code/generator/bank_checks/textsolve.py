#!/usr/bin/env python3
"""Independent solver: reads the PROMPT TEXT only and recomputes the answer.

No field written by build_item_bank.py is used here (not block, k, n_relevant_ops,
initial_value, query_entity, ...). The prompt shape is recognised from its own
first words. Each solver returns a dict:
    kind       prompt shape
    answer     recomputed answer (int, str, or sorted list)
    initial    initial value of the queried thing, read from the prompt
    n_ops      number of operation lines in the prompt
    n_touch    number of operations that change the queried thing
    op_kinds   Counter of operation line kinds (points prompts only)
Anything the grammar below does not cover raises ValueError (fail fast).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BANK = HERE.parents[2] / "data" / "item_bank" / "item_bank.jsonl"

NAME = r"[A-Z][a-z]+"

# ── points operations (blocks A, B, D, G, H, I) ──────────────────────────────
RE_GAIN = re.compile(rf"^({NAME}) gains (\d+) points?$")
RE_LOSE = re.compile(rf"^({NAME}) loses (\d+) points$")
RE_GIVE = re.compile(rf"^({NAME}) gives (\d+) points to ({NAME})$")
NO_TRANSFER = "No transfer occurs this round"


def parse_state(text: str) -> dict[str, int]:
    state = {}
    for part in text.split(", "):
        m = re.fullmatch(rf"({NAME}): (\d+) points", part)
        if not m:
            raise ValueError(f"bad state entry {part!r}")
        if m[1] in state:
            raise ValueError(f"entity listed twice: {m[1]}")
        state[m[1]] = int(m[2])
    if len(state) != 3:
        raise ValueError(f"expected 3 entities, got {len(state)}")
    return state


def apply_points(initial: dict[str, int], ops: list[str], query: str, kind: str) -> dict:
    """`ops` are operation sentences WITHOUT the final period."""
    if query not in initial:
        raise ValueError(f"query entity {query!r} is not in the initial state")
    state = dict(initial)
    n_touch = 0
    kinds: Counter = Counter()
    for op in ops:
        if op == NO_TRANSFER:
            kinds["no_transfer"] += 1
            continue
        if m := RE_GAIN.match(op):
            who, n = m[1], int(m[2])
            if op.endswith(" point") and n != 1:
                raise ValueError(f"singular 'point' with n != 1: {op!r}")
            state[who] += n
            n_touch += who == query
            kinds["gain_one_singular" if op.endswith(" point") else "gain"] += 1
        elif m := RE_LOSE.match(op):
            who, n = m[1], int(m[2])
            state[who] -= n
            n_touch += who == query
            kinds["lose"] += 1
        elif m := RE_GIVE.match(op):
            giver, n, receiver = m[1], int(m[2]), m[3]
            if giver == receiver:
                raise ValueError(f"self transfer: {op!r}")
            state[giver] -= n
            state[receiver] += n
            n_touch += query in (giver, receiver)
            kinds["transfer"] += 1
        else:
            raise ValueError(f"unknown operation line {op!r}")
        if any(name not in initial for name in re.findall(NAME, op) if name not in ("No",)):
            raise ValueError(f"operation names an entity outside the initial state: {op!r}")
    return {"kind": kind, "answer": state[query], "initial": initial[query], "n_ops": len(ops),
            "n_touch": n_touch, "op_kinds": kinds, "query": query, "final_state": state,
            "initial_state": dict(initial), "ops": list(ops)}


def strip_period(line: str) -> str:
    if not line.endswith("."):
        raise ValueError(f"operation line has no final period: {line!r}")
    return line[:-1]


def numbered(lines: list[str], fmt: str) -> list[str]:
    """Check the '  1. ' / '  Step 1: ' numbering and return the text after it."""
    out = []
    for i, line in enumerate(lines, start=1):
        prefix = fmt.format(i)
        if not line.startswith(prefix):
            raise ValueError(f"line {i} does not start with {prefix!r}: {line!r}")
        out.append(line[len(prefix):])
    return out


def solve_main(prompt: str, head: str, q_re: str, tail: str, kind: str) -> dict:
    """bare / cot / chat layouts: head, 'Initial state:', numbered operations, question, tail."""
    blocks = prompt.split("\n\n")
    if len(blocks) != 5:
        raise ValueError(f"{kind}: expected 5 blank-line separated parts, got {len(blocks)}")
    if blocks[0] != head or blocks[4] != tail:
        raise ValueError(f"{kind}: head or tail text differs")
    first, state_line = blocks[1].split("\n")
    if first != "Initial state:":
        raise ValueError(kind)
    op_lines = blocks[2].split("\n")
    if op_lines[0] != "Operations (apply in order):":
        raise ValueError(kind)
    ops = [strip_period(t) for t in numbered(op_lines[1:], "  {}. ")]
    m = re.fullmatch(q_re, blocks[3])
    if not m:
        raise ValueError(f"{kind}: question line {blocks[3]!r}")
    return apply_points(parse_state(state_line), ops, m[1], kind)


BARE_HEAD = "You will track a sequence of point updates. You cannot refer back to the initial state after reading it once."
BARE_Q = rf"After all operations, how many points does ({NAME}) have\?"
BARE_TAIL = "Respond with ONLY the final number."
COT_TAIL = "Think step by step, then give your final answer as a single number on the last line."
CHAT_HEAD = "Track the following point updates carefully."
CHAT_Q = rf"Question: After all operations, how many points does ({NAME}) have\?"
CHAT_TAIL = "Answer with ONLY the number."


def solve_formal(prompt: str) -> dict:
    blocks = prompt.split("\n\n")
    if len(blocks) != 5 or blocks[4] != "Please provide only the numerical answer.":
        raise ValueError("formal layout")
    first, state_line = blocks[1].split("\n")
    op_lines = blocks[2].split("\n")
    if first != "Starting values:" or op_lines[0] != "Modifications to apply sequentially:":
        raise ValueError("formal labels")
    ops = [strip_period(t) for t in numbered(op_lines[1:], "  Step {}: ")]
    m = re.fullmatch(rf"Question: What is the final point total for ({NAME}) after all modifications have been applied\?", blocks[3])
    if not m:
        raise ValueError("formal question")
    return apply_points(parse_state(state_line), ops, m[1], "G-formal")


def solve_casual(prompt: str) -> dict:
    blocks = prompt.split("\n\n")
    if len(blocks) != 5 or blocks[4] != "Just tell me the number.":
        raise ValueError("casual layout")
    if not blocks[1].startswith("So we start with: "):
        raise ValueError("casual state")
    op_lines = blocks[2].split("\n")
    if op_lines[0] != "Then these things happen one after another:":
        raise ValueError("casual label")
    ops = []
    for line in op_lines[1:]:
        if not line.startswith("- ") or line.endswith("."):
            raise ValueError(f"casual op line {line!r}")
        ops.append(line[2:])
    m = re.fullmatch(rf"So after all that, how many points does ({NAME}) end up with\?", blocks[3])
    if not m:
        raise ValueError("casual question")
    return apply_points(parse_state(blocks[1][len("So we start with: "):]), ops, m[1], "G-casual")


def solve_minimal(prompt: str) -> dict:
    # Option A changes only this exact trailing instruction; the three-line
    # problem and question still pass through the original strict parser.
    prompt = prompt.removesuffix("\n\nRespond with ONLY the final number.")
    lines = prompt.split("\n")
    if len(lines) != 3 or not lines[0].startswith("Start: ") or not lines[1].startswith("Changes: "):
        raise ValueError("minimal layout")
    m = re.fullmatch(rf"({NAME})'s final points = \?", lines[2])
    if not m:
        raise ValueError("minimal question")
    return apply_points(parse_state(lines[0][7:]), lines[1][9:].split("; "), m[1], "G-minimal")


def solve_verbose(prompt: str) -> dict:
    blocks = prompt.split("\n\n")
    if len(blocks) != 5 or blocks[4] != "Important: respond with ONLY the final number, nothing else.":
        raise ValueError("verbose layout")
    first, state_line = blocks[1].split("\n")
    if first != "Here are the initial point totals for each person:":
        raise ValueError("verbose state label")
    op_lines = blocks[2].split("\n")
    ops = []
    for text in numbered(op_lines[1:], "  {}. "):
        m = re.fullmatch(r"(.+)\. \(After this step, update (.+) accordingly\.\)", text)
        if not m:
            raise ValueError(f"verbose op line {text!r}")
        op, note = m[1], m[2]
        # the note must name exactly the people whose totals the operation changes
        if g := RE_GIVE.match(op):
            want = f"{g[1]}'s and {g[3]}'s running totals"
        elif (g := RE_GAIN.match(op)) or (g := RE_LOSE.match(op)):
            want = f"{g[1]}'s running total"
        else:
            raise ValueError(f"verbose: operation without a note rule {op!r}")
        if note != want:
            raise ValueError(f"verbose note {note!r} does not match operation {op!r}")
        ops.append(op)
    m = re.fullmatch(rf"Now that you have processed all (\d+) operations, please tell me: what is ({NAME})'s final point total\?", blocks[3])
    if not m:
        raise ValueError("verbose question")
    if int(m[1]) != len(ops):
        raise ValueError(f"verbose says {m[1]} operations, prompt lists {len(ops)}")
    return apply_points(parse_state(state_line), ops, m[2], "G-verbose")


# ── block C: one step ────────────────────────────────────────────────────────
C_FORMS = [
    ("C-points", rf"({NAME}) starts with (\d+) points\.", rf"({NAME}) (gains|loses) (\d+) points\.",
     rf"How many points does ({NAME}) have now\?", "Respond with ONLY the final number.", "gains"),
    ("C-inventory", rf"({NAME}) has (\d+) items in their warehouse\.",
     rf"({NAME}) (adds|removes) (\d+) items (?:to|from) their warehouse\.",
     rf"How many items does ({NAME}) have in their warehouse now\?", "Respond with ONLY the final number.", "adds"),
    ("C-accounts", rf"({NAME})'s account balance is \$(\d+)\.",
     rf"({NAME}) (deposits|withdraws) \$(\d+) (?:into|from) their account\.",
     rf"What is ({NAME})'s account balance now\?", "Respond with ONLY the final number (no $ sign).", "deposits"),
]


def solve_c(prompt: str) -> dict:
    lines = prompt.split("\n")
    if len(lines) != 4:
        raise ValueError("C: expected 4 lines")
    for kind, init_re, op_re, q_re, hint, plus_verb in C_FORMS:
        a, b, c = re.fullmatch(init_re, lines[0]), re.fullmatch(op_re, lines[1]), re.fullmatch(q_re, lines[2])
        if a and b and c and lines[3] == hint:
            if not (a[1] == b[1] == c[1]):
                raise ValueError("C: three lines name different entities")
            if (b[2] == "adds") != (" to their" in lines[1]) and kind == "C-inventory":
                raise ValueError("C: adds/removes preposition mismatch")
            if (b[2] == "deposits") != (" into their" in lines[1]) and kind == "C-accounts":
                raise ValueError("C: deposits/withdraws preposition mismatch")
            init, delta = int(a[2]), int(b[3])
            answer = init + delta if b[2] == plus_verb else init - delta
            return {"kind": kind, "answer": answer, "initial": init, "n_ops": 1, "n_touch": int(delta != 0),
                    "query": a[1], "delta": delta, "sign": "+" if b[2] == plus_verb else "-"}
    raise ValueError(f"C: no surface form matches {lines!r}")


# ── block E: direct assignment ───────────────────────────────────────────────
E_FORMS = {
    "color": (rf"({NAME}) is (\w+)", rf"({NAME}) changes color to (\w+)\.", rf"What color is ({NAME}) now\?"),
    "location": (rf"({NAME}) is in the (\w+)", rf"({NAME}) moves to the (\w+)\.", rf"Where is ({NAME}) now\?"),
    "status": (rf"({NAME})'s status is (\w+)", rf"({NAME}) changes status to (\w+)\.", rf"What is ({NAME})'s current status\?"),
}


def solve_e(prompt: str) -> dict:
    m = re.match(r"You will track a sequence of (color|location|status) changes\. Pay careful attention to each update\.\n\n", prompt)
    if not m:
        raise ValueError("E head")
    attr = m[1]
    init_re, op_re, q_re = E_FORMS[attr]
    blocks = prompt.split("\n\n")
    if len(blocks) != 5 or blocks[4] != f"Respond with ONLY the final {attr} (one word).":
        raise ValueError("E layout")
    first, state_line = blocks[1].split("\n")
    op_lines = blocks[2].split("\n")
    if first != "Initial state:" or op_lines[0] != "Updates (apply in order):":
        raise ValueError("E labels")
    state = {}
    for part in state_line.split(", "):
        s = re.fullmatch(init_re, part)
        if not s:
            raise ValueError(f"E state entry {part!r}")
        state[s[1]] = s[2]
    if len(state) != 3 or len(set(state.values())) != 3:
        raise ValueError("E: need 3 entities with distinct initial values")
    q = re.fullmatch(q_re, blocks[3])
    if not q or q[1] not in state:
        raise ValueError("E question")
    initial = state[q[1]]
    n_touch = 0
    values_seen = set(state.values())
    for text in numbered(op_lines[1:], "  {}. "):
        o = re.fullmatch(op_re, text)
        if not o or o[1] not in state:
            raise ValueError(f"E op {text!r}")
        if state[o[1]] == o[2]:
            raise ValueError(f"E: update to the same value {text!r}")
        state[o[1]] = o[2]
        values_seen.add(o[2])
        n_touch += o[1] == q[1]
    return {"kind": f"E-{attr}", "answer": state[q[1]], "initial": initial, "n_ops": len(op_lines) - 1,
            "n_touch": n_touch, "query": q[1], "values_seen": values_seen}


# ── block F: cumulative non-arithmetic ───────────────────────────────────────

def f_parts(prompt: str, question_re: str, tail: str) -> tuple[list[str], re.Match]:
    body, question, last = prompt.split("\n\n")
    q = re.fullmatch(question_re, question)
    if not q or last != tail:
        raise ValueError("F question or tail")
    return body.split("\n"), q


def solve_f_permissions(prompt: str) -> dict:
    lines, q = f_parts(prompt, rf"What permissions does ({NAME}) currently have\? List them in alphabetical order, separated by commas\. If none, say 'no permissions'\.", "Respond with ONLY the answer.")
    s = re.fullmatch(rf"({NAME}) starts with (.+)\.", lines[0])
    if not s or s[1] != q[1]:
        raise ValueError("F permissions first line")
    start = set() if s[2] == "no permissions" else set(s[2].split(", "))
    cur, n_touch = set(start), 0
    for line in lines[1:]:
        if m := re.fullmatch(rf"({NAME}) is granted (\w+) access\.", line):
            n_touch += m[2] not in cur
            cur.add(m[2])
        elif m := re.fullmatch(rf"({NAME})'s (\w+) access is revoked\.", line):
            n_touch += m[2] in cur
            cur.discard(m[2])
        else:
            raise ValueError(f"F permissions op {line!r}")
        if m[1] != q[1]:
            raise ValueError("F permissions: other entity")
    return {"kind": "F-permissions", "answer": sorted(cur), "initial": sorted(start), "n_ops": len(lines) - 1,
            "n_touch": n_touch, "query": q[1]}


def solve_f_inventory(prompt: str) -> dict:
    lines, q = f_parts(prompt, rf"What items does ({NAME}) currently have\? List them in alphabetical order, separated by commas\. If none, say 'nothing'\.", "Respond with ONLY the answer.")
    s = re.fullmatch(rf"({NAME}) starts with: (.+)\.", lines[0])
    if not s or s[1] != q[1]:
        raise ValueError("F inventory first line")
    start = set(s[2].split(", "))
    cur, n_touch = set(start), 0
    for line in lines[1:]:
        if m := re.fullmatch(rf"({NAME}) picks up the (\w+)\.", line):
            n_touch += m[2] not in cur
            cur.add(m[2])
        elif m := re.fullmatch(rf"({NAME}) drops the (\w+)\.", line):
            n_touch += m[2] in cur
            cur.discard(m[2])
        else:
            raise ValueError(f"F inventory op {line!r}")
        if m[1] != q[1]:
            raise ValueError("F inventory: other entity")
    return {"kind": "F-inventory", "answer": sorted(cur), "initial": sorted(start), "n_ops": len(lines) - 1,
            "n_touch": n_touch, "query": q[1]}


def solve_f_schedule(prompt: str) -> dict:
    body, question, last = prompt.split("\n\n")
    q = re.fullmatch(rf"How many meetings does ({NAME}) have now\?", question)
    if not q or last != "Respond with ONLY the final number.":
        raise ValueError("F schedule question or tail")
    lines = body.split("\n")
    s = re.fullmatch(rf"({NAME}) starts the day with (\d+) meetings\.", lines[0])
    if not s or s[1] != q[1]:
        raise ValueError("F schedule first line")
    cur = start = int(s[2])
    for line in lines[1:]:
        if m := re.fullmatch(rf"A ([a-z ]+) is added to ({NAME})'s schedule\.", line):
            cur += 1
        elif m := re.fullmatch(rf"The ([a-z ]+) is cancelled from ({NAME})'s schedule\.", line):
            cur -= 1
        else:
            raise ValueError(f"F schedule op {line!r}")
        if m[2] != q[1]:
            raise ValueError("F schedule: other entity")
        if cur < 0:
            raise ValueError("F schedule: negative count")
    return {"kind": "F-schedule", "answer": cur, "initial": start, "n_ops": len(lines) - 1,
            "n_touch": len(lines) - 1, "query": q[1]}


def solve(prompt: str) -> dict:
    """Dispatch on the prompt's own text."""
    if prompt.startswith(BARE_HEAD):
        if prompt.endswith(COT_TAIL):
            return solve_main(prompt, BARE_HEAD, BARE_Q, COT_TAIL, "main-cot")
        return solve_main(prompt, BARE_HEAD, BARE_Q, BARE_TAIL, "main-bare")
    if prompt.startswith(CHAT_HEAD):
        return solve_main(prompt, CHAT_HEAD, CHAT_Q, CHAT_TAIL, "main-chat")
    if prompt.startswith("The following is a sequential state-tracking exercise."):
        return solve_formal(prompt)
    if prompt.startswith("Hey, can you help me keep track of some scores?"):
        return solve_casual(prompt)
    if prompt.startswith("Start: "):
        return solve_minimal(prompt)
    if prompt.startswith("In this task, you need to carefully track point totals"):
        return solve_verbose(prompt)
    if prompt.startswith("You will track a sequence of ") and " changes. Pay careful attention" in prompt:
        return solve_e(prompt)
    if "What permissions does " in prompt:
        return solve_f_permissions(prompt)
    if "What items does " in prompt and "currently have?" in prompt:
        return solve_f_inventory(prompt)
    if "How many meetings does " in prompt:
        return solve_f_schedule(prompt)
    if prompt.count("\n") == 3:
        return solve_c(prompt)
    raise ValueError(f"unrecognised prompt shape: {prompt[:80]!r}")


def is_degenerate(sol: dict) -> tuple[bool, bool]:
    """Q10 definition applied to the text solution: (untouched, answer == initial value)."""
    return sol["n_touch"] == 0, sol["answer"] == sol["initial"]


def load_bank(path: Path | str = DEFAULT_BANK) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
