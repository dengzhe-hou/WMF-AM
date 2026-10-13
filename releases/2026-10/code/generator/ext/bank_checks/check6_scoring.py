#!/usr/bin/env python3
"""Check 6 / item (d): scoring.

A. What the main scorer's clean_numbers() does to a list answer, and the rule a list scorer needs.
B. analysis/score_ext.py against its own docstring, on example responses:
     entity identifiers [EUV]<digits> are removed before integers are counted;
     K: exactly one integer in the stripped text, equal to gold (score_bank.strict_int);
     L: exactly M integers in order, equal to gold, no thousands-separator cleaning.
C. The same scorer end to end (its main(), i.e. score_bank.load_output + scoring + CSV) on records
   shaped like the frozen runner's, built from the bank's gold. The records live in a temporary
   directory under verify_bank/ that is removed afterwards. They are test input, not model output.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXT = HERE.parents[3] / "analysis"
sys.path.insert(0, str(HERE.parents[3] / "analysis"))
sys.path.insert(0, str(HERE.parents[2] / "runner"))
import run_wmf_am_bank as runner  # noqa: E402
import score_bank  # noqa: E402

from promptparse import load_items  # noqa: E402

bank_list = load_items()
items = {item["item_id"]: item for item in bank_list}
failures: list[str] = []

# ── A ────────────────────────────────────────────────────────────────────────
print("A. main scorer: score_bank.clean_numbers() / integers() on list answers")
print('   clean_numbers: re.sub(r"(?<=\\d),(?=\\d{3}(?!\\d))", "", text.replace("$", ""))')
for text, gold in [("190,167", [190, 167]), ("190, 167", [190, 167]), ("190,167,135,203", [190, 167, 135, 203]),
                   ("190, 167, 135, 203", [190, 167, 135, 203]), ("[190,167]", [190, 167]), ("97,203", [97, 203]),
                   ("203,97", [203, 97]), ("190,97,167", [190, 97, 167])]:
    found = score_bank.integers(text)
    verdict = "merged: a correct answer would be scored wrong" if found != gold else "ok"
    print(f"   {text!r:22s} gold {str(gold):22s} clean_numbers -> {score_bank.clean_numbers(text)!r:20s} integers -> {str(found):22s} {verdict}")
multi = [item for item in bank_list if item["block"] == "L" and len(item["gold"]) > 1]
tight = sum(score_bank.integers(",".join(map(str, item["gold"]))) != item["gold"] for item in multi)
spaced = sum(score_bank.integers(", ".join(map(str, item["gold"]))) != item["gold"] for item in multi)
raw_tight = sum([int(t) for t in re.findall(r"-?\d+", ",".join(map(str, item["gold"])))] != item["gold"] for item in multi)
print(f"   80 L items with M>=2, gold written 'a,b' (no space): main-scorer integers() != gold on {tight}/80; written 'a, b': {spaced}/80")
print(f"   raw -?\\d+ without clean_numbers on 'a,b': {raw_tight}/80 wrong")
print("   RULE for a list scorer: no clean_numbers(); integers by -?\\d+ on the response (commas separate values);")
print("   correct = exactly M integers, equal to gold in order. L gold lies in 97..226, so no correct value has a thousands separator.")

# ── B ────────────────────────────────────────────────────────────────────────
score_ext_path = EXT / "score_ext.py"
sys.path.insert(0, str(EXT))
import score_ext  # noqa: E402

stat = score_ext_path.stat()
print(f"\nB. score_ext.py as found: mtime {datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds')}, {stat.st_size} bytes")
print(f"   ENTITY pattern {score_ext.ENTITY.pattern!r}   INT pattern {score_ext.INT.pattern!r}")


def scored(item_id: str, response: str | None) -> dict:
    return score_ext.score_record("m", items[item_id], {"raw_response": response, "finish_reason": "stop"})


cases = [
    # (item, response, expected correct, expected number of integers after entity removal)
    # L: exactly M integers in order; commas are separators; no thousands-separator cleaning
    ("L-r00-m2-tracking", "190, 167", 1, 2),
    ("L-r00-m2-tracking", "190,167", 1, 2),
    ("L-r00-m2-tracking", "190 167", 1, 2),
    ("L-r00-m2-tracking", "190;167", 1, 2),
    ("L-r00-m2-tracking", "[190, 167]", 1, 2),
    ("L-r00-m2-tracking", "190, 167.", 1, 2),
    ("L-r00-m2-tracking", "E0=190, E1=167", 1, 2),
    ("L-r00-m2-tracking", "E0=190,E1=167", 1, 2),
    ("L-r00-m2-tracking", "E0 = 190, E1 = 167", 1, 2),
    ("L-r00-m2-tracking", "E0: 190\nE1: 167", 1, 2),
    ("L-r00-m2-tracking", "- E0: 190\n- E1: 167", 1, 2),
    ("L-r00-m2-tracking", "The final values are 190 and 167.", 1, 2),
    ("L-r00-m2-tracking", "The final values for [E0, E1] are 190, 167.", 1, 2),
    ("L-r00-m2-tracking", "167, 190", 0, 2),
    ("L-r00-m2-tracking", "190", 0, 1),
    ("L-r00-m2-tracking", "190, 167, 135", 0, 3),
    ("L-r00-m2-tracking", "181, 173", 0, 2),
    ("L-r00-m2-tracking", "190, 167 (2 values)", 0, 3),
    ("L-r00-m2-tracking", "", 0, 0),
    ("L-r00-m2-tracking", None, 0, 0),
    ("L-r00-m4-tracking", "190,167,135,203", 1, 4),
    ("L-r00-m4-tracking", "190, 167, 135, 203", 1, 4),
    ("L-r00-m4-tracking", "E0 = 190, E1 = 167, E2 = 135, E3 = 203", 1, 4),
    ("L-r00-m4-tracking", "E0 = 190, E1 = 167, E2 = 135, E3 = ", 0, 3),
    ("L-r00-m4-readout", "181,173,130,215", 1, 4),
    ("L-r00-m1-readout", "181", 1, 1),
    ("L-r00-m1-readout", "E0 = 181", 1, 1),
    # K: exactly one integer after entity removal, equal to gold
    ("K-r00-q1", "104", 1, 1),
    ("K-r00-q1", "104.", 1, 1),
    ("K-r00-q1", "**104**", 1, 1),
    ("K-r00-q1", "E0 = 104", 1, 1),
    ("K-r00-q1", "E0=104", 1, 1),
    ("K-r00-q1", "E0 was 104.", 1, 1),
    ("K-r00-q2", "E0 = 103", 1, 1),
    ("K-r00-q1", "E0 was 104 immediately after slot 01.", 0, 2),      # echoes the slot number: two integers
    ("K-r00-q1", "After slot 01, E0 was 104.", 0, 2),
    ("K-r00-q1", "105 - 1 = 104", 0, 3),
    ("K-r00-q1", "103", 0, 1),
    ("K-r00-q1", "", 0, 0),
    ("K-r00-q1", None, 0, 0),
]
unexpected = 0
for item_id, response, want_correct, want_n in cases:
    row = scored(item_id, response)
    bad = (row["correct"], row["n_integers"]) != (want_correct, want_n) or row["compliant"] != int(
        want_n == (1 if item_id.startswith("K") else len(items[item_id]["gold"])))
    unexpected += bad
    print(f"   {item_id:18s} gold {str(items[item_id]['gold']):22s} {response!r:46s} integers={row['n_integers']} "
          f"compliant={row['compliant']} correct={row['correct']}{'   <-- expected correct=' + str(want_correct) + ' n=' + str(want_n) if bad else ''}")
print(f"   example responses that are not scored as the docstring describes: {unexpected}/{len(cases)}")
if unexpected:
    failures.append(f"{unexpected} example responses scored against the docstring")
labels = [("K-r00-q1", "102", "final_state"), ("K-r00-q1", "103", "next_checkpoint"), ("K-r00-q1", "105", "initial_state"),
          ("K-r00-q1", "104", "correct"), ("K-r00-q3", "103", "previous_checkpoint"), ("K-r00-q1", "99", "other"),
          ("K-r00-q1", "104 or 103", "noncompliant"), ("K-r00-q2", "102", "final_state")]
for item_id, response, want in labels:
    got = scored(item_id, response)["k_error_type"]
    print(f"   {item_id} response {response!r:14s} k_error_type={got!r}{'' if got == want else '   <-- expected ' + want}")
    if got != want:
        failures.append(f"k_error_type {item_id} {response!r}: {got} != {want}")
print("   note: for q=2 the next checkpoint value is the final value, so that wrong answer is always labelled 'final_state'.")

# ── C ────────────────────────────────────────────────────────────────────────
print("\nC. score_ext.py main() end to end on runner-shaped records built from gold (test input, removed afterwards)")


def record(item: dict, label: str, response: str) -> dict:
    return {"item_id": item["item_id"], "label": label, "request_model": label, "prompt": item["prompt"],
            "system": item.get("system"), "max_tokens": item["max_tokens"], "gold": item["gold"],
            "answer_type": item["answer_type"], "ts_utc": "test", "latency_s": 0.0, "attempts": 1, "error": None,
            "raw_response": response, "finish_reason": "stop", "response_model": label, "provider": None, "usage": {},
            "thinking_detected": False, **runner.score(response, item["gold"], item["answer_type"])}


def as_text(item: dict, style: str) -> str:
    gold = item["gold"]
    if item["block"] == "K":
        return {"requested": str(gold), "labelled": f"E0 = {gold}", "tight": f"{gold}.", "wrong": str(gold + 1)}[style]
    return {"requested": ", ".join(map(str, gold)),
            "labelled": ", ".join(f"E{i} = {v}" for i, v in enumerate(gold)),
            "tight": ",".join(map(str, gold)),
            "wrong": ", ".join(str(v + 1) for v in gold)}[style]


expected_correct = {"requested": 180, "labelled": 180, "tight": 180, "wrong": 0}
with tempfile.TemporaryDirectory(dir=HERE, prefix="tmp_e2e_") as tmp:
    tmp_path = Path(tmp)
    outputs = []
    for style in expected_correct:
        path = tmp_path / f"{style}.jsonl"
        path.write_text("".join(json.dumps(record(item, f"test-{style}", as_text(item, style))) + "\n" for item in bank_list),
                        encoding="utf-8")
        outputs.append(str(path))
    run = subprocess.run([sys.executable, "-B", str(score_ext_path), "--bank", str(HERE.parents[3] / "data" / "item_bank" / "ext_bank.jsonl"),
                          "--out-dir", str(tmp_path / "scored"), *outputs], capture_output=True, text=True)
    print("   exit code", run.returncode, "|", (run.stdout.strip() or run.stderr.strip()).replace(str(tmp_path), "<tmp>")[:200])
    if run.returncode != 0:
        failures.append("score_ext.py main() failed on runner-shaped records")
    else:
        rows = list(csv.DictReader((tmp_path / "scored" / "scored_items.csv").open(encoding="utf-8")))
        tally = Counter()
        for row in rows:
            tally[(row["model"], row["block"])] += int(row["correct"])
        for style, want in expected_correct.items():
            k, l = tally[(f"test-{style}", "K")], tally[(f"test-{style}", "L")]
            example = next(as_text(i, style) for i in bank_list if i["item_id"] == "L-r00-m4-tracking")
            print(f"   {style:10s} e.g. {example!r:42s} correct K {k}/60  L {l}/120{'' if k + l == want else '   <-- expected ' + str(want)}")
            if k + l != want:
                failures.append(f"end-to-end style {style}: {k + l} correct, expected {want}")
        if len(rows) != 4 * 180 or {row["status"] for row in rows} != {"answered"}:
            failures.append("end-to-end: scored_items.csv does not have 4 x 180 answered rows")
print(f"   temporary directory removed: {not any(HERE.glob('tmp_e2e_*'))}")

print(f"\nscoring problems: {len(failures)}")
for line in failures:
    print("  ", line)
print("CHECK 6 / (d):", "PASS" if not failures else "FAIL")
sys.exit(0 if not failures else 1)
