#!/usr/bin/env python3
"""Score the extension bank (blocks K and L) from stored raw responses.

Rules, fixed before any model was run (2026-10-04):
  K  intermediate-state query, one integer. Same rule as block A: the response
     contains exactly one integer and it equals gold (score_bank.strict_int).
  L  multi-target readout of M values. The response contains exactly M integers
     and they equal gold in the requested order. Integers are matched by -?\\d+ on
     the raw response. The thousands-separator cleaning of the one-integer rule is
     NOT applied here, because commas separate the requested values.
Entity identifiers contain digits in these two blocks (E0..E4, U0.., V0..). They are
removed (whole tokens matching [EUV]<digits>) before integers are counted, so that a
response such as "E0 = 103" counts as one integer, as "Alice has 12 points" does in block A.
A request that ended with an error is a missing trial and never counts as wrong.

Usage:
  python score_ext.py --bank ext_bank.jsonl --out-dir DIR  OUTPUT.jsonl [OUTPUT.jsonl ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import score_bank  # noqa: E402  (same integer definition and record checks as the main evaluation)

INT = re.compile(r"-?\d+")
ENTITY = re.compile(r"\b[EUV]\d+\b")

COLUMNS = ["model", "item_id", "block", "replicate_id", "query_after", "checkpoint_slot", "position_guess_correct",
           "target_count", "condition",
           "gold", "status", "finish_reason", "truncated", "n_integers", "compliant", "correct",
           "n_targets_correct", "k_error_type", "raw_response"]


def without_entities(response: str | None) -> str:
    return ENTITY.sub(" ", response or "")


def list_integers(response: str | None) -> list[int]:
    return [int(token) for token in INT.findall(without_entities(response))]


def k_error_type(item: dict, found: list[int]) -> str:
    """For a wrong one-integer answer in block K: which reference state it equals."""
    if len(found) != 1:
        return "noncompliant"
    value = found[0]
    if value == item["gold"]:
        return "correct"
    for name, key in (("final_state", "final_value"), ("next_checkpoint", "next_checkpoint_value"),
                      ("previous_checkpoint", "previous_checkpoint_value"), ("initial_state", "initial_value")):
        if item.get(key) is not None and value == item[key]:
            return name
    return "other"


def position_guess_correct(item: dict) -> int | str:
    """Block K only. 1 if guessing the number of updates from the slot position alone, as
    ceil(slot / 3) (three updates spread over nine slots), gives the right answer; 0 otherwise.
    Defined from the bank before any model was run; it does not depend on any response."""
    if item["block"] != "K":
        return ""
    return int(-(-int(item["checkpoint_slot"]) // 3) == int(item["query_after"]))


def score_record(model: str, item: dict, record: dict | None) -> dict:
    row = {"model": model, "item_id": item["item_id"], "block": item["block"],
           "replicate_id": item["replicate_id"], "query_after": item.get("query_after", ""),
           "checkpoint_slot": item.get("checkpoint_slot", ""),
           "position_guess_correct": position_guess_correct(item),
           "target_count": item.get("target_count", ""), "condition": item.get("condition", ""),
           "gold": json.dumps(item["gold"])}
    if record is None:
        row.update(status="missing", finish_reason="", truncated="", n_integers="", compliant="", correct="",
                   n_targets_correct="", k_error_type="", raw_response="")
        return row
    response = record["raw_response"]
    row.update(status="answered", finish_reason=record.get("finish_reason") or "",
               truncated=int(record.get("finish_reason") == "length"), raw_response=response or "")
    if item["block"] == "K":
        found = score_bank.integers(without_entities(response))
        row.update(n_integers=len(found), compliant=int(len(found) == 1),
                   correct=int(score_bank.strict_int(without_entities(response), item["gold"])),
                   n_targets_correct="", k_error_type=k_error_type(item, found))
    else:
        found = list_integers(response)
        gold = item["gold"]
        compliant = len(found) == len(gold)
        row.update(n_integers=len(found), compliant=int(compliant), correct=int(found == gold),
                   n_targets_correct=sum(a == b for a, b in zip(found, gold)) if compliant else "",
                   k_error_type="")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("outputs", type=Path, nargs="+")
    args = parser.parse_args()

    bank_list = [json.loads(line) for line in args.bank.read_text(encoding="utf-8").splitlines() if line.strip()]
    bank = {item["item_id"]: item for item in bank_list}
    rows: list[dict] = []
    seen: set[str] = set()
    for path in args.outputs:
        label, final, synthetic = score_bank.load_output(path, bank)
        if synthetic:
            sys.exit(f"{path}: synthetic records are not scored here")
        if label in seen:
            sys.exit(f"two output files for model {label}")
        seen.add(label)
        rows.extend(score_record(label, item, final.get(item["item_id"])) for item in bank_list)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "scored_items.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    missing = sum(row["status"] == "missing" for row in rows)
    print(f"scored {len(rows)} rows for {len(seen)} models; missing {missing}; wrote {args.out_dir / 'scored_items.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
