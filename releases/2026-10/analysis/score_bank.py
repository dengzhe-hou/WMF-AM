#!/usr/bin/env python3
"""Score runner outputs of the frozen WMF-AM item bank. Standard library only.

    python score_bank.py --out-dir scored out/*.jsonl
    python score_bank.py --bank ../frozen/item_bank.jsonl --out-dir scored out/*.jsonl

Input: the frozen bank and one output file per model written by
frozen/bundle/run_wmf_am_bank.py (JSONL, one record per finished request).
The runner's run logs (<label>.jsonl.runs.jsonl) are skipped.

Every item gets two scores, both computed here from the stored raw response.

  correct_strict     the strict rule of each answer type.
      int        scheme one (Q9, decided): the response holds exactly one
                 integer and it equals gold. Zero or several integers = wrong.
      int_last   block I (CoT wrapper): the last integer equals gold.
      word       block E, candidate rule (Q17 open): exactly one word of the
                 item's vocabulary occurs in the response and it is gold.
      set        block F, candidate rule (Q17 open): split on commas and "and",
                 compare as sets; the empty set must be said as
                 "no permissions", "nothing" or "none".
  correct_submitted  the rule the submitted script used for that block, named by
                 the item's `legacy_scoring` field and copied below with its source.

Block G (paraphrase): option A was selected on 2026-10-03, so correct_strict
is the score used. The other columns remain raw-response diagnostics.
Future analysis must select --g-rule strict; do not run analysis in the local-run task:
  correct_first_int  the submitted rule, wmf_am_paraphrase.py L169-170 and L175-177
                     copied: None -> "", <think>...</think> removed, FIRST signed
                     integer equals gold. Same value as correct_submitted on G.
  correct_last_int   the same preprocessing, LAST signed integer equals gold.
  correct_strict     scheme one above: exactly one integer, equal to gold.
These two columns are blank outside block G.
  truncated          1 if finish_reason is "length" (the reply hit max_tokens),
                     every block; blank for a missing item.

An item with no final record (no record, or only records with an error) is
MISSING. Missing items are left out of every numerator and denominator; they
are never counted as wrong.

Output files in --out-dir:
  scored_items.csv               one row per model x bank item
  accuracy_by_block.csv          model x block
  accuracy_by_k.csv              model x K, blocks A and B
  accuracy_by_n_relevant_ops.csv model x block x n_relevant_ops
  compliance.csv                 model x integer block: bare / single integer in
                                 text / non-compliant, and finish_reason=length
  accuracy_g_by_template.csv     model x G template (and ALL): the three G rules
                                 side by side, and the number truncated
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
DEFAULT_BANK = HERE.parent / "data" / "item_bank" / "item_bank.jsonl"

INT = re.compile(r"-?\d+")
UINT = re.compile(r"\d+")
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
EMPTY_SET_PHRASES = {"no permissions", "nothing", "none"}


# ── Strict scoring rules ────────────────────────────────────────────────────

def clean_numbers(text: str) -> str:
    """Drop '$' and thousands separators. Same expression as the runner."""
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text.replace("$", ""))


def integers(response: str | None) -> list[int]:
    return [int(token) for token in INT.findall(clean_numbers(response or ""))]


def strip_markup(response: str | None) -> str:
    """Remove '*', backticks, outer whitespace and one sentence-final period."""
    text = (response or "").replace("*", "").replace("`", "").strip()
    return text[:-1].rstrip() if text.endswith(".") else text


def int_format(response: str | None) -> str:
    """bare = the integer alone; single_in_text = one integer inside other text."""
    if len(integers(response)) != 1:
        return "noncompliant"
    bare = re.fullmatch(r"-?\d+", strip_markup(clean_numbers(response or "")))
    return "bare" if bare else "single_in_text"


def strict_int(response: str | None, gold: int) -> bool:
    found = integers(response)
    return len(found) == 1 and found[0] == gold


def strict_int_last(response: str | None, gold: int) -> bool:
    found = integers(response)
    return bool(found) and found[-1] == gold


def strict_word(response: str | None, gold: str, vocab: list[str]) -> bool:
    text = strip_markup(response).lower()
    present = {word for word in vocab if re.search(rf"\b{re.escape(word)}\b", text)}
    return present == {gold}


def strict_set(response: str | None, gold: list[str]) -> bool:
    text = strip_markup(response).lower()
    parts = {part.strip().rstrip(".").strip() for part in re.split(r",|\band\b", text)} - {""}
    if not parts:                       # an empty reply is not the empty set
        return False
    if len(parts) == 1 and parts <= EMPTY_SET_PHRASES:
        parts = set()
    return parts == set(gold)


# ── Rules of the submitted scripts (copied; file and lines given) ────────────

def submitted_frozen_v1(response: str | None, gold: int) -> bool:
    """wmf_am_frozen_v1.parse_answer L115-124: boxed, labelled, else last integer."""
    if not response:
        return False
    clean = THINK.sub("", response)
    boxed = re.findall(r"\\boxed\{(-?\d+)\}", clean)
    labelled = re.findall(r"(?:\*\*)?[Aa]nswer(?:\*\*)?[:\s]+(-?\d+)", clean)
    numbers = re.findall(r"-?\d+", clean)
    token = boxed[-1] if boxed else labelled[-1] if labelled else numbers[-1] if numbers else None
    return token is not None and int(token) == gold


def submitted_last_op(response: str | None, gold: int) -> bool:
    """wmf_am_last_op.parse_number L165-197."""
    if response is None:
        return False
    resp = re.sub(r"[\$,]", "", THINK.sub("", response).strip())
    for pattern in (r"\\boxed\{(-?\d+)\}", r"(?:\*\*)?[Aa]nswer(?:\*\*)?[:\s]+(-?\d+)", r"####\s*(-?\d+)"):
        found = re.findall(pattern, resp)
        if found:
            return int(found[-1]) == gold
    numbers = re.findall(r"-?\d+", resp)      # covers the "just a number" branch too
    return bool(numbers) and int(numbers[-1]) == gold


def submitted_pick(pattern: re.Pattern, index: int) -> Callable[[str | None, int], bool]:
    """first/last (un)signed integer of wmf_am_extended_k L63-65, wmf_am_yoked_control
    L207-209, wmf_am_paraphrase L175-177, wmf_am_template_harmonization L124-128.
    Their no-number sentinels (-1, -9999) equal no gold in the bank."""
    def rule(response: str | None, gold: int) -> bool:
        numbers = pattern.findall(THINK.sub("", response or ""))
        return bool(numbers) and int(numbers[index]) == gold
    return rule


def submitted_nonarithmetic(response: str | None, gold: str) -> bool:
    """wmf_am_nonarithmetic.evaluate_response L143-161: substring, or first word."""
    if response is None:
        return False
    resp = THINK.sub("", response).strip().lower()
    if gold.lower() in resp:
        return True
    first_word = re.sub(r"[^a-z]", "", resp.split()[0]) if resp else ""
    return first_word == gold.lower()


def submitted_cumulative(response: str | None, gold_text: str, domain: str) -> bool:
    """wmf_am_cumulative_logical.score_response L165-193."""
    if response is None:
        return False
    clean = THINK.sub("", response).strip().lower()
    correct = gold_text.lower()
    if domain == "schedule":
        numbers = re.findall(r"\d+", clean)
        return bool(numbers) and (numbers[-1] == correct or numbers[0] == correct)

    def normalize_set(text: str) -> set[str]:
        parts = [part.strip().lower() for part in text.split(",")]
        return {part for part in parts if part and part != "no permissions" and part != "nothing"}

    resp_set = normalize_set(clean)
    if correct in ("no permissions", "nothing"):
        return len(resp_set) == 0 or "no " in clean or "nothing" in clean
    return resp_set == normalize_set(correct)


# ── Block G: three integer rules side by side (choice open) ─────────────────

def g_first_int(response: str | None, gold: int) -> bool:
    """wmf_am_paraphrase.py L169-170 and L175-177, unchanged."""
    if response is None:
        response = ""
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    nums = re.findall(r"-?\d+", clean)
    predicted = int(nums[0]) if nums else -9999
    return predicted == gold


def g_last_int(response: str | None, gold: int) -> bool:
    """g_first_int with nums[-1] in place of nums[0]; nothing else differs."""
    if response is None:
        response = ""
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    nums = re.findall(r"-?\d+", clean)
    predicted = int(nums[-1]) if nums else -9999
    return predicted == gold


SUBMITTED_INT_RULES: dict[str, Callable[[str | None, int], bool]] = {
    "wmf_am_frozen_v1.parse_answer": submitted_frozen_v1,
    "wmf_am_last_op.parse_number": submitted_last_op,
    "last_int": submitted_pick(INT, -1),
    "first_int": submitted_pick(INT, 0),
    "first_int_unsigned": submitted_pick(UINT, 0),
    "last_int_unsigned": submitted_pick(UINT, -1),
}


def score_item(item: dict[str, Any], response: str | None) -> dict[str, Any]:
    """Both scores for one stored response. Raises on an unknown rule name."""
    kind, gold = item["answer_type"], item["gold"]
    legacy = item["legacy_scoring"].split(" (")[0]
    out = {"n_integers": "", "format": ""}
    if kind == "int":
        out.update(rule_strict="one_integer", correct_strict=strict_int(response, gold),
                   n_integers=len(integers(response)), format=int_format(response))
    elif kind == "int_last":
        out.update(rule_strict="last_integer", correct_strict=strict_int_last(response, gold),
                   n_integers=len(integers(response)))
    elif kind == "word":
        out.update(rule_strict="one_vocab_word", correct_strict=strict_word(response, gold, item["vocab"]))
    elif kind == "set":
        out.update(rule_strict="set_equal", correct_strict=strict_set(response, gold))
    else:
        raise ValueError(f"{item['item_id']}: unknown answer_type {kind!r}")

    if legacy == "wmf_am_nonarithmetic.evaluate_response":
        submitted = submitted_nonarithmetic(response, gold)
    elif legacy == "wmf_am_cumulative_logical.score_response":
        submitted = submitted_cumulative(response, item["gold_text"], item["domain"])
    else:
        submitted = SUBMITTED_INT_RULES[legacy](response, gold)
    out.update(rule_submitted=legacy, correct_submitted=submitted)

    if item["block"] == "G":
        out.update(correct_first_int=g_first_int(response, gold), correct_last_int=g_last_int(response, gold))
    else:
        out.update(correct_first_int="", correct_last_int="")
    return out


# ── Loading ──────────────────────────────────────────────────────────────────

def load_bank(path: Path) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [item["item_id"] for item in items]
    if len(set(ids)) != len(ids):
        sys.exit(f"{path}: duplicate item_id")
    return items


def load_output(path: Path, bank: dict[str, dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]], bool]:
    """Return (label, final record per item_id, synthetic flag). Stops on any
    record that does not belong to this bank."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        sys.exit(f"{path}: no records")
    labels = {record["label"] for record in records}
    if len(labels) != 1:
        sys.exit(f"{path}: more than one label in one file: {sorted(labels)}")
    synthetic = {bool(record.get("synthetic")) for record in records}
    if len(synthetic) != 1:
        sys.exit(f"{path}: synthetic and real records in one file")
    final: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = record["item_id"]
        if item_id not in bank:
            sys.exit(f"{path}: item_id {item_id!r} is not in the bank")
        item = bank[item_id]
        for key in ("prompt", "gold", "max_tokens"):
            if record[key] != item[key]:
                sys.exit(f"{path}: {item_id}: '{key}' differs from the bank")
        if record.get("system") != item.get("system"):
            sys.exit(f"{path}: {item_id}: system turn differs from the bank")
        if record["error"] is not None:
            continue                    # a failed request is a missing trial, not an answer
        if item_id in final:
            sys.exit(f"{path}: two final records for {item_id}")
        if record["thinking_detected"]:
            sys.exit(f"{path}: {item_id}: reasoning trace in the response; thinking must be off")
        if item["answer_type"] in ("int", "int_last") and record["n_numbers"] != len(integers(record["raw_response"])):
            sys.exit(f"{path}: {item_id}: integer count stored by the runner differs from the scorer's")
        final[item_id] = record
    return labels.pop(), final, synthetic.pop()


# ── Tables ───────────────────────────────────────────────────────────────────

ITEM_COLUMNS = ["model", "item_id", "block", "answer_type", "k", "seed", "template", "domain",
                "n_relevant_ops", "gold", "status", "finish_reason", "n_integers", "format",
                "rule_strict", "correct_strict", "rule_submitted", "correct_submitted",
                "correct_first_int", "correct_last_int", "truncated", "synthetic", "raw_response"]


def blank(value: Any) -> Any:
    return "" if value is None else value


def score_model(label: str, final: dict[str, dict[str, Any]], bank: list[dict[str, Any]],
                synthetic: bool) -> list[dict[str, Any]]:
    rows = []
    for item in bank:
        row = {"model": label, "item_id": item["item_id"], "block": item["block"],
               "answer_type": item["answer_type"], "k": blank(item.get("k")), "seed": blank(item.get("seed")),
               "template": item.get("template", ""), "domain": item.get("domain", ""),
               "n_relevant_ops": blank(item.get("n_relevant_ops")),
               "gold": json.dumps(item["gold"], ensure_ascii=False), "synthetic": int(synthetic)}
        record = final.get(item["item_id"])
        if record is None:
            row.update(status="missing", finish_reason="", n_integers="", format="", rule_strict="",
                       correct_strict="", rule_submitted="", correct_submitted="",
                       correct_first_int="", correct_last_int="", truncated="", raw_response="")
        else:
            scored = score_item(item, record["raw_response"])
            for key in ("correct_strict", "correct_submitted", "correct_first_int", "correct_last_int"):
                if scored[key] != "":
                    scored[key] = int(scored[key])
            row.update(status="scored", finish_reason=blank(record.get("finish_reason")),
                       truncated=int(record.get("finish_reason") == "length"),
                       raw_response=blank(record["raw_response"]), **scored)
        rows.append(row)
    return rows


def tally(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, int]]:
    """Counts per group. Missing rows only raise n_items and n_missing."""
    groups: dict[Any, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        group = groups[key(row)]
        group["n_items"] += 1
        if row["status"] == "missing":
            group["n_missing"] += 1
            continue
        group["n_scored"] += 1
        group["n_correct_strict"] += row["correct_strict"]
        group["n_correct_submitted"] += row["correct_submitted"]
        group["n_finish_length"] += row["finish_reason"] == "length"
        if row["format"]:
            group["n_" + row["format"]] += 1
    return groups


def rate(numerator: int, denominator: int, digits: int = 6) -> str:
    return f"{numerator / denominator:.{digits}f}" if denominator else ""


def accuracy_rows(groups: dict[Any, dict[str, int]], names: list[str]) -> list[dict[str, Any]]:
    out = []
    for key in groups:
        g = groups[key]
        out.append({**dict(zip(names, key)), "n_items": g["n_items"], "n_scored": g["n_scored"],
                    "n_missing": g["n_missing"], "n_correct_strict": g["n_correct_strict"],
                    "acc_strict": rate(g["n_correct_strict"], g["n_scored"]),
                    "n_correct_submitted": g["n_correct_submitted"],
                    "acc_submitted": rate(g["n_correct_submitted"], g["n_scored"]),
                    "n_finish_length": g["n_finish_length"]})
    return out


def compliance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Integer items only (answer_type int). Scope = all, one block, or one G template."""
    int_rows = [row for row in rows if row["answer_type"] == "int"]
    scopes = [("ALL_INT", int_rows)]
    for block in sorted({row["block"] for row in int_rows}):
        scopes.append((block, [row for row in int_rows if row["block"] == block]))
    for template in sorted({row["template"] for row in int_rows if row["block"] == "G"}):
        scopes.append((f"G:{template}", [row for row in int_rows if row["block"] == "G" and row["template"] == template]))
    out = []
    for scope, scope_rows in scopes:
        for (model,), g in tally(scope_rows, lambda row: (row["model"],)).items():
            compliant = g["n_bare"] + g["n_single_in_text"]
            out.append({"model": model, "scope": scope, "n_int_items": g["n_items"], "n_scored": g["n_scored"],
                        "n_missing": g["n_missing"], "n_bare": g["n_bare"],
                        "n_single_in_text": g["n_single_in_text"], "n_noncompliant": g["n_noncompliant"],
                        "compliance_rate": rate(compliant, g["n_scored"]),
                        "bare_rate": rate(g["n_bare"], g["n_scored"]),
                        "n_finish_length": g["n_finish_length"]})
    return out


def g_template_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Block G per model and template (plus ALL): the three rules and truncation."""
    g_rows = [row for row in rows if row["block"] == "G"]
    out = []
    for model in sorted({row["model"] for row in g_rows}):
        mine = [row for row in g_rows if row["model"] == model]
        for template in ["ALL"] + sorted({row["template"] for row in mine}):
            part = [row for row in mine if template == "ALL" or row["template"] == template]
            scored = [row for row in part if row["status"] == "scored"]
            entry = {"model": model, "template": template, "n_items": len(part), "n_scored": len(scored),
                     "n_missing": len(part) - len(scored)}
            for rule in ("first_int", "last_int", "strict"):
                hits = sum(row[f"correct_{rule}"] for row in scored)
                entry[f"n_correct_{rule}"] = hits
                entry[f"acc_{rule}"] = rate(hits, len(scored))
            entry["n_truncated"] = sum(row["truncated"] for row in scored)
            entry["truncated_rate"] = rate(entry["n_truncated"], len(scored))
            out.append(entry)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs", nargs="+", type=Path, help="runner output files, one per model")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    bank = load_bank(args.bank)
    bank_by_id = {item["item_id"]: item for item in bank}
    # the runner writes <out>.runs.jsonl next to each output; "out/*.jsonl" matches it too
    outputs = [path for path in args.outputs if not path.name.endswith(".runs.jsonl")]
    loaded = [load_output(path, bank_by_id) for path in outputs]
    labels = [label for label, _, _ in loaded]
    if len(set(labels)) != len(labels):
        sys.exit(f"the same label appears in two files: {sorted(l for l in set(labels) if labels.count(l) > 1)}")
    if len({synthetic for _, _, synthetic in loaded}) != 1:
        sys.exit("synthetic and real output files in one run")

    rows: list[dict[str, Any]] = []
    for label, final, synthetic in sorted(loaded, key=lambda entry: entry[0]):
        rows.extend(score_model(label, final, bank, synthetic))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "scored_items.csv", rows, ITEM_COLUMNS)
    by_block = tally(rows, lambda row: (row["model"], row["block"]))
    write_csv(args.out_dir / "accuracy_by_block.csv", accuracy_rows(by_block, ["model", "block"]))
    by_k = tally([row for row in rows if row["block"] in ("A", "B")], lambda row: (row["model"], row["block"], row["k"]))
    write_csv(args.out_dir / "accuracy_by_k.csv", accuracy_rows(by_k, ["model", "block", "k"]))
    by_ops = tally([row for row in rows if row["n_relevant_ops"] != ""],
                   lambda row: (row["model"], row["block"], row["n_relevant_ops"]))
    ops_rows = sorted(accuracy_rows(by_ops, ["model", "block", "n_relevant_ops"]),
                      key=lambda row: (row["model"], row["block"], row["n_relevant_ops"]))
    write_csv(args.out_dir / "accuracy_by_n_relevant_ops.csv", ops_rows)
    compliance = compliance_rows(rows)
    write_csv(args.out_dir / "compliance.csv", compliance)
    write_csv(args.out_dir / "accuracy_g_by_template.csv", g_template_rows(rows))

    if loaded[0][2]:
        print("SYNTHETIC outputs: these are not model results.")
    print(f"bank items: {len(bank)}   models: {len(labels)}")
    print(f"{'model':32s} {'A strict':>9s} {'A subm.':>8s} {'compliance':>10s} {'missing':>8s} {'length':>7s}")
    overall = {row["model"]: row for row in compliance if row["scope"] == "ALL_INT"}
    for label in sorted(labels):
        a = by_block[(label, "A")]
        missing = sum(g["n_missing"] for (model, _), g in by_block.items() if model == label)
        length = sum(g["n_finish_length"] for (model, _), g in by_block.items() if model == label)
        compliant = int(overall[label]["n_bare"]) + int(overall[label]["n_single_in_text"])
        print(f"{label:32s} {rate(a['n_correct_strict'], a['n_scored'], 3):>9s} "
              f"{rate(a['n_correct_submitted'], a['n_scored'], 3):>8s} "
              f"{rate(compliant, overall[label]['n_scored'], 3):>10s} {missing:>8d} {length:>7d}")
    print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
