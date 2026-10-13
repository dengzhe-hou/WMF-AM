#!/usr/bin/env python3
"""Report API run facts using raw records and score_bank.py's scored_items.csv.

    python localrun/report_api.py --api-root results/api \
        --scored-dir results/api/scored --out-dir results/api/report

No network requests, scoring, correlations, or statistical tests are performed.
Missing full-run files mean 0 finished / 840 missing. Smoke costs include all
saved attempts and are kept separate from full-run costs. Returned usage is
not an invoice: failed calls without saved records and internal retry charges
are unknown. Concurrent request latencies are never added to estimate elapsed
time. E/F scores use candidate rules; no final rule is selected here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
LABELS = (
    "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101", "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14", "command-a-03-2025",
    "command-r7b-12-2024", "gpt-5.2-2025-12-11",
)
BLOCKS = "ABCDEFGHI"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fraction(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def usage_summary(records: list[dict]) -> dict:
    fields = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [(row.get("usage") or {}).get(key) for row in records]
        present = [value for value in values if value is not None]
        fields[key] = {
            "reported_sum": sum(present) if present else None,
            "present_records": len(present), "missing_records": len(values) - len(present),
            "coverage": ("none" if not present else
                         "complete" if len(present) == len(values) else "partial"),
        }
    return {
        "record_count": len(records),
        "records_with_usage": sum(bool(row.get("usage")) for row in records),
        "records_without_usage": sum(not row.get("usage") for row in records),
        "fields": fields,
        "reported_extra_retry_attempts": sum(max(0, row.get("attempts", 1) - 1) for row in records),
        "billing_note": "Sums usage of saved responses only; the cost of internal retries and of requests without a saved response is unknown; this is not an invoice total.",
    }


def combined_usage(summaries: list[dict]) -> dict:
    result = {key: sum(summary[key] for summary in summaries) for key in
              ("record_count", "records_with_usage", "records_without_usage", "reported_extra_retry_attempts")}
    result["fields"] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [summary["fields"][key] for summary in summaries]
        present = sum(value["present_records"] for value in values)
        result["fields"][key] = {
            "reported_sum": sum(value["reported_sum"] for value in values
                                if value["reported_sum"] is not None) if present else None,
            "present_records": present,
            "missing_records": result["record_count"] - present,
            "coverage": ("none" if not present else "complete"
                         if present == result["record_count"] else "partial"),
        }
    return result


def record_summary(path: Path, label: str, expected_ids: set[str]) -> tuple[dict, list[dict]]:
    records = read_jsonl(path)
    final = {}
    for row in records:
        if row["label"] != label or row["item_id"] not in expected_ids:
            raise ValueError(f"{path}: unexpected label or item_id")
        if row["error"] is None:
            if row["item_id"] in final:
                raise ValueError(f"{path}: duplicate final record for {row['item_id']}")
            final[row["item_id"]] = row
    successful = list(final.values())
    starts = [utc(row["started_utc"]) for row in read_jsonl(Path(str(path) + ".runs.jsonl"))]
    ends = [utc(row["ts_utc"]) for row in records if row.get("ts_utc")]
    observed = None
    if starts and ends:
        observed = {
            "first_runner_start_utc": min(starts).isoformat(),
            "last_record_utc": max(ends).isoformat(),
            "seconds": (max(ends) - min(starts)).total_seconds(),
            "meaning": "Observed interval from the earliest runner start to the last response record; it includes gaps before resumption and is not a confirmed job end time.",
        }
    checks = read_json(Path(str(path) + ".checks.json"))
    return {
        "path": str(path.resolve()), "finished": len(final),
        "missing": len(expected_ids) - len(final), "n_expected": len(expected_ids),
        "failed_records": sum(row["error"] is not None for row in records),
        "provider_counts": dict(Counter(row.get("provider") for row in successful)),
        "response_model_counts": dict(Counter(row.get("response_model") for row in successful)),
        "request_model_counts": dict(Counter(row.get("request_model") for row in successful)),
        "finish_reason_counts": dict(Counter(row.get("finish_reason") for row in successful)),
        "truncated": sum(row.get("finish_reason") == "length" for row in successful),
        "usage": usage_summary(records), "observed_elapsed": observed,
        "checks": checks,
        "status": ("complete_checks_passed" if len(final) == len(expected_ids) and checks and checks["passed"]
                   else "records_complete_checks_pending_or_failed" if len(final) == len(expected_ids)
                   else "incomplete" if starts else "not_started"),
        "timing": read_jsonl(Path(str(path) + ".timing.jsonl")),
        "final_item_ids": sorted(final),
    }, records


def smoke_paths(api_root: Path, label: str) -> list[Path]:
    # A non-retryable HTTP error can leave only .runs.jsonl, without a response file.
    paths = set()
    for path in (api_root / "smoke").glob(f"{label}.*.jsonl"):
        if path.name.endswith(".runs.jsonl"):
            paths.add(Path(str(path).removesuffix(".runs.jsonl")))
        elif not path.name.endswith(".timing.jsonl"):
            paths.add(path)
    return sorted(paths)


def summarize(label: str, api_root: Path, scored: list[dict], bank: list[dict], smoke_ids: set[str]) -> dict:
    bank_ids = {row["item_id"] for row in bank}
    full, _ = record_summary(api_root / f"{label}.jsonl", label, bank_ids)
    all_smoke_records, attempts = [], []
    for path in smoke_paths(api_root, label):
        attempt, records = record_summary(path, label, smoke_ids)
        attempts.append(attempt)
        all_smoke_records.extend(records)
    latest = attempts[-1] if attempts else None
    smoke = {
        "kind": "5-item API path check; not the local S1–S6, and it does not require the 5 answers to be correct",
        "latest": latest, "attempts": attempts,
        "all_attempts_usage": usage_summary(all_smoke_records),
    }
    notes = []
    done = [row for row in scored if row["status"] == "scored"]
    current_scores = bool(scored) and (
        len(scored) == len(bank) and {row["item_id"] for row in scored} == bank_ids
        and {row["item_id"] for row in done} == set(full["final_item_ids"])
    )
    if scored and not current_scores:
        notes.append("The scored CSV does not match the current full bank/response records, so scores are not reported; rerun score_bank.py.")
    elif not scored and full["finished"]:
        notes.append("Full-run responses exist but there is no score_bank.py output yet.")
    block_counts = Counter(row["block"] for row in bank)
    blocks = {}
    for block in BLOCKS:
        group = [row for row in done if row["block"] == block] if current_scores else []
        blocks[block] = {
            "expected": block_counts[block],
            "correct": fraction(sum(int(row["correct_strict"]) for row in group), len(group))
                       if current_scores else None,
            "rule_status": "candidate rule, final choice pending" if block in "EF" else "current rule",
        }
    numeric = [row for row in done if row["answer_type"] == "int"]
    compliance = (fraction(sum(int(row["n_integers"]) == 1 for row in numeric), len(numeric))
                  if current_scores else None)
    return {
        "label": label, "full": full, "smoke": smoke,
        "blocks": blocks, "compliance_ALL_INT": compliance,
        "scoring_csv_matches_raw": current_scores,
        "endpoint_provenance": [str(path.resolve()) for path in
                                sorted((api_root / "provenance").glob(f"{label}.endpoints.*.json"))],
        "notes": notes,
    }


def pretty_fraction(value: dict | None) -> str:
    if value is None:
        return "not scored"
    if not value["denominator"]:
        return "0/0 (no finished item)"
    return f"{value['numerator']}/{value['denominator']} ({value['rate']:.1%})"


def smoke_status(smoke: dict) -> str:
    latest = smoke["latest"]
    if latest is None:
        return "not started"
    checks = latest["checks"]
    if checks is not None:
        return "passed" if checks.get("passed") is True else "failed"
    return f"{latest['finished']}/5 returned, no pass verdict saved"


def format_usage(usage: dict) -> tuple[str, str, str]:
    fields = usage["fields"]
    tokens = "; ".join(
        f"{key}={fields[key]['reported_sum'] if fields[key]['reported_sum'] is not None else 'unknown'}"
        f" [{fields[key]['present_records']}/{usage['record_count']}]"
        for key in ("prompt_tokens", "completion_tokens")
    )
    cost = fields["cost"]
    amount = f"${cost['reported_sum']:.6f}" if cost["reported_sum"] is not None else "unknown"
    coverage = f"{cost['present_records']}/{usage['record_count']} ({cost['coverage']})"
    return tokens, amount, coverage


def markdown(report: dict) -> str:
    lines = ["# Descriptive API run report", "", f"Generated: {report['generated_utc']}.", "",
             "Scores are read from the output of `analysis/score_bank.py` only; no correlation, significance test, or manuscript conclusion is made. "
             "Missing items are not counted as wrong and do not enter the accuracy denominator. E/F use candidate rules whose final choice is pending.", "",
             "| Model | Full run finished / missing | Latest smoke test | Exactly one integer on ordinary numeric items | Truncated | Observed interval (s) | Runner sessions (s) |",
             "| --- | ---: | --- | --- | ---: | ---: | --- |"]
    for model in report["models"]:
        full = model["full"]
        elapsed = full["observed_elapsed"]
        sessions = ", ".join(f"{row['elapsed_seconds']:.3f} (exit {row['runner_exit_code']})"
                             for row in full["timing"]) or "unknown"
        lines.append(f"| {model['label']} | {full['finished']} / {full['missing']} | "
                     f"{smoke_status(model['smoke'])} | {pretty_fraction(model['compliance_ALL_INT'])} | "
                     f"{full['truncated']} | {elapsed['seconds'] if elapsed else 'unknown'} | {sessions} |")
    lines.extend(["", "Ordinary numeric items are those with `answer_type=int`, with text around the integer allowed; the last-integer rule of I is not included. "
                  "The observed interval runs from the earliest runner start to the last response record; it may include gaps before resumption and is not a final job duration. "
                  "Runner sessions come from actual start/end timing and exclude the public endpoint GET and offline checks. "
                  "Latencies of concurrent requests are not added. The smoke test is a 5-item API path check; it does not require the 5 answers to be correct and is not the local S1–S6.", "",
                  "| Model | " + " | ".join(f"{block} {'(candidate)' if block in 'EF' else ''}" for block in BLOCKS) + " |",
                  "| --- | " + " | ".join("---:" for _ in BLOCKS) + " |"])
    for model in report["models"]:
        lines.append("| " + model["label"] + " | " + " | ".join(
            pretty_fraction(model["blocks"][block]["correct"]) for block in BLOCKS) + " |")
    lines.extend(["", "| Model | Scope | Returned token totals [field coverage/records] | Cost reported in returned usage | cost field coverage |",
                  "| --- | --- | --- | ---: | --- |"])
    for model in report["models"]:
        for scope, usage in (("full", model["full"]["usage"]),
                             ("smoke all attempts", model["smoke"]["all_attempts_usage"])):
            tokens, amount, coverage = format_usage(usage)
            lines.append(f"| {model['label']} | {scope} | {tokens} | {amount} | {coverage} |")
    for scope, usage in report["usage_totals"].items():
        tokens, amount, coverage = format_usage(usage)
        lines.append(f"| All models | {scope} | {tokens} | {amount} | {coverage} |")
    lines.extend(["", "Cost sums `usage.cost` of saved responses only. A missing value is not treated as 0; partial is the amount over the records that returned one, "
                  "and none means unknown. The cost of requests without a saved response, such as internal retries and authentication failures, is unknown, so these totals are not an invoice total.", ""])
    for model in report["models"]:
        full = model["full"]
        lines.append(f"- {model['label']}: full-run status={full['status']}; provider={json.dumps(full['provider_counts'], ensure_ascii=False)}; "
                     f"response_model={json.dumps(full['response_model_counts'], ensure_ascii=False)}.")
        lines.extend(f"  {note}" for note in model["notes"])
    lines.extend(["", "Raw paths, every smoke attempt, timing/check records, and endpoint provenance are in `API_RUN_REPORT.json` in the same directory.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-root", type=Path, default=ROOT / "results" / "api")
    parser.add_argument("--scored-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    scored_dir = args.scored_dir or args.api_root / "scored"
    out_dir = args.out_dir or args.api_root / "report"
    by_model = defaultdict(list)
    csv_path = scored_dir / "scored_items.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                by_model[row["model"]].append(row)
    bank = read_jsonl(ROOT / "frozen" / "item_bank.jsonl")
    smoke_ids = {row["item_id"] for row in read_jsonl(ROOT / "frozen" / "bundle" / "smoke_bank.jsonl")}
    report = {
        "generated_utc": datetime.now(timezone.utc).astimezone(ZoneInfo("UTC")).isoformat(),
        "scored_csv": str(csv_path.resolve()),
        "models": [summarize(label, args.api_root, by_model[label], bank, smoke_ids) for label in LABELS],
    }
    report["usage_totals"] = {
        "full": combined_usage([model["full"]["usage"] for model in report["models"]]),
        "smoke all attempts": combined_usage([model["smoke"]["all_attempts_usage"] for model in report["models"]]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "API_RUN_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "API_RUN_REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(f"Written {out_dir / 'API_RUN_REPORT.md'} and API_RUN_REPORT.json")


if __name__ == "__main__":
    main()
