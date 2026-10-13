#!/usr/bin/env python3
"""Offline API identity and response checks; never sends a network request.

Without --records, validate the saved endpoint metadata before paid inference.
With --records and --bank, also validate existing smoke/full responses.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def endpoint_checks(payload, request_model, canonical_model, provider):
    data = payload["data"]
    endpoints = [endpoint for endpoint in data["endpoints"]
                 if endpoint["provider_name"] == provider]
    return {
        f"endpoint request id == {request_model!r}": data["id"] == request_model,
        f"original provider endpoint exists ({provider})": bool(endpoints),
        f"original provider endpoints map to {canonical_model!r}":
            bool(endpoints) and all(
                endpoint["name"].partition(" | ")[2] == canonical_model
                for endpoint in endpoints),
    }


def reasoning_tokens(record):
    return (record["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") or 0


def response_checks(records, bank_ids, request_model, canonical_model, provider, mode):
    final = [record for record in records if record["error"] is None]
    failed = [record for record in records if record["error"] is not None]
    checks = {
        f"every bank item has exactly one final record ({len(bank_ids)} items)":
            sorted(record["item_id"] for record in final) == sorted(bank_ids),
        f"request_model == {request_model!r}":
            all(record["request_model"] == request_model for record in final),
        f"provider == {provider!r}":
            all(record["provider"] == provider for record in final),
        "reasoning tokens == 0 and no reasoning trace":
            all(reasoning_tokens(record) == 0 and not record["thinking_detected"]
                for record in final),
        f"response_model is the verified request id or canonical id":
            all(record["response_model"] in {request_model, canonical_model}
                for record in final),
    }
    if mode == "smoke":
        checks["error is null in every record"] = not failed
    return checks


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--request-model", required=True)
    parser.add_argument("--canonical-model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--report", type=Path, help="save the offline check result as JSON")
    args = parser.parse_args()
    if args.records is not None and (args.bank is None or args.mode is None):
        parser.error("--records requires --bank and --mode")
    if args.records is None and (args.bank is not None or args.mode is not None):
        parser.error("--bank and --mode require --records")

    checks = endpoint_checks(
        json.loads(args.endpoints.read_text(encoding="utf-8")),
        args.request_model, args.canonical_model, args.provider)
    if args.records is not None:
        records = read_jsonl(args.records)
        bank_ids = [item["item_id"] for item in read_jsonl(args.bank)]
        checks.update(response_checks(records, bank_ids, args.request_model,
                                      args.canonical_model, args.provider, args.mode))
        final = [record for record in records if record["error"] is None]
        print(f"  records={len(records)} final={len(final)} failed_attempts={len(records) - len(final)}")
        print(f"  response_model seen: {dict(Counter(record['response_model'] for record in final))}")
        print(f"  provider seen:       {dict(Counter(record['provider'] for record in final))}")
        print(f"  finish_reason seen:  {dict(Counter(record['finish_reason'] for record in final))}")
        reported_costs = [record["usage"]["cost"] for record in final
                          if record["usage"].get("cost") is not None]
        cost_text = f"{sum(reported_costs):.4f}" if reported_costs else "unknown"
        print(f"  tokens: prompt={sum(record['usage'].get('prompt_tokens') or 0 for record in final)} "
              f"completion={sum(record['usage'].get('completion_tokens') or 0 for record in final)} "
              f"reasoning={sum(reasoning_tokens(record) for record in final)} "
              f"reported_cost_usd={cost_text} cost_fields={len(reported_costs)}/{len(final)}")
        if args.mode == "smoke":
            for record in final:
                print(f"  {record['item_id']}: gold={record['gold']!r} raw_response={record['raw_response']!r}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if args.report is not None:
        args.report.write_text(json.dumps({
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "endpoints": str(args.endpoints.resolve()),
            "records": str(args.records.resolve()) if args.records else None,
            "bank": str(args.bank.resolve()) if args.bank else None,
            "request_model": args.request_model,
            "canonical_model": args.canonical_model,
            "provider": args.provider,
            "checks": checks,
            "passed": all(checks.values()),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
