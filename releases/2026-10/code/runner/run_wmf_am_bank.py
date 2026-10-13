#!/usr/bin/env python3
"""Run a frozen WMF-AM item bank against ONE OpenAI-compatible chat endpoint.

DRAFT (not yet in WMF-AM/code). Standard library only, so the remote GPU
server needs nothing beyond Python >= 3.10.

One invocation = one model. The same script serves both targets:

  vLLM on the A100 server
    python run_wmf_am_bank.py --bank item_bank.jsonl --out out/<label>.jsonl \
        --base-url http://127.0.0.1:8000/v1 --model <served-model-name>

  API model through OpenRouter, provider pinned
    python run_wmf_am_bank.py --bank item_bank.jsonl --out out/<label>.jsonl \
        --base-url https://openrouter.ai/api/v1 --model openai/gpt-4.1 \
        --api-key-env OPENROUTER_API_KEY --expect-provider OpenAI \
        --extra-body '{"provider":{"only":["openai"],"allow_fallbacks":false}}'

Bank format (JSONL, one item per line; schema of freeze/experiment_matrix.md):
    {"item_id": "<unique str>", "prompt": "<user turn>", "gold": <int | str | list>,
     "answer_type": "int" | "int_last" | <other>, "system": "<optional system turn>",
     "max_tokens": <optional int, overrides --max-tokens>}
Any other keys (block, k, seed, ...) are ignored here; analysis joins them by item_id.

Scoring. The raw response is the record of truth. Final scoring is done offline
by the one scoring function, for every model alike. For integer items this
runner also writes the last-integer parse of the submitted code plus two format
descriptors (`n_numbers`, `bare_format`), which is enough to apply either
scoring rule later without new requests. Other answer types are stored raw.

Output (JSONL, append-only): one record per finished request. A record with
"error": null is final and is never requested again. A record with an error
string is a MISSING trial (never scored as wrong); rerunning the same command
retries only those items and appends a new record. Nothing is ever rewritten.

Exit codes: 0 = bank complete, 1 = some items still missing, 2 = aborted.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNNER_VERSION = "wmf-am-bank-runner-draft-0.2"
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class Abort(Exception):
    """A condition that makes every further request pointless or invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Scoring: one rule for every model ────────────────────────────────────────

def clean_numbers(text: str) -> str:
    """Drop '$' and thousands separators, as wmf_am_last_op.parse_number does."""
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text.replace("$", ""))


def parse_int(text: str) -> int | None:
    """Same extraction order as wmf_am_frozen_v1.parse_answer:
    \\boxed{n}, then a labelled 'Answer: n', otherwise the last integer."""
    boxed = re.findall(r"\\boxed\{(-?\d+)\}", text)
    labelled = re.findall(r"(?:\*\*)?[Aa]nswer(?:\*\*)?[:\s]+(-?\d+)", text)
    numbers = re.findall(r"-?\d+", text)
    token = boxed[-1] if boxed else labelled[-1] if labelled else numbers[-1] if numbers else None
    return int(token) if token is not None else None


def score(response: str | None, gold: Any, answer_type: str) -> dict[str, Any]:
    """Integer items: last-integer parse plus format descriptors. Others: raw only.

    `correct` is the lenient rule of the submitted code. The stricter rule
    "exactly one integer in the response" is `n_numbers == 1 and parsed == gold`.
    """
    if answer_type not in ("int", "int_last"):
        return {"parsed": None, "correct": None, "bare_format": None, "n_numbers": None}
    text = clean_numbers(response or "")
    parsed = parse_int(text)
    return {
        "parsed": parsed,
        "correct": int(parsed == gold),
        "bare_format": bool(re.fullmatch(r"\s*-?\d+\s*\.?\s*", text)),
        "n_numbers": len(re.findall(r"-?\d+", text)),
    }


# ── Bank and resume state ────────────────────────────────────────────────────

def load_bank(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        for key in ("item_id", "prompt", "gold"):
            if key not in item:
                raise Abort(f"{path}:{line_no}: missing '{key}'")
        if not isinstance(item["item_id"], str) or not isinstance(item["prompt"], str):
            raise Abort(f"{path}:{line_no}: item_id and prompt must be strings")
        is_int = isinstance(item["gold"], int) and not isinstance(item["gold"], bool)
        item.setdefault("answer_type", "int" if is_int else "raw")
        if item["answer_type"] in ("int", "int_last") and not is_int:
            raise Abort(f"{path}:{line_no}: answer_type {item['answer_type']} needs an integer gold")
        if item["item_id"] in seen:
            raise Abort(f"{path}:{line_no}: duplicate item_id {item['item_id']!r}")
        seen.add(item["item_id"])
        items.append(item)
    if not items:
        raise Abort(f"{path}: empty bank")
    return items


def finished_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["error"] is None:
                done.add(record["item_id"])
    return done


def check_same_config(runs_path: Path, config: dict[str, Any]) -> None:
    """Resuming with different request settings would mix two conditions."""
    if not runs_path.exists():
        return
    first = json.loads(runs_path.read_text(encoding="utf-8").splitlines()[0])["config"]
    if first != config:
        changed = sorted(k for k in set(first) | set(config) if first.get(k) != config.get(k))
        raise Abort(
            f"{runs_path} was started with different settings ({', '.join(changed)}). "
            "Use a new --out path for a new configuration."
        )


# ── One request ──────────────────────────────────────────────────────────────

def build_body(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = []
    if item.get("system"):
        messages.append({"role": "system", "content": item["system"]})
    messages.append({"role": "user", "content": item["prompt"]})
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": int(item.get("max_tokens", args.max_tokens)),
    }
    if args.temperature is not None:
        body["temperature"] = args.temperature
    body.update(args.extra_body)
    return body


def post_chat(body: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if args.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_with_retry(body: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None, int]:
    """Return (payload, error, attempts). The request body never changes between attempts."""
    error = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            payload = post_chat(body, args)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code not in RETRYABLE_HTTP:
                raise Abort(f"HTTP {exc.code} from endpoint (not retryable): {detail}") from exc
            error = f"HTTP {exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            # OpenRouter can return HTTP 200 with an error object and no choices.
            if payload.get("choices"):
                return payload, None, attempt
            error = f"no choices in response: {json.dumps(payload)[:500]}"
        if attempt < args.max_attempts:
            time.sleep(args.retry_wait * 2 ** (attempt - 1))
    return None, error, args.max_attempts


def run_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    body = build_body(item, args)
    started = time.time()
    payload, error, attempts = call_with_retry(body, args)
    record: dict[str, Any] = {
        "item_id": item["item_id"],
        "label": args.label,
        "request_model": args.model,
        "prompt": item["prompt"],
        "system": item.get("system"),
        "max_tokens": body["max_tokens"],
        "gold": item["gold"],
        "answer_type": item["answer_type"],
        "ts_utc": utc_now(),
        "latency_s": round(time.time() - started, 3),
        "attempts": attempts,
        "error": error,
    }
    if payload is None:
        record.update({"raw_response": None, "parsed": None, "correct": None})
        return record

    choice = payload["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning_text = message.get("reasoning") or message.get("reasoning_content")
    usage = payload.get("usage") or {}
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    record.update({
        "raw_response": content,
        "finish_reason": choice.get("finish_reason"),
        "response_model": payload.get("model"),
        "provider": payload.get("provider"),
        "usage": usage,
        "thinking_detected": bool(reasoning_text) or reasoning_tokens > 0 or "<think>" in (content or ""),
        **score(content, item["gold"], item["answer_type"]),
    })
    return record


# ── Main loop ────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    def temperature(value: str) -> float | None:
        return None if value == "none" else float(value)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="model id sent in the request")
    parser.add_argument("--label", help="name written into every record (default: --model)")
    parser.add_argument("--api-key-env", help="name of the environment variable holding the key")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=temperature, default=0.0,
                        help="number, or 'none' to omit the field (endpoints that reject it)")
    parser.add_argument("--extra-body", type=json.loads, default={},
                        help="JSON object merged into every request body")
    parser.add_argument("--expect-provider", help="abort if the response 'provider' field differs")
    parser.add_argument("--allow-thinking", action="store_true",
                        help="do not abort on reasoning traces (never use for the main run)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, help="process at most this many pending items")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    args = parser.parse_args(argv)
    args.label = args.label or args.model
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.api_key_env and not os.environ.get(args.api_key_env):
            raise Abort(f"environment variable {args.api_key_env} is not set")
        bank = load_bank(args.bank)
        config = {
            "runner_version": RUNNER_VERSION,
            "label": args.label,
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "extra_body": args.extra_body,
            "bank": args.bank.name,
            "n_bank_items": len(bank),
        }
        runs_path = args.out.with_name(args.out.name + ".runs.jsonl")
        check_same_config(runs_path, config)
        done = finished_ids(args.out)
    except Abort as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2

    pending = [item for item in bank if item["item_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "started_utc": utc_now(), "config": config, "python": platform.python_version(),
            "concurrency": args.concurrency, "already_done": len(done), "to_run": len(pending),
        }, ensure_ascii=False) + "\n")
    print(f"{args.label}: bank={len(bank)} done={len(done)} to_run={len(pending)}")

    lock = threading.Lock()
    stop = threading.Event()
    abort_reason: list[str] = []
    counts = {"ok": 0, "missing": 0, "int_items": 0, "correct": 0, "not_bare": 0, "length": 0}

    def work(item: dict[str, Any]) -> None:
        if stop.is_set():
            return
        try:
            record = run_item(item, args)
        except Abort as exc:
            stop.set()
            abort_reason.append(str(exc))
            return
        with lock, args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["error"] is not None:
                counts["missing"] += 1
                print(f"  MISSING {record['item_id']}: {record['error'][:160]}")
                return
            counts["ok"] += 1
            counts["int_items"] += record["correct"] is not None
            counts["correct"] += record["correct"] or 0
            counts["not_bare"] += record["bare_format"] is False
            counts["length"] += record["finish_reason"] == "length"
            if counts["ok"] % 50 == 0:
                print(f"  {counts['ok']}/{len(pending)}")
            if record["thinking_detected"] and not args.allow_thinking:
                stop.set()
                abort_reason.append(f"reasoning trace in response to {record['item_id']}; thinking must be off")
            if args.expect_provider and record["provider"] != args.expect_provider:
                stop.set()
                abort_reason.append(
                    f"served by provider {record['provider']!r}, expected {args.expect_provider!r}"
                )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, pending))

    print(
        f"{args.label}: finished={counts['ok']} missing={counts['missing']} "
        f"int_items={counts['int_items']} correct_last_int={counts['correct']} not_bare_number={counts['not_bare']} "
        f"finish_reason_length={counts['length']}"
    )
    if abort_reason:
        print(f"ABORT: {abort_reason[0]}", file=sys.stderr)
        return 2
    still_missing = len(bank) - len(finished_ids(args.out))
    if still_missing:
        print(f"{still_missing} bank items have no final record yet; rerun the same command to resume.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
