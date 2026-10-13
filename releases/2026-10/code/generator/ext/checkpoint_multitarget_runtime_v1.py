#!/usr/bin/env python3
"""Shared fail-closed runtime for checkpoint and multi-target experiments.

This module contains no task generator.  A versioned experiment runner supplies
its frozen item bank, strict parser, scorer, and one or more schema preflights.
Each scored item is reserved and atomically checkpointed before its sole
provider request.  A provider failure remains missing and is never silently
retried or converted to an incorrect response.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bank_fingerprint(bank: Iterable[dict[str, Any]]) -> str:
    records = [
        {
            "item_id": item["item_id"],
            "prompt_sha256": item["prompt_sha256"],
            "correct_answer": item["correct_answer"],
            "answer_schema": item["answer_schema"],
        }
        for item in bank
    ]
    records.sort(key=lambda row: row["item_id"])
    return canonical_sha256(records)


def verify_pinned_sources(manifest: dict[str, Any]) -> None:
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("manifest must pin source_sha256")
    for relative, expected in sources.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid or placeholder source hash for {relative}")
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT.resolve())
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen source drift for {relative}")


def _ollama_manifest_path(model_id: str) -> Path:
    name, separator, tag = model_id.partition(":")
    if (
        not separator
        or not name
        or not tag
        or "/" in name
        or "/" in tag
        or ".." in model_id
    ):
        raise ValueError(f"unsafe Ollama model ID: {model_id}")
    return (
        Path.home()
        / ".ollama"
        / "models"
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / name
        / tag
    )


def register_and_verify_model(model: str, specification: dict[str, Any]) -> None:
    if model != specification["model"] or not model.startswith("ollama:"):
        raise ValueError("model is outside the frozen Ollama panel")
    model_id = model.removeprefix("ollama:")
    if model_id != specification["model_id"]:
        raise ValueError("model ID/prefix mismatch")
    path = _ollama_manifest_path(model_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing Ollama manifest: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != specification["ollama_manifest_sha256"]:
        raise ValueError(f"Ollama manifest hash drift for {model}")
    if len(payload) != int(specification["ollama_manifest_bytes"]):
        raise ValueError(f"Ollama manifest byte-count drift for {model}")


def validate_loopback_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only a plain loopback Ollama base URL is permitted")
    return base_url.rstrip("/")


def _http_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any] | None,
    timeout_s: int,
) -> dict[str, Any]:
    data = (
        None
        if payload is None
        else json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )
    request = urllib.request.Request(
        f"{base_url}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama returned a non-object payload")
    return parsed


def ollama_version(base_url: str, timeout_s: int) -> dict[str, Any]:
    return _http_json(base_url, "/api/version", None, timeout_s)


def loaded_model_row(
    base_url: str, model_id: str, timeout_s: int
) -> dict[str, Any] | None:
    rows = _http_json(base_url, "/api/ps", None, timeout_s).get("models", [])
    if not isinstance(rows, list):
        raise RuntimeError("Ollama /api/ps models field is not a list")
    return next(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and (row.get("name") == model_id or row.get("model") == model_id)
        ),
        None,
    )


def native_chat_request(
    model: str,
    prompt: str,
    runtime: dict[str, Any],
    answer_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.removeprefix("ollama:"),
        "messages": [{"role": "user", "content": prompt}],
        "format": answer_schema,
        "stream": False,
        "think": runtime["think"],
        "keep_alive": runtime["keep_alive"],
        "options": {
            "temperature": runtime["temperature"],
            "seed": runtime["seed"],
            "num_predict": runtime["num_predict"],
            "num_ctx": runtime["num_ctx"],
        },
    }


def run_native(
    base_url: str, request: dict[str, Any], runtime: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    response = _http_json(
        base_url,
        "/api/chat",
        request,
        int(runtime["request_timeout_seconds"]),
    )
    content = (response.get("message") or {}).get("content")
    if content is not None and not isinstance(content, str):
        raise RuntimeError("Ollama response content is not text")
    return content, response


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def schema_gpu_preflight(
    base_url: str,
    model_spec: dict[str, Any],
    runtime: dict[str, Any],
    answer_schema: dict[str, Any],
    prompt: str,
    expected: Any,
    parser: Callable[[str | None, dict[str, Any] | None], Any],
    schema_key: str,
    require_gpu: bool,
    reason: str,
) -> dict[str, Any]:
    model = model_spec["model"]
    started = utc_now()
    request = native_chat_request(model, prompt, runtime, answer_schema)
    raw: str | None = None
    response: dict[str, Any] | None = None
    predicted: Any = None
    loaded: dict[str, Any] | None = None
    runner_error: str | None = None
    try:
        raw, response = run_native(base_url, request, runtime)
        predicted = parser(raw, None)
        loaded = loaded_model_row(
            base_url,
            model.removeprefix("ollama:"),
            int(runtime["request_timeout_seconds"]),
        )
    except Exception as exc:
        runner_error = f"{type(exc).__name__}: {exc}"
    size_vram = int((loaded or {}).get("size_vram", 0))
    return {
        "model": model,
        "model_id": model.removeprefix("ollama:"),
        "schema_key": schema_key,
        "reason": reason,
        "started_utc": started,
        "finished_utc": utc_now(),
        "provider_request": request,
        "provider_response": response,
        "raw_response": raw,
        "predicted": predicted,
        "expected": expected,
        "schema_compatible": predicted == expected,
        "api_ps_model": loaded,
        "size_vram": size_vram,
        "gpu_required": require_gpu,
        "gpu_verified": size_vram > 0,
        "ollama_manifest_sha256": model_spec["ollama_manifest_sha256"],
        "ollama_manifest_bytes": model_spec["ollama_manifest_bytes"],
        "request_status": (
            "observed" if response is not None else "missing_provider_error"
        ),
        "runner_error": runner_error,
    }


def _new_payload(
    experiment_id: str,
    runner_version: str,
    manifest: dict[str, Any],
    bank_hash: str,
    manifest_file_hash: str,
    model_spec: dict[str, Any],
    base_url: str,
    allow_cpu: bool,
    version: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "runner_version": runner_version,
        "manifest_version": manifest["manifest_version"],
        "manifest_sha256": manifest_file_hash,
        "bank_sha256": bank_hash,
        "model": model_spec["model"],
        "selected_model": model_spec["model"],
        "model_id": model_spec["model_id"],
        "model_family": model_spec["family"],
        "ollama_base_url": base_url,
        "ollama_version": version,
        "allow_cpu": allow_cpu,
        "failure_policy": manifest["failure_policy"],
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "completed_utc": None,
        "shard_kind": "scored",
        "shard_status": "initialized",
        "preflight_gate_status": "pending",
        "model_preflights": [],
        "trials": [],
    }


def _load_or_initialize(
    output: Path,
    experiment_id: str,
    runner_version: str,
    manifest: dict[str, Any],
    bank_hash: str,
    manifest_file_hash: str,
    model_spec: dict[str, Any],
    base_url: str,
    allow_cpu: bool,
    version: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if resume:
        if not output.is_file():
            raise FileNotFoundError(f"--resume output does not exist: {output}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        frozen = {
            "experiment_id": experiment_id,
            "runner_version": runner_version,
            "manifest_sha256": manifest_file_hash,
            "bank_sha256": bank_hash,
            "model": model_spec["model"],
            "ollama_base_url": base_url,
            "allow_cpu": allow_cpu,
        }
        for key, expected in frozen.items():
            if payload.get(key) != expected:
                raise ValueError(f"cannot resume: {key} differs from frozen run")
        return payload
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    return _new_payload(
        experiment_id,
        runner_version,
        manifest,
        bank_hash,
        manifest_file_hash,
        model_spec,
        base_url,
        allow_cpu,
        version,
    )


def run_shard(
    *,
    experiment_id: str,
    runner_version: str,
    model_spec: dict[str, Any],
    bank: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_file_hash: str,
    output: Path,
    resume: bool,
    base_url: str,
    allow_cpu: bool,
    parser: Callable[[str | None, dict[str, Any] | None], Any],
    scorer: Callable[[Any, dict[str, Any]], dict[str, Any]],
    preflight_specs: list[dict[str, Any]],
) -> None:
    runtime = manifest["inference"]["runtime"]
    version = ollama_version(base_url, int(runtime["request_timeout_seconds"]))
    bank_hash = bank_fingerprint(bank)
    payload = _load_or_initialize(
        output,
        experiment_id,
        runner_version,
        manifest,
        bank_hash,
        manifest_file_hash,
        model_spec,
        base_url,
        allow_cpu,
        version,
        resume,
    )
    atomic_write_json(output, payload)

    model = model_spec["model"]
    timeout = int(runtime["request_timeout_seconds"])
    loaded = loaded_model_row(base_url, model_spec["model_id"], timeout)
    valid_keys = {
        row.get("schema_key")
        for row in payload["model_preflights"]
        if row.get("schema_compatible")
        and (allow_cpu or row.get("gpu_verified"))
    }
    expected_keys = {specification["schema_key"] for specification in preflight_specs}
    need_gate = (
        valid_keys != expected_keys
        or loaded is None
        or (not allow_cpu and int((loaded or {}).get("size_vram", 0)) <= 0)
    )
    if need_gate:
        reason = "initial_load" if not valid_keys else "resume_reload"
        new_rows = [
            schema_gpu_preflight(
                base_url=base_url,
                model_spec=model_spec,
                runtime=runtime,
                answer_schema=specification["answer_schema"],
                prompt=specification["prompt"],
                expected=specification["expected"],
                parser=parser,
                schema_key=specification["schema_key"],
                require_gpu=not allow_cpu,
                reason=reason,
            )
            for specification in preflight_specs
        ]
        payload["model_preflights"].extend(new_rows)
        passed = all(
            row["schema_compatible"] and (allow_cpu or row["gpu_verified"])
            for row in new_rows
        )
        payload["preflight_gate_status"] = "passed" if passed else "failed"
        payload["shard_status"] = "running" if passed else "failed_preflight"
        payload["updated_utc"] = utc_now()
        atomic_write_json(output, payload)
        if not passed:
            raise RuntimeError(
                f"structured-output/GPU preflight failed for {model}: "
                f"{[(row['schema_key'], row['raw_response'], row['size_vram']) for row in new_rows]}"
            )
    else:
        payload["preflight_gate_status"] = "passed"
        payload["shard_status"] = "running"
        payload["updated_utc"] = utc_now()
        atomic_write_json(output, payload)

    existing = {trial["item_id"] for trial in payload["trials"]}
    for index, item in enumerate(bank, start=1):
        if item["item_id"] in existing:
            continue
        request = native_chat_request(
            model, item["prompt"], runtime, item["answer_schema"]
        )
        trial = dict(item)
        trial.update(
            {
                "request_status": "reserved",
                "request_started_utc": utc_now(),
                "request_finished_utc": None,
                "raw_response": None,
                "provider_request": request,
                "provider_response": None,
                "predicted": None,
                "accurate": None,
                "scorer_status": None,
                "runner_error": None,
                "native_done_reason": None,
                "native_eval_count": None,
                "native_prompt_eval_count": None,
                "native_eval_duration": None,
                "native_load_duration": None,
            }
        )
        payload["trials"].append(trial)
        payload["updated_utc"] = utc_now()
        atomic_write_json(output, payload)
        try:
            raw, response = run_native(base_url, request, runtime)
            predicted = parser(raw, item)
            score_fields = scorer(predicted, item)
            trial.update(
                {
                    "request_status": "observed",
                    "raw_response": raw,
                    "provider_response": response,
                    "predicted": predicted,
                    "scorer_status": (
                        "strict_json_valid"
                        if predicted is not None
                        else "strict_json_invalid"
                    ),
                    "native_done_reason": response.get("done_reason"),
                    "native_eval_count": response.get("eval_count"),
                    "native_prompt_eval_count": response.get("prompt_eval_count"),
                    "native_eval_duration": response.get("eval_duration"),
                    "native_load_duration": response.get("load_duration"),
                    **score_fields,
                }
            )
        except Exception as exc:
            trial.update(
                {
                    "request_status": "missing_provider_error",
                    "runner_error": f"{type(exc).__name__}: {exc}",
                }
            )
        trial["request_finished_utc"] = utc_now()
        payload["updated_utc"] = utc_now()
        atomic_write_json(output, payload)
        print(
            f"[{model} {index:>3}/{len(bank)}] "
            f"accurate={trial.get('accurate')}"
        )

    payload["completed_utc"] = utc_now()
    payload["shard_status"] = "complete"
    payload["updated_utc"] = utc_now()
    atomic_write_json(output, payload)
