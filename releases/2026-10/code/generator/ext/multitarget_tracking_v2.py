#!/usr/bin/env python3
"""Frozen calibrated multi-target tracking experiment.

For M in {1, 2, 4}, each queried state receives exactly two non-cancelling
updates and every condition includes exactly four active distractors.  The
displayed operation count is therefore 2*M+4, namely 6, 8, or 12.  A matched
readout control uses the same initial state, operations, output length, slot
order, signs, and magnitudes but redirects target updates to shadow entities.

This design keeps distractor count and per-target update load fixed as M grows.
It replaces neither the submitted experiment nor the frozen v1
extension.  The default invocation only validates the frozen bank.  ``--execute``
performs strict structured-output and GPU preflights before scored requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import checkpoint_multitarget_runtime_v1 as runtime  # noqa: E402
import multitarget_tracking_v1 as v1  # noqa: E402


RUNNER_VERSION = "multitarget-tracking-v2.0"
EXPERIMENT_ID = "wmf-am-multitarget-tracking-v2"
MANIFEST_PATH = (
    ROOT / "multitarget_tracking_manifest_v2.json"
)
REPLICATES = tuple(range(20))
TARGET_LEVELS = (1, 2, 4)
CONDITIONS = ("parallel_tracking", "readout_control")
UPDATES_PER_TARGET = 2
ACTIVE_DISTRACTORS = 4


answer_schema = v1.answer_schema
parse_json_answers = v1.parse_json_answers
score_answers = v1.score_answers


def display_slots(target_count: int) -> int:
    if target_count not in TARGET_LEVELS:
        raise ValueError("target_count must be 1, 2, or 4")
    return UPDATES_PER_TARGET * target_count + ACTIVE_DISTRACTORS


def _materialize_operations(
    trajectory: dict[str, Any], target_count: int, condition: str
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for template in trajectory["slot_templates"]:
        if (
            template["role"] == "lane_update"
            and int(template["lane"]) >= target_count
        ):
            continue
        operation = dict(template)
        if operation["role"] == "lane_update":
            lane = int(operation["lane"])
            if condition == "parallel_tracking":
                operation["entity"] = trajectory["query_entities"][lane]
                operation["query_relevant"] = 1
            else:
                operation["entity"] = trajectory["shadow_entities"][lane]
                operation["query_relevant"] = 0
        else:
            operation["query_relevant"] = 0
        operation["active"] = 1
        operations.append(operation)
    return operations


def build_problem(
    replicate_id: int, target_count: int, condition: str
) -> dict[str, Any]:
    if replicate_id not in REPLICATES:
        raise ValueError("replicate_id must be in 0..19")
    if target_count not in TARGET_LEVELS:
        raise ValueError("target_count outside frozen grid")
    if condition not in CONDITIONS:
        raise ValueError("condition outside frozen grid")

    trajectory = v1._make_trajectory(replicate_id)
    operations = _materialize_operations(trajectory, target_count, condition)
    query_entities = trajectory["query_entities"][:target_count]
    state = dict(trajectory["initial_state"])
    for operation in operations:
        v1._apply(state, operation)
    correct_answer = [state[entity] for entity in query_entities]
    rendered = [
        v1._render(operation, slot)
        for slot, operation in enumerate(operations, start=1)
    ]
    initial = ", ".join(
        f"{entity}={trajectory['initial_state'][entity]}"
        for entity in trajectory["entities"]
    )
    ordered_targets = ", ".join(query_entities)
    slot_count = display_slots(target_count)
    prompt = (
        "Track integer point states through the displayed sequence. Apply all "
        f"{slot_count} operations exactly once and in order.\n\n"
        f"Initial state: {initial}\n\n"
        "Operation slots:\n"
        + "\n".join(rendered)
        + "\n\n"
        f"Return the final values for [{ordered_targets}] in exactly this "
        "order as the JSON answers array. Return only the required JSON object."
    )
    item = {
        "item_id": (
            f"multitarget_v2_r{replicate_id:02d}_m{target_count}_{condition}"
        ),
        "task_type": "numeric_multitarget_calibrated",
        "replicate_id": replicate_id,
        "replicate_seed": trajectory["replicate_seed"],
        "target_count": target_count,
        "condition": condition,
        "query_entities": query_entities,
        "potential_query_entities": trajectory["query_entities"],
        "shadow_entities": trajectory["shadow_entities"],
        "noise_entities": trajectory["noise_entities"],
        "initial_state": trajectory["initial_state"],
        "operations": operations,
        "display_slots": slot_count,
        "updates_per_target": UPDATES_PER_TARGET,
        "active_distractors": ACTIVE_DISTRACTORS,
        "correct_answer": correct_answer,
        "answer_schema": answer_schema(target_count),
        "rendered_slots": rendered,
        "prompt": prompt,
        "prompt_char_count": len(prompt),
        "prompt_sha256": runtime.prompt_sha256(prompt),
        "trajectory_sha256": runtime.canonical_sha256(
            {
                "initial_state": trajectory["initial_state"],
                "slot_templates": trajectory["slot_templates"],
            }
        ),
    }
    validate_problem(item)
    return item


def validate_problem(item: dict[str, Any]) -> None:
    if item.get("task_type") != "numeric_multitarget_calibrated":
        raise ValueError("wrong calibrated multi-target task label")
    m = int(item["target_count"])
    condition = item["condition"]
    if m not in TARGET_LEVELS or condition not in CONDITIONS:
        raise ValueError("calibrated multi-target cell outside frozen grid")
    if len(item["query_entities"]) != m:
        raise ValueError("query output length does not equal M")
    if item["answer_schema"] != answer_schema(m):
        raise ValueError("exact-M answer schema drift")

    operations = item["operations"]
    expected_slots = display_slots(m)
    if len(operations) != expected_slots or item["display_slots"] != expected_slots:
        raise ValueError("calibrated operation count drift")
    if any(operation.get("active") != 1 for operation in operations):
        raise ValueError("all calibrated operations must be active")

    lane_rows = [
        operation for operation in operations if operation["role"] == "lane_update"
    ]
    distractors = [
        operation for operation in operations if operation["role"] == "distractor"
    ]
    if len(lane_rows) != m * UPDATES_PER_TARGET:
        raise ValueError("per-target update count drift")
    if len(distractors) != ACTIVE_DISTRACTORS:
        raise ValueError("distractor count must remain four at every M")

    for lane in range(m):
        rows = [row for row in lane_rows if int(row["lane"]) == lane]
        if [int(row["step"]) for row in rows] != [1, 2]:
            raise ValueError("target update order drift")
        expected_entity = (
            f"E{lane}" if condition == "parallel_tracking" else f"U{lane}"
        )
        if {row["entity"] for row in rows} != {expected_entity}:
            raise ValueError("target update routing drift")
        expected_relevance = int(condition == "parallel_tracking")
        if {row["query_relevant"] for row in rows} != {expected_relevance}:
            raise ValueError("query relevance drift")
        total = 0
        prefixes: list[int] = []
        for row in rows:
            total += (1 if row["kind"] == "gain" else -1) * int(row["amount"])
            prefixes.append(total)
        if any(prefix == 0 for prefix in prefixes):
            raise ValueError("target lane admits an initial-value shortcut")
    if {int(row["lane"]) for row in lane_rows} != set(range(m)):
        raise ValueError("unexpected lane appears in calibrated item")

    query_set = set(item["query_entities"])
    if any(operation["entity"] in query_set for operation in distractors):
        raise ValueError("distractor touches a queried entity")
    state = dict(item["initial_state"])
    for operation in operations:
        v1._apply(state, operation)
    expected = [state[entity] for entity in item["query_entities"]]
    if expected != item["correct_answer"]:
        raise ValueError("stored answer is not executable")
    initial_outputs = [
        item["initial_state"][entity] for entity in item["query_entities"]
    ]
    if condition == "parallel_tracking":
        if any(
            observed == initial
            for observed, initial in zip(expected, initial_outputs)
        ):
            raise ValueError("parallel target permits copying its initial value")
    elif expected != initial_outputs:
        raise ValueError("readout control changed a queried state")
    if len(item["prompt"]) != item["prompt_char_count"]:
        raise ValueError("prompt character count mismatch")
    if runtime.prompt_sha256(item["prompt"]) != item["prompt_sha256"]:
        raise ValueError("prompt hash mismatch")


def _operation_skeleton(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in operation.items()
        if key not in {"entity", "query_relevant"}
    }


def build_bank() -> list[dict[str, Any]]:
    bank = [
        build_problem(replicate_id, target_count, condition)
        for replicate_id in REPLICATES
        for target_count in TARGET_LEVELS
        for condition in CONDITIONS
    ]
    validate_bank(bank)
    return bank


def validate_bank(bank: list[dict[str, Any]]) -> None:
    if len(bank) != 120 or len({item["item_id"] for item in bank}) != 120:
        raise ValueError("calibrated bank must contain 120 unique items")
    if len({item["prompt_sha256"] for item in bank}) != 120:
        raise ValueError("calibrated prompts must be unique")
    for item in bank:
        validate_problem(item)
    for replicate_id in REPLICATES:
        rows = [item for item in bank if item["replicate_id"] == replicate_id]
        if len(rows) != 6:
            raise ValueError("each trajectory must contain six cells")
        if len({row["trajectory_sha256"] for row in rows}) != 1:
            raise ValueError("all M cells must derive from one trajectory")
        for m in TARGET_LEVELS:
            pair = [row for row in rows if row["target_count"] == m]
            if {row["condition"] for row in pair} != set(CONDITIONS):
                raise ValueError("matched condition pair is incomplete")
            if len({row["prompt_char_count"] for row in pair}) != 1:
                raise ValueError("parallel and readout prompts are not length matched")
            if [_operation_skeleton(row) for row in pair[0]["operations"]] != [
                _operation_skeleton(row) for row in pair[1]["operations"]
            ]:
                raise ValueError("parallel and readout operation skeletons differ")


def bank_fingerprint(bank: Iterable[dict[str, Any]]) -> str:
    return runtime.bank_fingerprint(bank)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong calibrated multi-target manifest")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="One exact frozen ollama:<tag>")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--ollama-base-url", default=runtime.DEFAULT_OLLAMA_BASE_URL
    )
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.execute and args.output is None:
        parser.error("--execute requires --output")
    try:
        args.ollama_base_url = runtime.validate_loopback_url(args.ollama_base_url)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if manifest.get("status") != "frozen_before_inference":
        raise SystemExit("manifest is not frozen before inference")
    runtime.verify_pinned_sources(manifest)
    manifest_file_hash = runtime.file_sha256(args.manifest)
    specifications = {entry["model"]: entry for entry in manifest["model_panel"]}
    if args.model not in specifications:
        raise SystemExit(f"model outside frozen panel: {args.model}")
    model_spec = specifications[args.model]
    runtime.register_and_verify_model(args.model, model_spec)
    bank = build_bank()
    observed_hash = bank_fingerprint(bank)
    expected_hash = manifest["design"]["bank_sha256"]
    if observed_hash != expected_hash:
        raise SystemExit(f"bank drift: {observed_hash} != {expected_hash}")
    if len(bank) != int(manifest["design"]["items_per_model_exact"]):
        raise SystemExit("bank request count drift")

    print(f"Runner: {RUNNER_VERSION}")
    print(f"Model: {args.model}")
    print(f"Bank SHA-256: {observed_hash}")
    print(f"Exact scored requests: {len(bank)}")
    print("Unscored exact-M strict-JSON/GPU preflight requests: 3")
    print(f"GPU required: {not args.allow_cpu}")
    if not args.execute:
        print("Dry-run only; no output file was created.")
        return

    preflights = []
    for m in TARGET_LEVELS:
        expected = list(range(7, 7 + m))
        preflights.append(
            {
                "schema_key": f"answers_m{m}",
                "answer_schema": answer_schema(m),
                "prompt": (
                    f"Return the integer array {json.dumps(expected)} in the "
                    "required JSON object."
                ),
                "expected": expected,
            }
        )
    runtime.run_shard(
        experiment_id=EXPERIMENT_ID,
        runner_version=RUNNER_VERSION,
        model_spec=model_spec,
        bank=bank,
        manifest=manifest,
        manifest_file_hash=manifest_file_hash,
        output=args.output,
        resume=args.resume,
        base_url=args.ollama_base_url,
        allow_cpu=args.allow_cpu,
        parser=parse_json_answers,
        scorer=score_answers,
        preflight_specs=preflights,
    )
    print(f"Wrote checkpointed shard to {args.output}")


if __name__ == "__main__":
    main()
