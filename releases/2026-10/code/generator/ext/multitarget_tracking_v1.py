#!/usr/bin/env python3
"""Frozen multi-target tracking and matched readout-control experiment.

For M in {1, 2, 4}, the parallel condition requires updating M queried states.
Its matched readout control requests the same M outputs but redirects every
otherwise matched update to shadow entities, leaving queried states unchanged.
All conditions share twelve active displayed operations, initial states, slot
order, signs, and one-digit magnitudes.  This separates multi-target update
tracking from JSON output-length/readout burden without claiming a latent
mechanistic measure.

The default invocation validates the frozen bank without inference.  Execution
uses exact-M JSON schemas, three schema/GPU preflights, one request per item,
and reserve-before-request atomic checkpointing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import checkpoint_multitarget_runtime_v1 as runtime  # noqa: E402


RUNNER_VERSION = "multitarget-tracking-v1.0"
EXPERIMENT_ID = "wmf-am-multitarget-tracking-v1"
MANIFEST_PATH = (
    ROOT / "multitarget_tracking_manifest_v1.json"
)
REPLICATES = tuple(range(20))
TARGET_LEVELS = (1, 2, 4)
CONDITIONS = ("parallel_tracking", "readout_control")
DISPLAY_SLOTS = 12
LANES = 4
UPDATES_PER_LANE = 2
ACTIVE_DISTRACTORS = 4
MASTER_SEED = 27072421


def answer_schema(target_count: int) -> dict[str, Any]:
    if target_count not in TARGET_LEVELS:
        raise ValueError("target_count must be 1, 2, or 4")
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": target_count,
                "maxItems": target_count,
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


# The twenty trajectory seeds. They were derived once, when the bank was first generated, from a SHA-256
# hash of the experiment label of that time, MASTER_SEED and the replicate index. The label has since been
# renamed, so the derived values are listed here unchanged.
REPLICATE_SEEDS = (
    10748805906976498200, 9108726611265226409, 2667065808058574191, 11239429967004865623,
    17111569250092013843, 10269009388769635589, 3772349186765994568, 5186500984340979299,
    15902834079591739867, 17725048762966396516, 851448512711341746, 2548845145191707503,
    6800174994930950500, 14853041029041278054, 973957725586278140, 3954552550196571748,
    8524542584573276064, 5168853822788486868, 3394282046341227142, 6601625612775739620,
)


def replicate_seed(replicate_id: int) -> int:
    if replicate_id not in REPLICATES:
        raise ValueError("replicate_id must be in 0..19")
    return REPLICATE_SEEDS[replicate_id]


def _apply(state: dict[str, int], operation: dict[str, Any]) -> None:
    amount = int(operation["amount"])
    if operation["kind"] == "gain":
        state[operation["entity"]] += amount
    elif operation["kind"] == "loss":
        state[operation["entity"]] -= amount
    else:
        raise ValueError(f"unknown operation kind: {operation['kind']}")


def _render(operation: dict[str, Any], slot: int) -> str:
    verb = "gains" if operation["kind"] == "gain" else "loses"
    return (
        f"  {slot:02d} | {operation['entity']} {verb} "
        f"{operation['amount']} points."
    )


def _make_trajectory(replicate_id: int) -> dict[str, Any]:
    rng = random.Random(replicate_seed(replicate_id))
    query_entities = [f"E{index}" for index in range(LANES)]
    shadow_entities = [f"U{index}" for index in range(LANES)]
    noise_entities = [f"V{index}" for index in range(LANES)]
    entities = query_entities + shadow_entities + noise_entities
    initial_state = dict(zip(entities, rng.sample(range(100, 220), len(entities))))

    lane_templates: list[dict[str, Any]] = []
    for lane in range(LANES):
        while True:
            signs = [rng.choice((-1, 1)), rng.choice((-1, 1))]
            amounts = [rng.randint(2, 9), rng.randint(2, 9)]
            prefixes = [
                signs[0] * amounts[0],
                signs[0] * amounts[0] + signs[1] * amounts[1],
            ]
            if all(prefix_value != 0 for prefix_value in prefixes):
                break
        for step, (sign, amount) in enumerate(zip(signs, amounts), start=1):
            lane_templates.append(
                {
                    "operation_id": f"L{lane}_{step}",
                    "role": "lane_update",
                    "lane": lane,
                    "step": step,
                    "kind": "gain" if sign > 0 else "loss",
                    "amount": amount,
                }
            )

    distractor_templates: list[dict[str, Any]] = []
    for index in range(ACTIVE_DISTRACTORS):
        distractor_templates.append(
            {
                "operation_id": f"D{index}",
                "role": "distractor",
                "kind": rng.choice(("gain", "loss")),
                "amount": rng.randint(2, 9),
                "entity": noise_entities[index],
            }
        )

    # Random interleaving with each lane's step 1 before step 2.  The same
    # operation-ID order is used in every M/condition cell of this trajectory.
    pending = {
        lane: [
            template
            for template in lane_templates
            if int(template["lane"]) == lane
        ]
        for lane in range(LANES)
    }
    remaining_distractors = list(distractor_templates)
    slot_templates: list[dict[str, Any]] = []
    while any(pending.values()) or remaining_distractors:
        choices = [
            ("lane", lane) for lane, rows in pending.items() if rows
        ] + ([("distractor", -1)] if remaining_distractors else [])
        kind, selected = rng.choice(choices)
        if kind == "lane":
            slot_templates.append(pending[selected].pop(0))
        else:
            slot_templates.append(remaining_distractors.pop(0))

    return {
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed(replicate_id),
        "query_entities": query_entities,
        "shadow_entities": shadow_entities,
        "noise_entities": noise_entities,
        "entities": entities,
        "initial_state": initial_state,
        "slot_templates": slot_templates,
    }


def _materialize_operations(
    trajectory: dict[str, Any], target_count: int, condition: str
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for template in trajectory["slot_templates"]:
        operation = dict(template)
        if operation["role"] == "lane_update":
            lane = int(operation["lane"])
            if condition == "parallel_tracking" and lane < target_count:
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
    if target_count not in TARGET_LEVELS:
        raise ValueError("target_count outside frozen grid")
    if condition not in CONDITIONS:
        raise ValueError("condition outside frozen grid")
    trajectory = _make_trajectory(replicate_id)
    operations = _materialize_operations(trajectory, target_count, condition)
    query_entities = trajectory["query_entities"][:target_count]
    state = dict(trajectory["initial_state"])
    for operation in operations:
        _apply(state, operation)
    correct_answer = [state[entity] for entity in query_entities]
    rendered = [_render(operation, slot) for slot, operation in enumerate(operations, 1)]
    initial = ", ".join(
        f"{entity}={trajectory['initial_state'][entity]}"
        for entity in trajectory["entities"]
    )
    ordered_targets = ", ".join(query_entities)
    prompt = (
        "Track integer point states through the displayed sequence. Apply all "
        "twelve operations exactly once and in order.\n\n"
        f"Initial state: {initial}\n\n"
        "Operation slots:\n"
        + "\n".join(rendered)
        + "\n\n"
        f"Return the final values for [{ordered_targets}] in exactly this "
        "order as the JSON answers array. Return only the required JSON object."
    )
    item = {
        "item_id": (
            f"multitarget_r{replicate_id:02d}_m{target_count}_{condition}"
        ),
        "task_type": "numeric_multitarget",
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
        "display_slots": DISPLAY_SLOTS,
        "updates_per_lane": UPDATES_PER_LANE,
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
    if item.get("task_type") != "numeric_multitarget":
        raise ValueError("wrong multi-target task label")
    m = int(item["target_count"])
    condition = item["condition"]
    if m not in TARGET_LEVELS or condition not in CONDITIONS:
        raise ValueError("multi-target cell outside frozen grid")
    if len(item["query_entities"]) != m:
        raise ValueError("query output length does not equal M")
    if item["answer_schema"] != answer_schema(m):
        raise ValueError("exact-M answer schema drift")
    operations = item["operations"]
    if len(operations) != DISPLAY_SLOTS or item["display_slots"] != DISPLAY_SLOTS:
        raise ValueError("multi-target item must display exactly twelve slots")
    if any(operation.get("active") != 1 for operation in operations):
        raise ValueError("all displayed multi-target operations must be active")
    lane_rows = [
        operation for operation in operations if operation["role"] == "lane_update"
    ]
    distractors = [
        operation for operation in operations if operation["role"] == "distractor"
    ]
    if len(lane_rows) != LANES * UPDATES_PER_LANE:
        raise ValueError("lane update count drift")
    if len(distractors) != ACTIVE_DISTRACTORS:
        raise ValueError("distractor count drift")
    for lane in range(LANES):
        rows = [row for row in lane_rows if int(row["lane"]) == lane]
        if [int(row["step"]) for row in rows] != [1, 2]:
            raise ValueError("lane update order drift")
        expected_entity = (
            f"E{lane}"
            if condition == "parallel_tracking" and lane < m
            else f"U{lane}"
        )
        if {row["entity"] for row in rows} != {expected_entity}:
            raise ValueError("lane-to-query/shadow mapping drift")
        expected_relevance = int(
            condition == "parallel_tracking" and lane < m
        )
        if {row["query_relevant"] for row in rows} != {expected_relevance}:
            raise ValueError("query-relevance metadata drift")
        signed_prefixes: list[int] = []
        total = 0
        for row in rows:
            total += (1 if row["kind"] == "gain" else -1) * int(row["amount"])
            signed_prefixes.append(total)
        if any(prefix == 0 for prefix in signed_prefixes):
            raise ValueError("lane admits an initial-value cancellation shortcut")

    query_set = set(item["query_entities"])
    if any(operation["entity"] in query_set for operation in distractors):
        raise ValueError("distractor touches a queried entity")
    state = dict(item["initial_state"])
    for operation in operations:
        _apply(state, operation)
    expected = [state[entity] for entity in item["query_entities"]]
    if expected != item["correct_answer"]:
        raise ValueError("stored multi-target answer is not executable")
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


def build_bank() -> list[dict[str, Any]]:
    bank = [
        build_problem(replicate_id, target_count, condition)
        for replicate_id in REPLICATES
        for target_count in TARGET_LEVELS
        for condition in CONDITIONS
    ]
    validate_bank(bank)
    return bank


def _operation_skeleton(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in operation.items()
        if key not in {"entity", "query_relevant"}
    }


def validate_bank(bank: list[dict[str, Any]]) -> None:
    if len(bank) != 120 or len({item["item_id"] for item in bank}) != 120:
        raise ValueError("multi-target bank must contain exactly 120 unique items")
    if len({item["prompt_sha256"] for item in bank}) != 120:
        raise ValueError("multi-target prompts must be unique")
    for item in bank:
        validate_problem(item)
    for replicate_id in REPLICATES:
        rows = [item for item in bank if item["replicate_id"] == replicate_id]
        if len(rows) != 6:
            raise ValueError("each trajectory must contain six frozen cells")
        if len({row["trajectory_sha256"] for row in rows}) != 1:
            raise ValueError("M/condition cells do not share one trajectory")
        for target_count in TARGET_LEVELS:
            pair = [
                row for row in rows if row["target_count"] == target_count
            ]
            if {row["condition"] for row in pair} != set(CONDITIONS):
                raise ValueError("matched condition pair is incomplete")
            if len({row["prompt_char_count"] for row in pair}) != 1:
                raise ValueError("parallel/readout prompts are not length matched")
            if [_operation_skeleton(row) for row in pair[0]["operations"]] != [
                _operation_skeleton(row) for row in pair[1]["operations"]
            ]:
                raise ValueError("parallel/readout operation structure is not yoked")


def bank_fingerprint(bank: Iterable[dict[str, Any]]) -> str:
    return runtime.bank_fingerprint(bank)


def parse_json_answers(
    response: str | None, item: dict[str, Any] | None = None
) -> list[int] | None:
    if not isinstance(response, str) or not response.strip():
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"answers"}:
        return None
    answers = parsed["answers"]
    if not isinstance(answers, list):
        return None
    if item is not None and len(answers) != int(item["target_count"]):
        return None
    if len(answers) not in TARGET_LEVELS:
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) for value in answers):
        return None
    return answers


def score_answers(predicted: Any, item: dict[str, Any]) -> dict[str, Any]:
    m = int(item["target_count"])
    if not isinstance(predicted, list) or len(predicted) != m:
        component_correct = [0] * m
        valid = False
    else:
        component_correct = [
            int(observed == expected)
            for observed, expected in zip(predicted, item["correct_answer"])
        ]
        valid = True
    component_accuracy = sum(component_correct) / m
    joint = int(valid and all(component_correct))
    return {
        "component_correct": component_correct,
        "component_accuracy": component_accuracy,
        "joint_accuracy": joint,
        "accurate": joint,
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong multi-target manifest")
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
