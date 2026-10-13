#!/usr/bin/env python3
"""Frozen randomized-checkpoint numeric control.

Each of twenty seeded trajectories contains exactly nine displayed slots:
three same-sign unit updates to the queried entity, four active distractors
disjoint from that entity, and two inactive fillers.  Three yoked items ask
for the state immediately after the first, second, or third relevant update.
The default invocation only validates the frozen bank and model artifact.
``--execute`` performs one strict scalar-JSON/GPU preflight and then issues
each scored request once with reserve-before-request atomic checkpointing.
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


RUNNER_VERSION = "checkpoint-query-v1.0"
EXPERIMENT_ID = "wmf-am-checkpoint-query-v1"
MANIFEST_PATH = (
    ROOT / "checkpoint_query_manifest_v1.json"
)
REPLICATES = tuple(range(20))
QUERY_AFTER_LEVELS = (1, 2, 3)
DISPLAY_SLOTS = 9
RELEVANT_UPDATES = 3
ACTIVE_DISTRACTORS = 4
INACTIVE_FILLERS = 2
MASTER_SEED = 27072411

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


# The twenty trajectory seeds. They were derived once, when the bank was first generated, from a SHA-256
# hash of the experiment label of that time, MASTER_SEED and the replicate index. The label has since been
# renamed, so the derived values are listed here unchanged.
REPLICATE_SEEDS = (
    7946566641827233022, 760423692187966593, 7058057792846994041, 15133618255703534602,
    12153330640221897368, 5272563116954996378, 8844833269021964324, 10966383051008618007,
    10772302575147109105, 594438210244992050, 17132977871288340713, 17537739030626363720,
    2375131591634198123, 13220201663522620857, 12401366404991850401, 6459694271510638288,
    1634506432302729108, 6089366941890424628, 17475096466284541904, 16475088234772050125,
)


def replicate_seed(replicate_id: int) -> int:
    if replicate_id not in REPLICATES:
        raise ValueError("replicate_id must be in 0..19")
    return REPLICATE_SEEDS[replicate_id]


def _apply(state: dict[str, int], operation: dict[str, Any]) -> None:
    if int(operation["active"]) == 0:
        return
    amount = int(operation["amount"])
    if operation["kind"] == "gain":
        state[operation["entity"]] += amount
    elif operation["kind"] == "loss":
        state[operation["entity"]] -= amount
    elif operation["kind"] == "transfer":
        state[operation["giver"]] -= amount
        state[operation["receiver"]] += amount
    else:
        raise ValueError(f"unknown operation kind: {operation['kind']}")


def _render(operation: dict[str, Any], slot: int) -> str:
    prefix = f"  {slot:02d} | active={operation['active']} | "
    if operation["kind"] == "gain":
        body = f"{operation['entity']} gains {operation['amount']} point."
    elif operation["kind"] == "loss":
        body = f"{operation['entity']} loses {operation['amount']} point."
    else:
        body = (
            f"{operation['giver']} gives {operation['amount']} point to "
            f"{operation['receiver']}."
        )
    return prefix + body


def _make_trajectory(replicate_id: int) -> dict[str, Any]:
    rng = random.Random(replicate_seed(replicate_id))
    entities = [f"E{index}" for index in range(5)]
    query_entity = "E0"
    initial_state = dict(zip(entities, rng.sample(range(100, 150), len(entities))))
    sign = rng.choice((-1, 1))
    relevant = [
        {
            "operation_id": f"R{index}",
            "role": "relevant",
            "kind": "gain" if sign > 0 else "loss",
            "entity": query_entity,
            "amount": 1,
            "active": 1,
            "target_update_index": index,
        }
        for index in range(1, RELEVANT_UPDATES + 1)
    ]

    distractors: list[dict[str, Any]] = []
    others = entities[1:]
    for index in range(1, ACTIVE_DISTRACTORS + 1):
        kind = rng.choice(("gain", "loss", "transfer"))
        operation: dict[str, Any] = {
            "operation_id": f"D{index}",
            "role": "distractor",
            "kind": kind,
            "amount": 1,
            "active": 1,
        }
        if kind in {"gain", "loss"}:
            operation["entity"] = rng.choice(others)
        else:
            operation["giver"], operation["receiver"] = rng.sample(others, 2)
        distractors.append(operation)

    fillers: list[dict[str, Any]] = []
    for index in range(1, INACTIVE_FILLERS + 1):
        fillers.append(
            {
                "operation_id": f"F{index}",
                "role": "filler",
                "kind": rng.choice(("gain", "loss")),
                "entity": rng.choice(others),
                "amount": 1,
                "active": 0,
            }
        )

    # Random interleaving preserves the order R1,R2,R3 so "after update q"
    # has a unique displayed slot while otherwise varying checkpoint location.
    labels = ["R"] * RELEVANT_UPDATES + ["O"] * (
        ACTIVE_DISTRACTORS + INACTIVE_FILLERS
    )
    rng.shuffle(labels)
    other_pool = distractors + fillers
    rng.shuffle(other_pool)
    operations: list[dict[str, Any]] = []
    ri = oi = 0
    for label in labels:
        if label == "R":
            operations.append(dict(relevant[ri]))
            ri += 1
        else:
            operations.append(dict(other_pool[oi]))
            oi += 1

    state = dict(initial_state)
    state_after_slot: dict[int, dict[str, int]] = {}
    checkpoint_slots: dict[int, int] = {}
    checkpoint_answers: dict[int, int] = {}
    for slot, operation in enumerate(operations, start=1):
        _apply(state, operation)
        state_after_slot[slot] = dict(state)
        if operation["role"] == "relevant":
            query_after = int(operation["target_update_index"])
            checkpoint_slots[query_after] = slot
            checkpoint_answers[query_after] = state[query_entity]

    return {
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed(replicate_id),
        "entities": entities,
        "query_entity": query_entity,
        "initial_state": initial_state,
        "same_sign": sign,
        "operations": operations,
        "checkpoint_slots": checkpoint_slots,
        "checkpoint_answers": checkpoint_answers,
        "state_after_slot": state_after_slot,
    }


def build_problem(replicate_id: int, query_after: int) -> dict[str, Any]:
    if query_after not in QUERY_AFTER_LEVELS:
        raise ValueError("query_after must be 1, 2, or 3")
    trajectory = _make_trajectory(replicate_id)
    operations = trajectory["operations"]
    rendered = [_render(operation, slot) for slot, operation in enumerate(operations, 1)]
    initial = ", ".join(
        f"{entity}={trajectory['initial_state'][entity]}"
        for entity in trajectory["entities"]
    )
    checkpoint_slot = trajectory["checkpoint_slots"][query_after]
    prompt = (
        "Track integer point states through the displayed sequence. There are "
        "exactly nine slots. Apply each slot exactly once and in order only "
        "when active=1; ignore active=0 slots.\n\n"
        f"Initial state: {initial}\n\n"
        "Operation slots:\n"
        + "\n".join(rendered)
        + "\n\n"
        f"What was {trajectory['query_entity']} immediately after slot "
        f"{checkpoint_slot:02d} (the {query_after}th active update to "
        f"{trajectory['query_entity']})? Return only the required JSON object."
    )
    answers = trajectory["checkpoint_answers"]
    correct_answer = answers[query_after]
    item = {
        "item_id": f"checkpoint_r{replicate_id:02d}_q{query_after}",
        "task_type": "numeric_checkpoint",
        "replicate_id": replicate_id,
        "replicate_seed": trajectory["replicate_seed"],
        "query_after": query_after,
        "checkpoint_slot": checkpoint_slot,
        "query_entity": trajectory["query_entity"],
        "initial_state": trajectory["initial_state"],
        "same_sign": trajectory["same_sign"],
        "relevant_updates": RELEVANT_UPDATES,
        "active_distractors": ACTIVE_DISTRACTORS,
        "inactive_fillers": INACTIVE_FILLERS,
        "display_slots": DISPLAY_SLOTS,
        "operations": operations,
        "rendered_slots": rendered,
        "checkpoint_slots": answers and trajectory["checkpoint_slots"],
        "checkpoint_answers": answers,
        "initial_answer": trajectory["initial_state"][trajectory["query_entity"]],
        "previous_checkpoint_answer": answers.get(query_after - 1),
        "next_checkpoint_answer": answers.get(query_after + 1),
        "final_answer": answers[3],
        "correct_answer": correct_answer,
        "answer_schema": ANSWER_SCHEMA,
        "prompt": prompt,
        "prompt_char_count": len(prompt),
        "prompt_sha256": runtime.prompt_sha256(prompt),
        "trajectory_sha256": runtime.canonical_sha256(
            {
                "initial_state": trajectory["initial_state"],
                "operations": operations,
            }
        ),
    }
    validate_problem(item)
    return item


def validate_problem(item: dict[str, Any]) -> None:
    if item.get("task_type") != "numeric_checkpoint":
        raise ValueError("wrong checkpoint task label")
    if item["replicate_id"] not in REPLICATES:
        raise ValueError("replicate outside frozen bank")
    if item["query_after"] not in QUERY_AFTER_LEVELS:
        raise ValueError("query level outside frozen grid")
    operations = item["operations"]
    if len(operations) != DISPLAY_SLOTS or item["display_slots"] != DISPLAY_SLOTS:
        raise ValueError("checkpoint item must display exactly nine slots")
    active_relevant = [
        operation
        for operation in operations
        if operation["active"] == 1 and operation["role"] == "relevant"
    ]
    active_distractors = [
        operation
        for operation in operations
        if operation["active"] == 1 and operation["role"] == "distractor"
    ]
    inactive_fillers = [
        operation
        for operation in operations
        if operation["active"] == 0 and operation["role"] == "filler"
    ]
    if (
        len(active_relevant) != RELEVANT_UPDATES
        or len(active_distractors) != ACTIVE_DISTRACTORS
        or len(inactive_fillers) != INACTIVE_FILLERS
    ):
        raise ValueError("frozen role counts drifted")
    query = item["query_entity"]
    signs = {
        1 if operation["kind"] == "gain" else -1
        for operation in active_relevant
    }
    if len(signs) != 1 or signs != {int(item["same_sign"])}:
        raise ValueError("relevant updates are not all same-sign")
    if any(operation["amount"] != 1 for operation in operations):
        raise ValueError("checkpoint task is frozen to unit arithmetic")
    for operation in active_distractors:
        affected = (
            {operation["entity"]}
            if operation["kind"] in {"gain", "loss"}
            else {operation["giver"], operation["receiver"]}
        )
        if query in affected:
            raise ValueError("active distractor touches queried entity")

    state = dict(item["initial_state"])
    observed_slots: dict[int, int] = {}
    observed_answers: dict[int, int] = {}
    for slot, operation in enumerate(operations, 1):
        _apply(state, operation)
        if operation["role"] == "relevant":
            level = int(operation["target_update_index"])
            observed_slots[level] = slot
            observed_answers[level] = state[query]
    if observed_slots != {int(key): value for key, value in item["checkpoint_slots"].items()}:
        raise ValueError("checkpoint slot metadata is not executable")
    if observed_answers != {
        int(key): value for key, value in item["checkpoint_answers"].items()
    }:
        raise ValueError("checkpoint answer metadata is not executable")
    if any(value == item["initial_answer"] for value in observed_answers.values()):
        raise ValueError("a relevant prefix permits copying the initial state")
    if item["correct_answer"] != observed_answers[item["query_after"]]:
        raise ValueError("stored checkpoint answer is wrong")
    if item["checkpoint_slot"] != observed_slots[item["query_after"]]:
        raise ValueError("queried slot is not the requested target-update checkpoint")
    if len(item["prompt"]) != item["prompt_char_count"]:
        raise ValueError("prompt character count mismatch")
    if runtime.prompt_sha256(item["prompt"]) != item["prompt_sha256"]:
        raise ValueError("prompt hash mismatch")
    if item["answer_schema"] != ANSWER_SCHEMA:
        raise ValueError("answer schema drift")


def build_bank() -> list[dict[str, Any]]:
    bank = [
        build_problem(replicate_id, query_after)
        for replicate_id in REPLICATES
        for query_after in QUERY_AFTER_LEVELS
    ]
    validate_bank(bank)
    return bank


def validate_bank(bank: list[dict[str, Any]]) -> None:
    if len(bank) != 60 or len({item["item_id"] for item in bank}) != 60:
        raise ValueError("checkpoint bank must contain exactly 60 unique items")
    if len({item["prompt_sha256"] for item in bank}) != 60:
        raise ValueError("checkpoint bank prompts must be unique")
    for item in bank:
        validate_problem(item)
    for replicate_id in REPLICATES:
        rows = [item for item in bank if item["replicate_id"] == replicate_id]
        if len(rows) != 3:
            raise ValueError("each trajectory must have three checkpoint queries")
        if len({row["trajectory_sha256"] for row in rows}) != 1:
            raise ValueError("checkpoint queries do not share one trajectory")
        if len({row["prompt_char_count"] for row in rows}) != 1:
            raise ValueError("checkpoint query prompts are not length matched")
        if {row["query_after"] for row in rows} != set(QUERY_AFTER_LEVELS):
            raise ValueError("checkpoint query levels are incomplete")


def bank_fingerprint(bank: Iterable[dict[str, Any]]) -> str:
    return runtime.bank_fingerprint(bank)


def parse_json_answer(
    response: str | None, item: dict[str, Any] | None = None
) -> int | None:
    del item
    if not isinstance(response, str) or not response.strip():
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"answer"}:
        return None
    answer = parsed["answer"]
    if isinstance(answer, bool) or not isinstance(answer, int):
        return None
    return answer


def score_answer(predicted: Any, item: dict[str, Any]) -> dict[str, Any]:
    return {"accurate": int(predicted == item["correct_answer"])}


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong checkpoint-query manifest")
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
    print("Unscored strict-JSON/GPU preflight requests: 1")
    print(f"GPU required: {not args.allow_cpu}")
    if not args.execute:
        print("Dry-run only; no output file was created.")
        return

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
        parser=parse_json_answer,
        scorer=score_answer,
        preflight_specs=[
            {
                "schema_key": "scalar_integer",
                "answer_schema": ANSWER_SCHEMA,
                "prompt": "Return the integer seven in the required JSON object.",
                "expected": 7,
            }
        ],
    )
    print(f"Wrote checkpointed shard to {args.output}")


if __name__ == "__main__":
    main()
