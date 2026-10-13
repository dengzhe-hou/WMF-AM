#!/usr/bin/env bash
# 2026-10-05 copy of ext/api_run_ext.sh for the budget block (author decision). Only BANK and OUT differ.
# WMF-AM evaluation. API side: 8 models through OpenRouter, each pinned to its original vendor.
# Written 2026-10-03. NOT executed when written. `smoke` and `full` send PAID requests.
#
#   bash api_run_commands.sh smoke <label|all>   5 items of bundle/smoke_bank.jsonl per model (40 calls, about $0.01 in total)
#   bash api_run_commands.sh full  <label|all>   the frozen v3/A bank
#
# Labels, in the recommended run order (earliest vendor retirement first; GPT-5.2 last because of Q8):
#   claude-sonnet-4-5-20250929  claude-haiku-4-5-20251001  claude-opus-4-5-20251101
#   gpt-4.1-2025-04-14  gpt-4.1-mini-2025-04-14  command-a-03-2025  command-r7b-12-2024  gpt-5.2-2025-12-11
#
# Smoke pass criteria, fixed before any request is sent. All five must hold for a model:
#   1. the runner exits with code 0 (all 5 smoke items have a final record)
#   2. "error" is null in every record (no failed attempt at all)
#   3. "provider" equals the expected vendor in every record (OpenAI / Anthropic / Cohere)
#   4. usage.completion_tokens_details.reasoning_tokens is 0 (or absent) and no reasoning text in every record
#   5. saved endpoint metadata maps the request id to the expected dated canonical id at the original vendor;
#      "response_model" is that verified request id or canonical id in every record
# 1, 3 and 4 are enforced by the runner itself (exit 2 on violation). 2 and 5 are checked by check_records below.
# The same check runs after every full run (there a failed attempt that was later retried successfully is allowed).
# `smoke all` smokes every model even when an earlier one fails, then lists the failures and exits 1.
# `full all` stops at the first model that fails; rerunning the same command resumes (finished items are kept).
#
# GPT-5.2 (Q8, settled 2026-10-03). Pass the chosen configuration explicitly for this script:
#   Q8_GPT52=openrouter_no_temperature bash api_run_commands.sh full all     run it as below
#   Q8_GPT52=skip                      bash api_run_commands.sh full all     do not run it from this script
# Without Q8_GPT52 the other 7 models run, GPT-5.2 is reported as pending and the script exits with code 3.
# The GPT-5.2 smoke uses the OpenRouter configuration below.
#
# Request fields (checked 2026-10-03 against https://openrouter.ai/docs/guides/routing/provider-selection,
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens and GET /api/v1/models/<id>/endpoints):
#   provider.only=["<vendor>"]        base slug; does not match the service-tier endpoints openai/flex and openai/fast
#   provider.allow_fallbacks=false    no other provider is tried
#   provider.require_parameters=true  the request is routed only to an endpoint that supports every parameter sent
#   temperature=0                     7 models. Not sent for GPT-5.2: none of its endpoints lists `temperature`,
#                                     so with require_parameters=true a request carrying it would find no endpoint
#   reasoning.effort="none"           GPT-5.2 only (OpenRouter default_effort is "medium"; "none" is in supported_efforts)
#   max_tokens                        per item from the bank (16 / 32 / 48 / 512; block G = 16, option A); 16 for smoke
# Thinking is never requested for Claude or Cohere; the runner aborts if reasoning tokens or text appear.
#
# Cost estimate of full (bank v3/A, prices checked 2026-10-03): all 8 models,
# $4.28597 typical output / $6.32717 if every output token allowance is used.
# Inputs: approximately 160,763 tokens per model (o200k_base plus message overhead).
# Outputs: typical 22,400 tokens per model; sum of per-item allowances 48,320.
# These are estimates, not a hard spending cap; retries, smoke tests and fees are extra.
# Per-model rates, original-vendor endpoint sources and assumptions:
#   ../localrun/API_COST_CURRENT_20261003.md
# The earlier G=1024 estimates no longer apply to this frozen bank.
#

# The runner prints the number of correct answers at the end of a run. In `full` mode this one line is kept
# out of the terminal and stays in <label>.jsonl.stdout.log, so scores are not seen before the analysis plan is settled.

set -euo pipefail

FROZEN=/path/to/repo/frozen
RUNNER=$FROZEN/bundle/run_wmf_am_bank.py
API_CHECKER=$FROZEN/check_api_records.py
BANK=/path/to/repo/budget_20261005/budget_bank.jsonl   # budget block: the 60 A items at 512 tokens
SMOKE_BANK=$FROZEN/bundle/smoke_bank.jsonl
OUT=/path/to/repo/results/budget_20261005/api
ENV_FILE=/path/to/earlier_repo/.env
BASE_URL=https://openrouter.ai/api/v1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PENDING_Q8=0

ORDER=(
  claude-sonnet-4-5-20250929   # deprecated 2026-09-30, retires 2026-11-30
  claude-haiku-4-5-20251001    # active, retirement "not sooner than October 15, 2026"
  claude-opus-4-5-20251101     # active, retirement "not sooner than November 24, 2026"
  gpt-4.1-2025-04-14
  gpt-4.1-mini-2025-04-14
  command-a-03-2025
  command-r7b-12-2024
  gpt-5.2-2025-12-11           # last: its full run waits for Q8
)

PIN_OPENAI='{"provider":{"only":["openai"],"allow_fallbacks":false,"require_parameters":true}}'
PIN_ANTHROPIC='{"provider":{"only":["anthropic"],"allow_fallbacks":false,"require_parameters":true}}'
PIN_COHERE='{"provider":{"only":["cohere"],"allow_fallbacks":false,"require_parameters":true}}'
PIN_OPENAI_GPT52='{"provider":{"only":["openai"],"allow_fallbacks":false,"require_parameters":true},"reasoning":{"effort":"none"}}'

# label = vendor id used by BFCL. MODEL = OpenRouter id sent in the request.
# EXPECT_MODEL = dated canonical id verified in the selected provider's endpoint metadata.
# Responses may carry either that canonical id or the request alias MODEL.
spec() {
  case "$1" in
    claude-sonnet-4-5-20250929) MODEL=anthropic/claude-sonnet-4.5; EXPECT_MODEL=anthropic/claude-4.5-sonnet-20250929; PROVIDER=Anthropic; TEMP=0; EXTRA=$PIN_ANTHROPIC ;;
    claude-haiku-4-5-20251001)  MODEL=anthropic/claude-haiku-4.5;  EXPECT_MODEL=anthropic/claude-4.5-haiku-20251001;  PROVIDER=Anthropic; TEMP=0; EXTRA=$PIN_ANTHROPIC ;;
    claude-opus-4-5-20251101)   MODEL=anthropic/claude-opus-4.5;   EXPECT_MODEL=anthropic/claude-4.5-opus-20251124;   PROVIDER=Anthropic; TEMP=0; EXTRA=$PIN_ANTHROPIC ;;
    gpt-4.1-2025-04-14)         MODEL=openai/gpt-4.1;              EXPECT_MODEL=openai/gpt-4.1-2025-04-14;            PROVIDER=OpenAI;    TEMP=0; EXTRA=$PIN_OPENAI ;;
    gpt-4.1-mini-2025-04-14)    MODEL=openai/gpt-4.1-mini;         EXPECT_MODEL=openai/gpt-4.1-mini-2025-04-14;       PROVIDER=OpenAI;    TEMP=0; EXTRA=$PIN_OPENAI ;;
    command-a-03-2025)          MODEL=cohere/command-a;            EXPECT_MODEL=cohere/command-a-03-2025;             PROVIDER=Cohere;    TEMP=0; EXTRA=$PIN_COHERE ;;
    command-r7b-12-2024)        MODEL=cohere/command-r7b-12-2024;  EXPECT_MODEL=cohere/command-r7b-12-2024;           PROVIDER=Cohere;    TEMP=0; EXTRA=$PIN_COHERE ;;
    gpt-5.2-2025-12-11)         MODEL=openai/gpt-5.2;              EXPECT_MODEL=openai/gpt-5.2-20251211;              PROVIDER=OpenAI;    TEMP=none; EXTRA=$PIN_OPENAI_GPT52 ;;
    *) echo "unknown label: $1" >&2; exit 64 ;;
  esac
}

# Deterministic checks on a records file. Prints no accuracy.
check_records() {  # records.jsonl bank.jsonl endpoints.json request_model canonical_model provider smoke|full
  python3 "$API_CHECKER" --records "$1" --bank "$2" --endpoints "$3" \
    --request-model "$4" --canonical-model "$5" --provider "$6" --mode "$7" \
    --report "$1.checks.json"
}

run_one() {  # smoke|full label
  local mode=$1 label=$2 bank out concurrency endpoints started ended
  local -a run_status
  spec "$label"

  if [[ $label == gpt-5.2-2025-12-11 ]]; then
    case "${Q8_GPT52:-undecided}" in
      openrouter_no_temperature) ;;
      skip) echo "== $label: skipped (Q8_GPT52=skip)"; return 0 ;;
      undecided)
        if [[ $mode == full ]]; then
          echo "== $label: NOT RUN. Q8 is undecided. Set Q8_GPT52=openrouter_no_temperature or Q8_GPT52=skip."
          PENDING_Q8=1
          return 0
        fi ;;
      *) echo "Q8_GPT52 must be openrouter_no_temperature or skip, got: $Q8_GPT52" >&2; exit 64 ;;
    esac
  fi

  if [[ $mode == smoke ]]; then
    bank=$SMOKE_BANK; out=$OUT/smoke/$label.$STAMP.jsonl; concurrency=1
  else
    bank=$BANK; out=$OUT/$label.jsonl; concurrency=4
  fi
  mkdir -p "$(dirname "$out")" "$OUT/provenance"

  echo "== $mode $label  request model=$MODEL  canonical model=$EXPECT_MODEL  provider=$PROVIDER"
  # Public endpoint list at run time (GET, no key): the record of which dated version the alias pointed to.
  endpoints=$OUT/provenance/$label.endpoints.$STAMP.json
  curl -sS --fail -m 30 "$BASE_URL/models/$MODEL/endpoints" -o "$endpoints"
  # Check the selected original vendor and dated version before sending any paid request.
  python3 "$API_CHECKER" --endpoints "$endpoints" --request-model "$MODEL" \
    --canonical-model "$EXPECT_MODEL" --provider "$PROVIDER"

  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if PYTHONUNBUFFERED=1 python3 "$RUNNER" --bank "$bank" --out "$out" --label "$label" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key-env OPENROUTER_API_KEY \
      --max-tokens 16 --temperature "$TEMP" --expect-provider "$PROVIDER" \
      --concurrency "$concurrency" --extra-body "$EXTRA" 2>&1 \
    | tee -a "$out.stdout.log" \
    | { if [[ $mode == full ]]; then sed '/correct_last_int=/d'; else cat; fi; }; then
    run_status=("$?" "${PIPESTATUS[@]}")
  else
    run_status=("$?" "${PIPESTATUS[@]}")
  fi
  ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 - "$out.timing.jsonl" "$started" "$ended" "${run_status[0]}" "${run_status[1]}" "$mode" "$label" <<'PY'
import json
import sys
from datetime import datetime

path, started, ended, pipeline_exit, runner_exit, mode, label = sys.argv[1:]
elapsed = (datetime.fromisoformat(ended.replace("Z", "+00:00"))
           - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds()
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "started_utc": started, "ended_utc": ended, "elapsed_seconds": elapsed,
        "runner_exit_code": int(runner_exit), "pipeline_exit_code": int(pipeline_exit),
        "mode": mode, "label": label,
    }) + "\n")
PY
  if (( run_status[0] != 0 )); then
    return "${run_status[0]}"
  fi

  check_records "$out" "$bank" "$endpoints" "$MODEL" "$EXPECT_MODEL" "$PROVIDER" "$mode"
  echo "== $mode $label: all checks passed -> $out"
}

main() {
  local mode=${1:-} target=${2:-} label
  if [[ $mode != smoke && $mode != full ]] || [[ -z $target ]]; then
    sed -n '2,10p' "$0" >&2
    exit 64
  fi
  if [[ $mode == smoke && $target == all ]]; then
    # One process per model, so that a failure stops that model only.
    local failed=()
    for label in "${ORDER[@]}"; do
      bash "$0" smoke "$label" || failed+=("$label")
    done
    if (( ${#failed[@]} )); then
      echo "SMOKE FAILED for: ${failed[*]}" >&2
      exit 1
    fi
    echo "SMOKE: no failure."
    return 0
  fi
  set -a; source "$ENV_FILE"; set +a

  local labels=("$target")
  if [[ $target == all ]]; then labels=("${ORDER[@]}"); fi
  for label in "${labels[@]}"; do
    run_one "$mode" "$label"
  done

  if (( PENDING_Q8 )); then
    echo "API side INCOMPLETE: gpt-5.2-2025-12-11 is waiting for the Q8 decision." >&2
    exit 3
  fi
}

main "$@"
