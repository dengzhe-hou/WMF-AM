# WMF-AM local run of ONE model: run protocol checks, run records, and steps 5B to 5F.
# Sourced by run_one_model.sbatch (node-a MIG slice) and run_70b_node-b.sbatch (node-b, 2x RTX PRO 6000).
# The caller sets: LABEL, TP (1 or 2), GPU_UTIL, READY_TIMEOUT_MIN, EXPECT_NGPU.
# RESULTS_ROOT may point to a fresh retry root; existing model directories are never overwritten.
# ALLOW_S3_FAILURE=1 applies the 2026-10-03 exception only to Granite 350M and Llama 1B.
# Their observed S3 remains fail; all other smoke conditions must still pass.
# SERVE_EXTRA_ARGS, when set by the caller, contains additional runtime implementation arguments.
# Follows localtest_granite31/job.sbatch (Job N, S1-S6 passed, all items finished) with these changes:
#   one serve attempt with VLLM_USE_FLASHINFER_SAMPLER=0 and CUDA_HOME from the start (the settings of
#   Job N's successful second attempt); port derived from the array index (two servers can share node-a);
#   S6 reads `ps` of the saved PID; outputs in results/open/<label>/; extra manifest fields; DRY_RUN mode;
#   TP=2 check before the smoke test.
# DRY_RUN=1: login node, no GPU. Checks paths, bank, weights, environment, renders the chat prompt
#   (CPU only), prints the serve, smoke, run, and shutdown commands; starts no server, sends no request.
set -uo pipefail

REPO_ROOT=/path/to/repo
LOCALRUN=$REPO_ROOT/localrun
BUNDLE=${BUNDLE:-$REPO_ROOT/frozen/wmf_am_server_bundle_20261003.tar.gz}
RESULTS=${RESULTS_ROOT:-$REPO_ROOT/results/open}
VENV=/path/to/venv
CUDA_HOME_DIR=/path/to/conda_env/targets/x86_64-linux
DRY=${DRY_RUN:-0}
N_DRY_FAIL=0

# ── Output directory (= ~/wmf_am of the run protocol, one per model) ──────────────
if [ "$DRY" = 1 ]; then
  OUT=$(mktemp -d "${TMPDIR:-/tmp}/wmfam_dryrun.XXXXXX")/$LABEL
else
  [ -n "${SLURM_JOB_ID-}" ] || { echo "Run this through sbatch, or with DRY_RUN=1 on the login node." >&2; exit 1; }
  if [ -d "$RESULTS/$LABEL" ] && [ -n "$(ls -A "$RESULTS/$LABEL")" ]; then
    echo "STOP: $RESULTS/$LABEL already holds files from an earlier job. Nothing is overwritten or resumed here; resuming is your decision." >&2
    exit 1
  fi
  OUT=$RESULTS/$LABEL
fi
mkdir -p "$OUT"/{out,smoke,manifest,logs}
cd "$OUT" || exit 1
: > deviations.txt

tl(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/logs/timeline.log"; }
deviation(){ echo "$*" >> "$OUT/deviations.txt"; tl "DEVIATION: $*"; }
die(){
  tl "FATAL: $*"
  {
    printf 'Run stopped for %s\n\n%s\n' "$LABEL" "$*"
    if [ -f "$OUT/manifest/$LABEL.smoke_result.json" ]; then
      printf '\nSmoke conditions and observations\n'
      cat "$OUT/manifest/$LABEL.smoke_result.json"
    fi
    for log in "smoke_evaluation_$LABEL.log" "smoke_A_$LABEL.log" "smoke_B_$LABEL.log" "vllm_$LABEL.log"; do
      if [ -f "$OUT/logs/$log" ]; then
        printf '\nLast 50 lines of logs/%s\n' "$log"
        tail -50 "$OUT/logs/$log"
      fi
    done
    printf '\nAll files are retained. See logs/timeline.log for the run sequence.\n'
  } > "$OUT/REPORT.md"
  exit 1
}
need(){   # need "what" cmd...: a real job stops on failure; a dry run records it and goes on
  local what=$1; shift
  if "$@"; then tl "ok: $what"; return 0; fi
  if [ "$DRY" = 1 ]; then N_DRY_FAIL=$((N_DRY_FAIL + 1)); tl "DRY_RUN FAIL (a real job stops here): $what"; return 1; fi
  die "$what"
}
show(){ tl "DRY_RUN would run: $*"; }

tl "label=$LABEL tp=$TP out=$OUT node=$(hostname) job=${SLURM_JOB_ID-none} array=${SLURM_ARRAY_JOB_ID-}_${SLURM_ARRAY_TASK_ID-} dry_run=$DRY"
if [ "$DRY" != 1 ]; then
  ln -s "$RESULTS/_slurm/slurm-$SLURM_JOB_ID.out" logs/slurm.out
  scontrol show job "$SLURM_JOB_ID" > logs/scontrol_job.txt 2>&1
fi

# ── Section 1: unpack the bundle into the output directory, check line counts ─
need "bundle exists: $BUNDLE" test -f "$BUNDLE"
tar xzf "$BUNDLE" --strip-components=1 -C "$OUT"
need "item_bank.jsonl has 840 lines" test "$(wc -l < item_bank.jsonl)" = 840
need "smoke_bank.jsonl has 5 lines" test "$(wc -l < smoke_bank.jsonl)" = 5
need "bank matches the G=16 option A" python3 "$LOCALRUN/helper.py" bank item_bank.jsonl
# Extension run: EXT_BANK names an additional item bank. The bundle still supplies the runner,
# the smoke bank and models_open.tsv, and its checks above are unchanged; only the full run uses the other bank.
RUN_BANK=item_bank.jsonl
if [ -n "${EXT_BANK-}" ]; then
  need "extension bank exists: $EXT_BANK" test -f "$EXT_BANK"
  cp "$EXT_BANK" ext_bank.jsonl
  RUN_BANK=ext_bank.jsonl
  tl "extension run: the full run uses ext_bank.jsonl ($(wc -l < ext_bank.jsonl) items) from $EXT_BANK"
fi
export RUN_BANK
export MODELS_TSV=$OUT/models_open.tsv
IFS=$'\t' read -r REPO REV TRUST < <(python3 "$LOCALRUN/helper.py" lookup "$LABEL")
[ -n "${REPO-}" ] || die "label $LABEL not in $MODELS_TSV"
[ "$TRUST" = no ] || die "$LABEL needs trust_remote_code (minicpm3-4b); it is excluded from the local runs (decided 2026-10-03)"
tl "repo=$REPO rev=$REV"

# ── Environment ───────────────────────────────────────────────────────────────
export HF_HUB_CACHE=/path/to/hf_cache
export HF_HUB_OFFLINE=1
need "weights of $LABEL present at $REV" python3 "$LOCALRUN/helper.py" present "$LABEL"
deviation "5A download done beforehand by localrun/download_models.sh into HF_HUB_CACHE=$HF_HUB_CACHE; the job runs with HF_HUB_OFFLINE=1"
# shellcheck disable=SC1091
source "$VENV/bin/activate" || die "cannot activate $VENV"
export OMP_NUM_THREADS=4
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$CUDA_HOME_DIR
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
need "nvcc under CUDA_HOME" test -x "$CUDA_HOME/bin/nvcc"
deviation "OMP_NUM_THREADS=4 (torch on this cluster crashes with more; job has ${SLURM_CPUS_PER_TASK:-?} CPUs)"
deviation "CUDA_HOME=$CUDA_HOME set as in Job N's successful attempt; the run protocol sets only VLLM_USE_FLASHINFER_SAMPLER=0. Neither changes greedy outputs."
deviation "paths: ~/wmf_am is $OUT, venv is $VENV (vLLM 0.30.0, PyPI CUDA 13.0 build)"
if [ "${NCCL_P2P_DISABLE-}" = 1 ]; then
  deviation "node-b communication workaround: NCCL_P2P_DISABLE=1, NCCL_DEBUG=${NCCL_DEBUG-}; see profile_output/comm_20261003/REPORT.md. Collective implementation changes; no claim of bitwise equivalence."
fi

# ── Section 3 step 1 and step 3: hardware and software record ────────────────
if [ "$DRY" = 1 ]; then
  show "nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv; nvidia-smi; nvidia-smi -L; driver >= 580 check"
else
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv | tee logs/nvidia_smi_query_before.csv
  nvidia-smi > logs/nvidia_smi_full_before.txt 2>&1
  nvidia-smi -L > logs/nvidia_smi_L.txt 2>&1
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  tl "driver_version=$DRIVER"
  [ "${DRIVER%%.*}" -ge 580 ] || die "driver $DRIVER is older than 580 but the PyPI CUDA 13 build is installed"
fi
df -h "$HF_HUB_CACHE/" > logs/df.txt
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available(), torch.cuda.device_count())" > logs/env_check.txt 2>&1
ENV_LINE=$(tail -1 logs/env_check.txt)
tl "env check: $ENV_LINE"
need "vLLM 0.30.0 importable" test "${ENV_LINE%% *}" = 0.30.0
[ "$DRY" = 1 ] || [[ "$ENV_LINE" == *" True $EXPECT_NGPU" ]] || die "env check does not end with True $EXPECT_NGPU"
pip freeze > logs/pip_freeze.txt

# ── GPU selection ─────────────────────────────────────────────────────────────
SLURM_CVD="${CUDA_VISIBLE_DEVICES-}"
probe(){ CUDA_VISIBLE_DEVICES="$1" python -c "import torch; ok=torch.cuda.is_available(); n=torch.cuda.device_count() if ok else 0; print(ok, n, [torch.cuda.get_device_name(i) for i in range(n)], [torch.cuda.mem_get_info(i) for i in range(n)])" 2>&1 | tail -1; }
if [ "$DRY" = 1 ]; then
  CVD=$([ "$TP" = 1 ] && echo 0 || echo "<SLURM CUDA_VISIBLE_DEVICES>")
  show "probe CUDA_VISIBLE_DEVICES=0 and the SLURM value; TP=1 uses 0 if it sees exactly one device (as in Job N), TP=2 uses the SLURM value"
elif [ "$TP" = 1 ]; then
  P0=$(probe 0); tl "probe CUDA_VISIBLE_DEVICES=0 -> $P0"
  PS=$(probe "$SLURM_CVD"); tl "probe CUDA_VISIBLE_DEVICES=$SLURM_CVD (SLURM) -> $PS"
  if [[ "$P0" == "True 1 "* ]]; then
    CVD=0
  else
    [[ "$PS" == "True 1 "* ]] || die "no usable GPU with either setting"
    CVD="$SLURM_CVD"
    deviation "5B CUDA_VISIBLE_DEVICES=0 does not select the allocated MIG slice; used the SLURM value $SLURM_CVD"
  fi
else
  CVD="$SLURM_CVD"
  PS=$(probe "$CVD"); tl "probe CUDA_VISIBLE_DEVICES=$CVD (SLURM) -> $PS"
  [[ "$PS" == "True $TP "* ]] || die "SLURM CUDA_VISIBLE_DEVICES=$CVD does not give $TP usable GPUs"
  [ "$CVD" = 0,1 ] || deviation "5B CUDA_VISIBLE_DEVICES=$CVD from SLURM instead of 0,1"
fi

# ── Port: 8000 + 10 x array index, first free of ten; two servers can share node-a ─
BASE=$((8000 + 10 * ${SLURM_ARRAY_TASK_ID:-0}))
PORT=
for p in $(seq "$BASE" $((BASE + 9))); do
  if python -c "import socket,sys; s=socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1])))" "$p" 2>/dev/null; then PORT=$p; break; fi
done
[ -n "$PORT" ] || die "ports $BASE-$((BASE + 9)) all taken"
[ "$PORT" = 8000 ] || deviation "5B port $PORT instead of 8000 (up to two servers share node-a; port = 8000 + 10 x array index, first free)"
tl "port=$PORT"

# ── B. Serve ─────────────────────────────────────────────────────────────────
SERVE_ARGS=(serve "$REPO" --revision "$REV" --served-model-name "$LABEL" --dtype bfloat16)
[ "$TP" = 1 ] || SERVE_ARGS+=(--tensor-parallel-size "$TP")
SERVE_ARGS+=(--generation-config vllm --max-model-len 4096 --gpu-memory-utilization "$GPU_UTIL" --seed 0 --host 127.0.0.1 --port "$PORT")
SERVE_ARGS+=("${SERVE_EXTRA_ARGS[@]}")
SERVE_CMD="VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=$CUDA_HOME CUDA_VISIBLE_DEVICES=$CVD vllm ${SERVE_ARGS[*]} > logs/vllm_$LABEL.log 2>&1 &"
if [ "${NCCL_P2P_DISABLE-}" = 1 ]; then
  SERVE_CMD="NCCL_P2P_DISABLE=1 NCCL_DEBUG=${NCCL_DEBUG-} $SERVE_CMD"
fi
PIDFILE=$OUT/logs/vllm_$LABEL.pid
RUN=(python run_wmf_am_bank.py --base-url "http://127.0.0.1:$PORT/v1" --model "$LABEL" --max-tokens 16 --temperature 0 --concurrency 1)

if [ "$DRY" = 1 ]; then
  show "$SERVE_CMD ; echo \$! > $PIDFILE ; wait up to $READY_TIMEOUT_MIN min for $LABEL in /v1/models"
  show "grep the vLLM log for: FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0."
  [ "$TP" = 1 ] || show "TP=2 check (before the smoke test): 2 GPUs in nvidia-smi, each memory.used >= 61440 MiB; engine line has tensor_parallel_size=2; a compute process on each GPU UUID"
else
  wait_ready(){   # 0 = ready, 1 = process exited, 2 = timeout
    local pid; pid=$(cat "$PIDFILE")
    for _ in $(seq 1 $((READY_TIMEOUT_MIN * 12))); do
      if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$LABEL\""; then return 0; fi
      kill -0 "$pid" 2>/dev/null || return 1
      sleep 5
    done
    return 2
  }
  cleanup(){
    [ -n "${MON_PID-}" ] && kill "$MON_PID" 2>/dev/null
    if [ -f "$PIDFILE" ]; then
      local pid; pid=$(cat "$PIDFILE")
      kill -0 "$pid" 2>/dev/null && { tl "cleanup: kill server $pid"; kill "$pid"; }
    fi
  }
  trap cleanup EXIT

  # GPU monitor every 30 s (not part of the run protocol; memory and both-GPU evidence).
  ( while true; do
      echo "$(date -u +%FT%TZ) gpu: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | tr '\n' ';') apps: $(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>&1 | tr '\n' ';')"
      sleep 30
    done ) > logs/gpu_monitor.log 2>&1 &
  MON_PID=$!

  tl "serve: $SERVE_CMD"
  T_LAUNCH=$(date +%s)
  CUDA_VISIBLE_DEVICES=$CVD vllm "${SERVE_ARGS[@]}" > "$OUT/logs/vllm_$LABEL.log" 2>&1 &
  echo $! > "$PIDFILE"
  wait_ready; RC=$?
  if [ "$RC" != 0 ]; then
    tl "server not ready (rc=$RC: 1 = exited, 2 = $READY_TIMEOUT_MIN min timeout); last 50 log lines follow"
    tail -50 "logs/vllm_$LABEL.log"
    die "server not ready; no smoke test, no bank run (section 5B)"
  fi
  LOAD_SECONDS=$(( $(date +%s) - T_LAUNCH ))
  tl "server ready, load_seconds=$LOAD_SECONDS"
  printf '%s\n' "$SERVE_CMD" > "logs/serve_command_$LABEL.txt"
  curl -s "http://127.0.0.1:$PORT/v1/models" > "logs/v1_models_$LABEL.json"
  nvidia-smi > logs/nvidia_smi_full_loaded.txt 2>&1
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader,nounits > logs/nvidia_smi_gpus_loaded.csv 2>&1
  nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader,nounits > logs/nvidia_smi_apps_loaded.csv 2>&1
  grep -q "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0." "logs/vllm_$LABEL.log" \
    || die "vLLM log lacks the FlashInfer-sampler-disabled line (section 5B)"

  TP_CHECK=
  if [ "$TP" != 1 ]; then
    # Pre-stated TP=2 check, fixed before any run (TP=2 was never tested on node-b). PASS only if
    #  (a) nvidia-smi lists exactly 2 GPUs and each has memory.used >= 61440 MiB (each rank holds about
    #      65.7 GiB of the 131.4 GiB of weights, plus KV cache);
    #  (b) the vLLM engine line says tensor_parallel_size=2;
    #  (c) nvidia-smi lists at least one compute process on each of the 2 GPU UUIDs.
    # On FAIL: no smoke test and no bank run; the server is shut down and the job stops.
    TP_CHECK=$(python - "logs/nvidia_smi_gpus_loaded.csv" "logs/nvidia_smi_apps_loaded.csv" "logs/vllm_$LABEL.log" <<'EOF'
import sys
gpus = [l.split(", ") for l in open(sys.argv[1]).read().strip().splitlines()]
apps = [l.split(", ") for l in open(sys.argv[2]).read().strip().splitlines() if l.strip()]
log = open(sys.argv[3], errors="replace").read()
a = len(gpus) == 2 and all(int(g[3]) >= 61440 for g in gpus)
b = "tensor_parallel_size=2," in log
c = len(gpus) == 2 and all(any(app[1] == g[1] for app in apps) for g in gpus)
mem = [int(g[3]) for g in gpus]
print(f"{'PASS' if a and b and c else 'FAIL'} a_two_gpus_each_ge_61440MiB={a} (memory.used MiB {mem}) b_engine_tp2={b} c_process_on_each_gpu={c}")
EOF
)
    tl "TP=2 check: $TP_CHECK"
    [[ "$TP_CHECK" == PASS* ]] || die "TP=2 check failed; no smoke test, no bank run"
  fi
fi

# ── C. Save the rendered chat prompt (heredoc copied from the run protocol; CPU only) ─
if python3 "$LOCALRUN/helper.py" present "$LABEL" > /dev/null; then
python - "$REPO" "$REV" "$LABEL" <<'EOF' > "logs/stepC_$LABEL.log" 2>&1
import json, sys
from transformers import AutoTokenizer
repo, rev, label = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=(label == "minicpm3-4b"))
item = json.loads(open("smoke_bank.jsonl").readline())
text = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)
open(f"manifest/{label}.rendered_prompt.txt", "w").write(text)
print(text)
with_system = next((json.loads(l) for l in open("item_bank.jsonl") if json.loads(l).get("system")), None)
if with_system:
    messages = [{"role": "system", "content": with_system["system"]}, {"role": "user", "content": with_system["prompt"]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    open(f"manifest/{label}.rendered_prompt_system.txt", "w").write(text)
    print(text)
EOF
  RC_C=$?
  tl "step C exit=$RC_C"
  [ "$RC_C" = 0 ] || tail -20 "logs/stepC_$LABEL.log"
  need "step C rendered prompt written" test "$RC_C" = 0
else
  need "step C needs the weights' tokenizer files" false
fi
SERVER_DATE_LOCAL=$(date +%F); SERVER_UTC_OFFSET=$(date +%z)

if [ "$DRY" = 1 ]; then
  show "${RUN[*]} --bank smoke_bank.jsonl --out smoke/${LABEL}_A.jsonl   (then _B)"
  show "ps -o pid=,comm=,args= -p \$(cat $PIDFILE); python $LOCALRUN/eval_smoke.py $LABEL EXIT_A EXIT_B logs/serve_command_$LABEL.txt logs/ps_args_pid_$LABEL.txt; stop unless S1-S6 all pass or the S3-only exception applies (ALLOW_S3_FAILURE=${ALLOW_S3_FAILURE:-0})"
  show "${RUN[*]} --bank $RUN_BANK --out out/$LABEL.jsonl   (again while exit code is 1, at most 4 invocations)"
  show "python $LOCALRUN/write_manifest.py logs/manifest_state_$LABEL.json ; kill \$(cat $PIDFILE) ; nvidia-smi"
  tl "DRY_RUN done: $N_DRY_FAIL check(s) would stop a real job. Files in $OUT"
  exit $(( N_DRY_FAIL > 0 ))
fi

# ── D. Smoke test, evaluated before the full run ─────────────────────────────
SMOKE_STARTED_UTC=$(date -u +%FT%TZ); T0=$(date +%s)
"${RUN[@]}" --bank smoke_bank.jsonl --out "smoke/${LABEL}_A.jsonl" 2>&1 | tee "logs/smoke_A_$LABEL.log"
EXIT_A=${PIPESTATUS[0]}
"${RUN[@]}" --bank smoke_bank.jsonl --out "smoke/${LABEL}_B.jsonl" 2>&1 | tee "logs/smoke_B_$LABEL.log"
EXIT_B=${PIPESTATUS[0]}
SMOKE_SECONDS=$(( $(date +%s) - T0 ))
tl "smoke exit A=$EXIT_A B=$EXIT_B seconds=$SMOKE_SECONDS"
ps -o args= -C vllm > "logs/ps_args_C_vllm_$LABEL.txt" 2>&1
ps -o pid=,comm=,args= -p "$(cat "$PIDFILE")" > "logs/ps_args_pid_$LABEL.txt" 2>&1
python "$LOCALRUN/eval_smoke.py" "$LABEL" "$EXIT_A" "$EXIT_B" "logs/serve_command_$LABEL.txt" "logs/ps_args_pid_$LABEL.txt" \
  2>&1 | tee "logs/smoke_evaluation_$LABEL.log"
RC_S=${PIPESTATUS[0]}
tl "smoke evaluation exit=$RC_S"
if [ "$RC_S" != 0 ]; then
  if [ "${ALLOW_S3_FAILURE:-0}" = 1 ] && [ "$RC_S" = 1 ]; then
    python - "$LABEL" <<'EOF' || die "smoke failure is outside the S3-only exception"
import json, os, sys
from pathlib import Path
label = sys.argv[1]
assert label in {"granite-4.0-350m", "llama-3.2-1b-instruct"}, label
observed = json.loads(Path(f"manifest/{label}.smoke_result.json").read_text())["smoke_result"]
expected = {f"S{i}": "fail" if i == 3 else "pass" for i in range(1, 7)}
assert observed == expected, observed
exception = {
    "label": label, "waived_conditions": ["S3"],
    "smoke_result": observed,
    "reason": ("Extension bank run: the same S3-only exception as the main run of 2026-10-03 is applied to the same two models. All other smoke checks remain mandatory."
               if os.environ.get("RUN_BANK") == "ext_bank.jsonl" else
               "All items are run under the original configuration while the S3 failure is retained, after 54 diagnostic responses. All other smoke checks remain mandatory."),
}
Path(f"manifest/{label}.smoke_exception.json").write_text(json.dumps(exception, ensure_ascii=False, indent=2) + "\n")
print("S3-only exception recorded; original smoke_result is unchanged.")
EOF
    deviation "Exception 2026-10-03: S3 remains fail; S1/S2/S4/S5/S6 pass. Proceed with all items of $RUN_BANK; see manifest/$LABEL.smoke_exception.json."
  else
    die "a smoke condition failed; the item bank is not run (section 5D)"
  fi
fi

# ── E. Full run (identical command again while exit code is 1, at most 4 invocations) ─
RUN_STARTED_UTC=$(date -u +%FT%TZ); T0=$(date +%s)
N_INV=0; RC_E=1
: > "logs/runner_summary_$LABEL.txt"
while [ "$RC_E" = 1 ] && [ "$N_INV" -lt 4 ]; do
  N_INV=$((N_INV + 1))
  "${RUN[@]}" --bank "$RUN_BANK" --out "out/$LABEL.jsonl" 2>&1 | tee "logs/run_${LABEL}_inv$N_INV.log"
  RC_E=${PIPESTATUS[0]}
  grep -E "^$LABEL: finished=" "logs/run_${LABEL}_inv$N_INV.log" >> "logs/runner_summary_$LABEL.txt"
  tl "full run invocation $N_INV exit=$RC_E"
done
RUN_FINISHED_UTC=$(date -u +%FT%TZ)
RUN_SECONDS=$(( $(date +%s) - T0 ))
tl "full run seconds=$RUN_SECONDS invocations=$N_INV last_exit=$RC_E"
nvidia-smi > logs/nvidia_smi_full_after_run.txt 2>&1
curl -s "http://127.0.0.1:$PORT/metrics" > "logs/vllm_metrics_$LABEL.txt"

# ── F. Manifest and shutdown ─────────────────────────────────────────────────
export LABEL REPO REV SERVE_CMD CVD SLURM_CVD TP GPU_UTIL PORT BUNDLE TP_CHECK SERVER_DATE_LOCAL SERVER_UTC_OFFSET \
  LOAD_SECONDS SMOKE_STARTED_UTC RUN_STARTED_UTC RUN_FINISHED_UTC RC_E N_INV OUT
python - <<'EOF'
import json, os, socket
from pathlib import Path
e = os.environ
label = e["LABEL"]
snap = Path(e["HF_HUB_CACHE"]) / ("models--" + e["REPO"].replace("/", "--")) / "snapshots"
state = {
    "label": label, "repo": e["REPO"], "rev": e["REV"],
    "revision_resolved": sorted(p.name for p in snap.iterdir()),
    "serve_command": e["SERVE_CMD"],
    "gpu": Path("logs/nvidia_smi_query_before.csv").read_text().strip().splitlines(),
    "gpus_used": e["CVD"], "slurm_cvd": e["SLURM_CVD"], "tp": e["TP"], "gpu_util": e["GPU_UTIL"],
    "server_date_local": e["SERVER_DATE_LOCAL"], "server_utc_offset": e["SERVER_UTC_OFFSET"],
    "load_seconds": int(e["LOAD_SECONDS"]),
    "smoke_started_utc": e["SMOKE_STARTED_UTC"], "run_started_utc": e["RUN_STARTED_UTC"], "run_finished_utc": e["RUN_FINISHED_UTC"],
    "runner_summary_lines": Path(f"logs/runner_summary_{label}.txt").read_text().strip().splitlines(),
    "runner_exit_code": int(e["RC_E"]), "n_invocations": int(e["N_INV"]),
    "node": socket.gethostname(), "slurm_job_id": e.get("SLURM_JOB_ID"),
    "slurm_array": f"{e.get('SLURM_ARRAY_JOB_ID', '')}_{e.get('SLURM_ARRAY_TASK_ID', '')}",
    "port": e["PORT"], "bundle": e["BUNDLE"], "tp_check": e["TP_CHECK"] or None,
    "environment": {k: e.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER", "CUDA_HOME", "OMP_NUM_THREADS", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "NCCL_P2P_DISABLE", "NCCL_DEBUG", "VLLM_SKIP_P2P_CHECK")},
    "deviations": Path(e["OUT"], "deviations.txt").read_text().strip().splitlines(),
}
Path(f"logs/manifest_state_{label}.json").write_text(json.dumps(state, indent=2) + "\n")
EOF
python "$LOCALRUN/write_manifest.py" "logs/manifest_state_$LABEL.json" || die "manifest not written"

PID=$(cat "$PIDFILE")
tl "shutdown: kill $PID"
kill "$PID"
for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -0 "$PID" 2>/dev/null && tl "server still alive 120 s after kill" || tl "server process gone"
sleep 5
nvidia-smi > logs/nvidia_smi_full_after_shutdown.txt 2>&1
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv | tee logs/nvidia_smi_apps_after_shutdown.csv
kill "$MON_PID" 2>/dev/null; MON_PID=
tl "job done: load=$LOAD_SECONDS s smoke=$SMOKE_SECONDS s full=$RUN_SECONDS s last_exit=$RC_E"
exit "$RC_E"
