#!/bin/bash
# Submit the small open models to node-a as one job array, at most 2 running at once (--array ...%2).
# minicpm3-4b is excluded (decided 2026-10-03, analysis uses 18 models); llama-3.3-70b runs
# separately on node-b (run_70b_node-b.sbatch).
#
#   submit_small_models.sh             check, then sbatch
#   DRY_RUN=1 submit_small_models.sh   check and print the sbatch command; nothing is submitted
#
# A label is left out, with a message, when
#   - it is a meta-llama repo and the HEAD request says access is still pending (HTTP 401/403), or
#   - results/open/<label>/ already exists (an earlier job; nothing is overwritten).
# The script stops before submitting anything when the bank in the bundle is not the G=16 option-A bank,
# or when a selected model is not downloaded (run download_models.sh first).
set -uo pipefail

HERE=/path/to/repo/localrun
RESULTS=/path/to/repo/results/open
BUNDLE=/path/to/repo/frozen/wmf_am_server_bundle_20261003.tar.gz
export HF_HUB_CACHE=/path/to/hf_cache
SMALL=(granite-4.0-350m llama-3.2-1b-instruct falcon3-1b-instruct llama-3.2-3b-instruct falcon3-3b-instruct
       qwen3-4b-instruct-2507 falcon3-7b-instruct granite-3.1-8b-instruct falcon3-10b-instruct)

DRY=${DRY_RUN:-0}
N_STOP=0
die(){ echo "STOP: $*" >&2; exit 1; }
stop(){   # a real run stops; a dry run reports and goes on
  if [ "$DRY" = 1 ]; then echo "DRY_RUN, a real run would STOP here: $*"; N_STOP=$((N_STOP + 1)); else die "$*"; fi
}

# Bank inside the bundle the jobs will unpack.
TMP=$(mktemp -d "${TMPDIR:-/tmp}/wmfam_submit.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
tar xzf "$BUNDLE" -C "$TMP" bundle/item_bank.jsonl bundle/models_open.tsv || die "cannot read $BUNDLE"
python3 "$HERE/helper.py" bank "$TMP/bundle/item_bank.jsonl" || stop "the bundle's bank is not the one to run; re-pack the bundle first"
export MODELS_TSV=$TMP/bundle/models_open.tsv

SEL=()
for label in "${SMALL[@]}"; do
  if [ -d "$RESULTS/$label" ] && [ -n "$(ls -A "$RESULTS/$label")" ]; then
    echo "skip $label: $RESULTS/$label already exists (earlier job; inspect it)"
    continue
  fi
  if [[ "$label" == llama-* ]] && ! python3 "$HERE/helper.py" access "$label"; then
    echo "skip $label: Hugging Face access still pending; submit it later with this script"
    continue
  fi
  python3 "$HERE/helper.py" present "$label" || { stop "$label is not downloaded; run download_models.sh first"; continue; }
  SEL+=("$label")
done
[ "${#SEL[@]}" -gt 0 ] || stop "nothing to submit"

CMD=(sbatch --array=1-${#SEL[@]}%2 "$HERE/run_one_model.sbatch" "${SEL[@]}")
echo "array task -> label:"
for i in "${!SEL[@]}"; do echo "  $((i + 1)) ${SEL[$i]}"; done
if [ "$DRY" = 1 ]; then
  echo "DRY_RUN, not submitted ($N_STOP check(s) would stop a real run): ${CMD[*]}"
  exit $(( N_STOP > 0 ))
fi
mkdir -p "$RESULTS/_slurm"
"${CMD[@]}"
