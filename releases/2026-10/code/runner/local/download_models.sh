#!/bin/bash
# Download the open-weight models for the local WMF-AM runs into the shared HF cache.
#
#   download_models.sh            plan only (default): size table from the HF API (GET) and the quota.
#                                 Nothing is downloaded.
#   download_models.sh now        download the NOW group (non-gated)
#   download_models.sh later [LABEL...]
#                                 download the LATER group (meta-llama, gated). First a HEAD request per
#                                 repo; if any requested repo is still not accessible, stop before
#                                 downloading anything.
#
# Every repo is fetched at its pinned revision from models_open.tsv of the frozen bundle, with
# --exclude "original/*" and only the files vLLM and the chat-template step read
# (*.safetensors, *.json, *.jinja, tokenizer.model, merges.txt). A repo whose pinned snapshot already
# holds config, tokenizer config and every weight shard is skipped (granite-3.1-8b-instruct is).
# minicpm3-4b is not downloaded: it is excluded from the analysis (decided 2026-10-03).
# Runs on the login node; it is network and disk bound, not CPU bound.
set -uo pipefail

HERE=/path/to/repo/localrun
export HF_HUB_CACHE=/path/to/hf_cache   # symlink to ~/.cache/huggingface/hub
HF=/path/to/venv/bin/hf              # huggingface_hub 1.33.0, same venv as the runs
PY=python3

NOW=(granite-4.0-350m falcon3-1b-instruct falcon3-3b-instruct qwen3-4b-instruct-2507 falcon3-7b-instruct falcon3-10b-instruct granite-3.1-8b-instruct)
LATER=(llama-3.2-1b-instruct llama-3.2-3b-instruct llama-3.3-70b-instruct)

die(){ echo "STOP: $*" >&2; exit 1; }

quota(){
  echo "quota (tank/home):"
  zfs get -H -o property,value userused@USER,userquota@USER tank/home | sed 's/^/  /'
}

download(){   # $1 = label
  local label=$1 repo rev
  if $PY "$HERE/helper.py" present "$label"; then return 0; fi
  IFS=$'\t' read -r repo rev _ < <($PY "$HERE/helper.py" lookup "$label")
  echo "[$(date -u +%FT%TZ)] download $label: $repo @ $rev"
  "$HF" download "$repo" --revision "$rev" \
    --include "*.safetensors" --include "*.json" --include "*.jinja" --include "tokenizer.model" --include "merges.txt" \
    --exclude "original/*" || die "hf download failed for $label"
  $PY "$HERE/helper.py" complete "$label" || die "$label incomplete after download"
}

case "${1:-plan}" in
  plan)
    echo "== NOW group (non-gated) =="
    $PY "$HERE/helper.py" table "${NOW[@]}"
    echo
    echo "== LATER group (meta-llama, gated) =="
    $PY "$HERE/helper.py" table "${LATER[@]}"
    echo
    echo "== access to the gated repos (HEAD, status code only) =="
    $PY "$HERE/helper.py" access "${LATER[@]}"
    echo
    quota
    echo "Nothing was downloaded. To download, run: $0 now   (later: $0 later)"
    ;;
  now)
    quota
    for label in "${NOW[@]}"; do download "$label"; done
    quota
    ;;
  later)
    shift
    if [ $# -gt 0 ]; then want=("$@"); else want=("${LATER[@]}"); fi
    $PY "$HERE/helper.py" access "${want[@]}" \
      || die "access to at least one requested meta-llama repo is still pending (HTTP 401/403 above). Nothing downloaded. Rerun when Hugging Face shows the request as accepted, or name only the accepted labels: $0 later llama-3.2-1b-instruct llama-3.2-3b-instruct"
    quota
    for label in "${want[@]}"; do download "$label"; done
    quota
    ;;
  *) die "usage: $0 [plan|now|later [LABEL...]]" ;;
esac
