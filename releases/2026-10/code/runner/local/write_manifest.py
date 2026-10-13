#!/usr/bin/env python3
"""Write manifest/<label>.json with the fields of the run protocol (step 5F), plus the local-run fields
(node, SLURM job, GPU type, MIG profile, port, environment). Run from the model's output directory,
inside the vLLM venv. Usage: write_manifest.py STATE_JSON  (values recorded by procedure.sh)

Based on localtest_granite31/write_manifest.py; tensor_parallel_size and vllm_build are no longer fixed.
"""
import json
import os
import platform
import re
import sys
from pathlib import Path

import torch
import transformers
import vllm

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
label = state["label"]

final_ids = set()
for line in Path(f"out/{label}.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        record = json.loads(line)
        if record["error"] is None:
            final_ids.add(record["item_id"])
run_bank = os.environ.get("RUN_BANK", "item_bank.jsonl")   # ext_bank.jsonl in an extension run
n_bank = sum(1 for line in Path(run_bank).read_text(encoding="utf-8").splitlines() if line.strip())
smoke = json.loads(Path(f"manifest/{label}.smoke_result.json").read_text(encoding="utf-8"))

# GPU type and MIG profile from `nvidia-smi -L` as seen inside the job.
gpu_list = Path("logs/nvidia_smi_L.txt").read_text(encoding="utf-8").splitlines()
gpu_names = [m.group(1) for line in gpu_list if (m := re.match(r"GPU \d+: (.+?) \(UUID", line))]
mig_profiles = [m.group(1) for line in gpu_list if (m := re.search(r"MIG (\S+)\s+Device", line))]
# Attention backend vLLM chose (can differ between A100 and RTX PRO 6000).
vllm_log = Path(f"logs/vllm_{label}.log").read_text(encoding="utf-8", errors="replace").splitlines()
attention_lines = sorted({line.split("] ", 1)[-1] for line in vllm_log if "attention backend" in line})

manifest = {
    "label": label,
    "hf_repo": state["repo"],
    "revision_requested": state["rev"],
    "revision_resolved": state["revision_resolved"],
    "serve_command": state["serve_command"],
    "vllm_non_default_args": smoke["detail"]["S6"]["non_default_args_line"],
    "vllm_version": vllm.__version__,
    "vllm_build": "github-cu129" if "+cu129" in vllm.__version__ else "pypi-cuda13.0",
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "transformers_version": transformers.__version__,
    "python_version": platform.python_version(),
    "gpu": state["gpu"],
    "gpu_type": gpu_names,
    "mig_profile": mig_profiles or None,
    "nvidia_smi_L": gpu_list,
    "gpus_used": state["gpus_used"],
    "cuda_visible_devices_from_slurm": state["slurm_cvd"],
    "tensor_parallel_size": int(state["tp"]),
    "gpu_memory_utilization": float(state["gpu_util"]),
    "dtype": "bfloat16",
    "quantization": "none",
    "sampling": "temperature 0, max_tokens taken from each bank item and 16 for the smoke items, "
                "generation-config vllm, seed 0, concurrency 1",
    "chat_template_source": "the repo's own tokenizer chat template, no override",
    "rendered_prompt_files": [f"manifest/{label}.rendered_prompt.txt", f"manifest/{label}.rendered_prompt_system.txt"],
    "server_date_local": state["server_date_local"],
    "server_utc_offset": state["server_utc_offset"],
    "load_seconds": state["load_seconds"],
    "smoke_started_utc": state["smoke_started_utc"],
    "run_started_utc": state["run_started_utc"],
    "run_finished_utc": state["run_finished_utc"],
    "smoke_result": smoke["smoke_result"],
    "runner_summary_lines": state["runner_summary_lines"],
    "n_items_final": len(final_ids),
    "n_items_missing": n_bank - len(final_ids),
    "runner_exit_code": state["runner_exit_code"],
    "n_invocations": state["n_invocations"],
    "local_run": {
        "node": state["node"],
        "slurm_job_id": state["slurm_job_id"],
        "slurm_array": state["slurm_array"],
        "port": int(state["port"]),
        "bundle": state["bundle"],
        "environment": state["environment"],
        "tp_check": state.get("tp_check"),
        "attention_backend_lines": attention_lines,
    },
    "deviations": state["deviations"],
}
Path(f"manifest/{label}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote manifest/{label}.json n_items_final={manifest['n_items_final']} n_items_missing={manifest['n_items_missing']}")
