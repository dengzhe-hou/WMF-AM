#!/usr/bin/env python3
"""Evaluate the smoke conditions S1 to S6 of the run protocol (step 5D), as written.

Same pass conditions as localtest_granite31/eval_smoke.py (Job N, all six passed there).
Missing records after a runner abort explicitly fail the applicable conditions.
Run from the model's output directory. In the local runs PS_ARGS_FILE is `ps` of the saved server PID,
because two servers can run on node-a at once and `ps -C vllm` would list both.
Usage: eval_smoke.py LABEL EXIT_A EXIT_B SERVE_COMMAND_FILE PS_ARGS_FILE
Writes manifest/<label>.smoke_result.json and exits 0 only if all six pass.
"""
import json
import re
import sys
from pathlib import Path

label, exit_a, exit_b, serve_cmd_file, ps_args_file = sys.argv[1:6]
exit_a, exit_b = int(exit_a), int(exit_b)


def records(path):
    path = Path(path)
    # The runner can abort before it writes its first response (for example, HTTP 400).
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


run_a = records(f"smoke/{label}_A.jsonl")
run_b = records(f"smoke/{label}_B.jsonl")
smoke_items = records("smoke_bank.jsonl")
result, detail = {}, {}

# S1. Both runs exit with code 0.
result["S1"] = exit_a == 0 and exit_b == 0
detail["S1"] = {"exit_A": exit_a, "exit_B": exit_b}

# S2. Each file has 5 records and every record has "error": null.
result["S2"] = all(len(run) == 5 and all(r["error"] is None for r in run) for run in (run_a, run_b))
detail["S2"] = {"n_A": len(run_a), "n_B": len(run_b),
                "errors_A": [r["error"] for r in run_a], "errors_B": [r["error"] for r in run_b],
                "missing_files": [f"smoke/{label}_{run}.jsonl" for run in ("A", "B")
                                  if not Path(f"smoke/{label}_{run}.jsonl").exists()]}

# S3. The two echo items have "correct": 1 in run A.
by_id_a = {r["item_id"]: r for r in run_a}
by_id_b = {r["item_id"]: r for r in run_b}
echo = {i: by_id_a[i]["correct"] if i in by_id_a else None for i in ("smoke_echo_7", "smoke_echo_42")}
result["S3"] = all(v == 1 for v in echo.values())
detail["S3"] = echo

# S4. raw_response in run A equals raw_response in run B, for each of the 5 items.
same = {item["item_id"]: item["item_id"] in by_id_a and item["item_id"] in by_id_b
        and by_id_a[item["item_id"]]["raw_response"] == by_id_b[item["item_id"]]["raw_response"]
        for item in smoke_items}
result["S4"] = all(same.values())
detail["S4"] = same

# S5. The rendered prompt contains the smoke prompt text unchanged and no <think>.
rendered = Path(f"manifest/{label}.rendered_prompt.txt").read_text(encoding="utf-8")
smoke_prompt = smoke_items[0]["prompt"]
result["S5"] = smoke_prompt in rendered and "<think>" not in rendered
detail["S5"] = {"contains_smoke_prompt": smoke_prompt in rendered, "contains_think": "<think>" in rendered}

# S6. Started with --dtype bfloat16, --generation-config vllm, no quantization flag;
#     the "non-default args:" line lists dtype bfloat16 and generation_config vllm,
#     and lists neither quantization nor reasoning_parser.
serve_cmd = Path(serve_cmd_file).read_text(encoding="utf-8")
ps_args = Path(ps_args_file).read_text(encoding="utf-8")
log_lines = Path(f"logs/vllm_{label}.log").read_text(encoding="utf-8", errors="replace").splitlines()
nd_lines = [line for line in log_lines if "non-default args:" in line]
nd = nd_lines[0] if nd_lines else ""
cmd_ok = "--dtype bfloat16" in serve_cmd and "--generation-config vllm" in serve_cmd and "--quantization" not in serve_cmd
ps_ok = "--dtype bfloat16" in ps_args and "--generation-config vllm" in ps_args and "--quantization" not in ps_args
nd_ok = (bool(re.search(r"'dtype': 'bfloat16'", nd)) and bool(re.search(r"'generation_config': 'vllm'", nd))
         and "quantization" not in nd and "reasoning_parser" not in nd)
result["S6"] = cmd_ok and nd_ok
detail["S6"] = {"recorded_command_ok": cmd_ok, "ps_args_ok": ps_ok, "ps_args_empty": not ps_args.strip(),
                "n_non_default_lines": len(nd_lines), "non_default_line_ok": nd_ok, "non_default_args_line": nd}

out = {"smoke_result": {k: "pass" if v else "fail" for k, v in result.items()}, "detail": detail}
Path(f"manifest/{label}.smoke_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out["smoke_result"]))
sys.exit(0 if all(result.values()) else 1)
