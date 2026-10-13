# Budget block (2026-10-05)

Author decision 2026-10-05: A's 60 core items with A's prompt and parser, output limit 512 tokens instead of 16.
Purpose: separate the output budget from the scoring rule and the written-steps instruction in the A/H/I comparison.

- build_budget_bank.py -> budget_bank.jsonl (60 items, ids A512-*, prompts and gold identical to A)
- budget_bank_hosted70b.jsonl: same items plus the two date lines of the Llama 3.x chat template as a system message,
  as for blocks K and L (Llama-3.3-70B through OpenRouter Novita bf16)
- api_run_budget.sh: copy of ext/api_run_ext.sh with BANK and OUT changed
- local models: localrun/run_one_model.sbatch with EXT_BANK=budget_bank.jsonl, RESULTS_ROOT=results/budget_20261005/open
- 70B: ext/api_run_70b_hosted.sh with BANK_OVERRIDE / OUT_OVERRIDE
Outputs: results/budget_20261005/{open,api,hosted70b}
