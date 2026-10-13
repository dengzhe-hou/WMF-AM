# State-tracking probe: evaluation materials (anonymous review copy)

This package holds the item banks, the raw model responses, the scoring and analysis code, and the stored analysis outputs behind the paper "A Working-Memory-Inspired Programmable Probe of Cumulative State Tracking in Large Language Models". It contains no model weights and no credentials.

The paper refers to the blocks by name. This package uses the following codes in file names, item identifiers, CSV columns and this README.

| Code in this package | Block name in the paper |
| --- | --- |
| A | Core |
| B | Depth |
| C | Arithmetic |
| D | Cancellation |
| E | Assignment |
| F | Sets/counts |
| G | Wordings |
| H | Instructions |
| I | Written steps |
| K | Intermediate states |
| L | Multi-target |
| A512 | Large budget |

## The two archives

The materials come as two archives. Extract both archives into the same directory. Together they form one tree, and every command below is run from the root of that tree.

| Archive | Holds |
| --- | --- |
| `state_tracking_probe_software.zip` | `code/`, the Python scripts in `analysis/` and `analysis/paper/`, `analysis/paper/style/`, `external/bfcl_snapshot/build_criterion_csv.py` |
| `state_tracking_probe_data.zip` | `data/`, `analysis/results/`, `analysis/paper/tables/`, `analysis/paper/figures/`, `analysis/paper/model_provenance.json`, `external/bfcl_snapshot/score/`, `external/bfcl_snapshot/bfcl_criterion_19.csv`, `external/bfcl_snapshot/LICENSE`, `external/model_metadata/` |

`README.md`, `LICENSE`, `LICENSE-DATA` and `NOTICE` are in both archives with identical content.

## Contents

```
code/generator/          build_item_bank.py and three bank checks
code/generator/ext/      blocks K and L: build_ext_bank.py, the generator modules and manifests it reads, five bank checks
code/runner/             run_wmf_am_bank.py (the only client that talks to a model), model list, smoke items,
                         local serving scripts (as run on a SLURM cluster), API commands and identity check
data/item_bank/          item_bank.jsonl, the bank of 840 requests per model, and the build log;
                         three bank files for blocks K and L (180 requests per model)
data/responses/local/    10 files, one per locally served model, 840 records each
data/responses/api/      8 files, one per API model, 840 records each
data/responses/ext/      blocks K and L: local/ (9 files) and api/ (9 files), 180 records each;
                         first_runs_plural_m1/ (3 files); hosted_70b_core_check/ (4 files)
data/run_records/        smoke-test records, rendered prompts, serve commands, pip freeze, request timing, endpoint snapshots;
                         ext/ holds the records of the K/L runs
data/scored/             item-level scores for all 18 models and five summary tables; ext/ holds the K/L item-level scores
analysis/                scoring, descriptive analyses, association analyses, independent recomputation scripts
analysis/results/        stored outputs of those scripts; ext/ holds the K/L outputs
analysis/paper/          table and figure generators, with the generated tables and figures; style/ holds the
                         matplotlib style block that gen_depth.py reads
external/bfcl_snapshot/  BFCL score files used as the external criteria, with the script that extracts them
external/model_metadata/ pinned model revisions and parameter counts
```

The main evaluation has 15,120 response records (18 models, 840 requests each, none missing). Blocks K and L add 3,240 records (18 models, 180 requests each, none missing); they are described in step 8.

File and script names that contain `wmf_am` use the internal project name.

## Requirements

The checks below were run with Python 3.13.12, numpy 2.4.3, scipy 1.17.1 and matplotlib 3.10.8.

- The generators, the runner and the scorers use the Python standard library only.
- The analyses need numpy and scipy.
- The figures need matplotlib and the Arial font. With other versions or another font the figures may differ in their bytes.

No GPU, model weights, API key or network access is needed for any step below.

## Reproduction steps

Steps 1 to 5 and steps 7 and 8 are cheap: every command finished within a few seconds when the package was checked. Step 6 holds the long bootstrap analyses; their stored outputs are provided in `analysis/results/`, so the other steps do not depend on rerunning step 6.

### 1. Rebuild the item bank (cheap)

```bash
python code/generator/build_item_bank.py --include-cot --paraphrase-mode original \
    --minimal-final-number --max-tokens-int 16 --max-tokens-block none --out /tmp/item_bank.jsonl
cmp /tmp/item_bank.jsonl data/item_bank/item_bank.jsonl
python code/generator/bank_checks/check1_solver.py
python code/generator/bank_checks/check4_degenerate.py
python code/generator/bank_checks/check5_structure.py
```

Expected: `cmp` prints nothing, so the rebuilt bank is byte-identical to the packaged bank (840 items). The three checks end with `CHECK 1: PASS`, `CHECK 4: PASS` and `CHECK 5: PASS`. Check 1 recomputes every answer from the prompt text alone.

### 2. Score the raw responses (cheap)

```bash
python analysis/score_bank.py --out-dir data/scored data/responses/local/*.jsonl data/responses/api/*.jsonl
```

Expected: the scorer prints `bank items: 840   models: 18` and writes 15,120 scored rows. Every field equals the stored `data/scored/scored_items.csv`. Row order differs from the stored file, which lists the ten local models first; the five summary tables contain the same rows.

Columns: `correct_strict` is the rule used for every block in the paper. For blocks E and F the paper reports two parsers side by side: Parser 1 is `correct_submitted` and Parser 2 is `correct_strict`.

### 3. External criteria (cheap)

```bash
(cd external/bfcl_snapshot && python build_criterion_csv.py)
python analysis/prepare_bfcl_panel.py
```

Expected: `external/bfcl_snapshot/bfcl_criterion_19.csv`, `analysis/results/BFCL_INPUTS_20261004.csv` and `analysis/results/BFCL_INPUTS_20261004.json` are regenerated byte-identically. The 19th listed model was not run and is not part of the 18-model panel.

The point estimates of the main association table can be recomputed from the regenerated input:

```bash
python - <<'EOF'
import csv
from scipy.stats import kendalltau
rows = list(csv.DictReader(open("analysis/results/BFCL_INPUTS_20261004.csv")))
for column in ("overall", "multi_turn", "agentic"):
    tau = kendalltau([float(r["A_accuracy"]) for r in rows], [float(r[f"bfcl_{column}_1216"]) for r in rows]).statistic
    print(column, f"{tau:.6f}")
EOF
```

Expected: `overall 0.448867`, `multi_turn 0.417228`, `agentic 0.448867`. These equal the stored `tau_b` values for block A in `analysis/results/bfcl_comparison_20261004/associations.csv`.

### 4. Descriptive analyses (cheap, no resampling)

```bash
python analysis/report_protocol_pairs.py
python analysis/report_seed_prompt_stability.py
python analysis/report_depth_stability.py
python analysis/check_stability_independent.py
python analysis/report_relevant_updates.py
```

Expected: the protocol, seed, prompt and depth CSV files are regenerated byte-identically. The two stability JSON files differ from the stored ones only in the absolute paths they record. Each script also writes a Markdown report next to its outputs; these reports are not part of the package.

The fourth command recomputes the seed, prompt and depth tables independently, without resampling, and prints a JSON report whose first field is `"passed": true`. It rewrites `analysis/results/stability_20261004/INDEPENDENT_CHECK.json`; the rewritten file is not byte-identical to the stored one: it records the local paths, and it writes the `expected_summary` block at a different position. The values are the same.

The last command regenerates `analysis/results/relevant_updates_accuracy.csv` and `relevant_updates_by_k.csv` byte-identically. It prints core accuracy by the number of updates to the queried entity, `1 132/252 52.4%`, `2 116/360 32.2%`, `3 58/270 21.5%` and `4+ 36/198 18.2%` (Section 4.1), and the median number of such updates per K, 2 at K=3 and 45.5 at K=100 (Section 3.1).

### 5. Tables and figures (cheap)

```bash
cd analysis/paper
python gen_tables.py
python gen_high_subset_tables.py
python gen_documentation.py
python gen_ext_tables.py
python gen_depth.py
python gen_protocol.py
python gen_fig_framework.py
python gen_fig_overview.py
python gen_fig_main_results.py
python gen_fig_bfcl.py
python gen_fig_ext.py
cd ../..
```

Expected: all 27 files in `analysis/paper/tables/` and all seven figure PDFs in `analysis/paper/figures/` are regenerated byte-identically, and they are byte-identical to the manuscript sources. Of these 27 TeX files, ten provide printed tables, one provides the printed prompt example, and sixteen contain supplementary tables that are not printed. The paper also has one table written directly in its sections, giving eleven printed tables in total. Three of the figures are supplementary and are not printed (see the table below).

The main-text protocol association table, `protocol_associations.tex` (Table 7), reports five conditions for all three BFCL criteria with their intervals. The appendix table, `protocol_pearson.tex` (Table A2), retains the two paired Kendall differences and the Pearson sensitivity results in Appendix D. All protocol estimates remain in the saved CSV files under `analysis/results/bfcl_protocol_pearson_20261005/` and `analysis/results/budget/`.

The main-text domain table, `ef_domains.tex` (Table 4), selects the six domain rows from the same saved E/F pooled results used by the complete eight-row `ef_pooled.tex`. Both parsers are retained in parallel. The complete table remains in the supplementary materials; its block totals are also reported in Appendix E text.

### 6. Association analyses (long: 10,000 paired model resamples each, seed 42)

```bash
python analysis/run_bfcl_comparison.py       # Table 6, supplementary BFCL association plot, family_diagnostics.tex
python analysis/run_bfcl_protocol_pearson.py # Tables 7 and A2
python analysis/score_budget.py               # Tables 5 and 7, Figure 3c
python analysis/run_bfcl_controls.py         # supplementary controls.tex, full_controls.tex, full_control_differences.tex
python analysis/run_bfcl_task_controls.py    # supplementary controls.tex
python analysis/run_ef_comparison.py         # Tables 4 and A3, supplementary ef_pooled.tex and ef_domains_by_model.tex
python analysis/report_bfcl_high_subset.py   # supplementary high_subset_associations.tex, high_subset_scores.tex
python analysis/check_bfcl_controls_independent.py
python analysis/check_bfcl_task_controls_independent.py
python analysis/check_ef_comparison_independent.py
python analysis/check_bfcl_high_subset_independent.py
python analysis/run_bfcl_truncation_sensitivity.py  # Section 4.6, without the four models that truncated most core responses
python analysis/run_bfcl_subcategories.py           # Section 4.6, the 15 BFCL subcategory scores
```

These commands were not executed again when this package was assembled, and no run time is stated for them, except the last one. It was added later, takes a few seconds, and prints the three core-minus-arithmetic differences with their intervals: `overall 0.096 [-0.072, 0.254]`, `multi_turn 0.005 [-0.206, 0.188]` and `agentic 0.098 [-0.075, 0.258]`. It writes `truncation_sensitivity.csv` and `truncation_sensitivity.json` in `analysis/results/bfcl_comparison_20261004/`; this analysis was added after the main results and is not pre-specified. The command after it, `run_bfcl_subcategories.py`, was also added later and takes a few seconds. It repeats the Kendall and Pearson comparisons for the three main criteria and the 15 subcategory scores of the same BFCL snapshot, writes `subcategory_associations.csv` in the same folder, and prints the Kendall subcategory intervals that exclude zero: core minus arithmetic below zero for `live, mt_miss_func, non_live`, and none above zero. Their stored outputs are in `analysis/results/` and are what the tables in step 5 are typeset from. The intervals in the tables are the stored bootstrap outputs; the point estimates of Table 6 can be recomputed without resampling as shown in step 3.

### 7. Cost and duration of the API runs of the main evaluation (cheap)

```bash
python - <<'EOF'
import glob, json
cost = n = 0
for path in sorted(glob.glob("data/responses/api/*.jsonl")):
    for line in open(path, encoding="utf-8"):
        usage = json.loads(line).get("usage") or {}
        if "cost" in usage:
            cost += usage["cost"]; n += 1
print(n, round(cost, 10))
EOF
```

Expected: `6720 4.0246491625`, the sum of `usage.cost` in USD over the 6,720 full-run records. The 40 successful preflight records in `data/run_records/api/smoke/` add 0.0069887375 USD. The eight `*.timing.jsonl` files in `data/run_records/api/` span 30 minutes 45 seconds from the first start to the last end. These are sums of the usage fields returned with the responses, not invoice totals.

The two usage sums together are approximately USD 4.032. This covers 840 main-bank requests per API model plus the 40 successful preflight responses; it excludes the further blocks and is not a total for the 60-item core alone.

### 8. Blocks K and L (cheap)

Blocks K and L, the intermediate-state queries and the multi-target items, are described in Section 4.2 (Figure 4 and Table 3) and Appendix G of the paper (Table A4).

- Block K, 60 items: 20 generated trajectories; three items per trajectory ask for the value of one entity immediately after a numbered slot, the slot of its first, second or third update.
- Block L, 120 items: 20 generated trajectories; for M = 1, 2 or 4 queried entities there is a tracking item and a matched readout control.

All 18 models were run on these 180 items: 18 x 180 = 3,240 response records, none missing.

| Path | Holds |
| --- | --- |
| `data/item_bank/ext_bank.jsonl` | the final bank, 180 items |
| `data/item_bank/ext_bank_v1_m1_plural.jsonl` | the bank of the first runs: the 40 single-target items of block L end with the plural instruction of the other L items; nothing else differs |
| `data/item_bank/ext_bank_hosted70b.jsonl` | the final bank with a `system` field, used for the hosted Llama-3.3-70B run |
| `data/responses/ext/local/` | 9 files, one per locally served model, 180 records each |
| `data/responses/ext/api/` | 9 files, 180 records each: the eight API models, and `llama-3.3-70b-instruct.jsonl` from the hosted endpoint |
| `data/responses/ext/first_runs_plural_m1/` | 3 files: the first runs of Falcon3-1B, Falcon3-3B and Llama-3.2-3B, on the plural bank; kept as a record and not analysed |
| `data/responses/ext/hosted_70b_core_check/` | 4 files: the 60 core items (block A) on the hosted endpoint without a system message (top level) and with it (`with_system_header/`), and two 2-item probes of the system text |
| `data/run_records/ext/` | the run records of all of these runs, with a manifest for each local run; each local run folder also holds the copy of the bank that the run read, and `hosted_70b_core_check/` holds the banks of the hosted check |
| `data/scored/ext/scored_items.csv` | item-level scores, 3,240 rows |
| `analysis/results/ext/` | `analysis.json`, `accuracy_by_model.csv` and `REPORT.md`, written by `analysis/analyze_ext.py`; `comparison.json`, the hosted check |

How the models were run. Nine models were served locally as in the main evaluation. The eight API models were called as in the main evaluation, with `code/runner/api/api_run_ext.sh`. For blocks K and L, Llama-3.3-70B was run through a hosted endpoint (OpenRouter, provider Novita, endpoint listed as bf16, no fallback; `code/runner/api/api_run_70b_hosted.sh`), while its main evaluation was run locally. Its K/L requests carried a system message with the two date lines of the model's default chat template (`Cutting Knowledge Date: December 2023` and `Today Date: 26 Jul 2024`), which made the prompt token counts equal to those of the local deployment. The hosted check is the evidence for this choice: on the 60 core items the endpoint returned 31 correct responses with this system message (local run: 31; same verdict on 56 items; identical response text on 48), and 25 without it, with prompts 24 tokens shorter (`analysis/results/ext/comparison.json`).

The first runs. The 40 single-target items of block L first carried the plural instruction. Three models were run with it. The instruction was then changed to the singular form, before any other model was run, and the three models were run again on the final bank. Only the runs on the final bank are scored and analysed.

Rebuild and check the bank:

```bash
python code/generator/ext/build_ext_bank.py --out /tmp/ext_bank.jsonl
cmp /tmp/ext_bank.jsonl data/item_bank/ext_bank.jsonl
python code/generator/ext/bank_checks/check1_solver.py
python code/generator/ext/bank_checks/check3_design.py
python code/generator/ext/bank_checks/check4_shortcuts.py
python code/generator/ext/bank_checks/check5_format.py
python code/generator/ext/bank_checks/check6_scoring.py
python - <<'EOF'
import json
load = lambda name: [json.loads(line) for line in open("data/item_bank/" + name, encoding="utf-8")]
final, plural, hosted = load("ext_bank.jsonl"), load("ext_bank_v1_m1_plural.jsonl"), load("ext_bank_hosted70b.jsonl")
differing = [a["item_id"] for a, b in zip(final, plural) if a != b]
print(len(final), len(plural), len(hosted))
print(len(differing), all(a["target_count"] == 1 and {k for k in a if a[k] != b[k]} == {"prompt"} for a, b in zip(final, plural) if a != b))
print(all({k: v for k, v in h.items() if k != "system"} == f for f, h in zip(final, hosted)), sorted({h["system"] for h in hosted}))
EOF
```

Expected: the builder prints `wrote /tmp/ext_bank.jsonl (180 items)` and `cmp` prints nothing, so the rebuilt bank is byte-identical to the final bank. The five checks end with `CHECK 1: PASS`, `CHECK 3: PASS`, `hard conditions: hold`, `CHECK 5a: PASS` and `CHECK 6 / (d): PASS`. Check 1 recomputes all 180 reference answers from the prompt text alone. The last command prints `180 180 180`, then `40 True` (the plural bank differs from the final bank in the prompt of 40 single-target items and in nothing else), then `True` followed by the one system text of the hosted bank.

Score the raw responses:

```bash
python analysis/score_ext.py --bank data/item_bank/ext_bank.jsonl --out-dir /tmp/ext_17 \
    data/responses/ext/local/*.jsonl data/responses/ext/api/{claude,command,gpt}-*.jsonl
python analysis/score_ext.py --bank data/item_bank/ext_bank_hosted70b.jsonl --out-dir /tmp/ext_70b \
    data/responses/ext/api/llama-3.3-70b-instruct.jsonl
(cat /tmp/ext_17/scored_items.csv; tail -n +2 /tmp/ext_70b/scored_items.csv) > /tmp/ext_scored_items.csv
cmp /tmp/ext_scored_items.csv data/scored/ext/scored_items.csv
```

Expected: the scorer prints `scored 3060 rows for 17 models; missing 0` and `scored 180 rows for 1 models; missing 0`, and `cmp` prints nothing. The hosted run is scored against its own bank because the scorer requires the `system` field of every record to equal that of the bank.

Analyse, recompute independently, and typeset the two tables:

```bash
python analysis/analyze_ext.py --scored data/scored/ext/scored_items.csv --out-dir /tmp/ext_analysis
for f in analysis.json accuracy_by_model.csv REPORT.md; do cmp /tmp/ext_analysis/$f analysis/results/ext/$f; done
python analysis/check_ext_independent.py
(cd analysis/paper && python gen_ext_tables.py)
```

Expected: `cmp` prints nothing. `analysis.json` records the Python and numpy versions, so it is byte-identical only under the versions listed above. `check_ext_independent.py` shares no code with the scorer or the analysis script; it rescores every response, recomputes every count, mean and interval, and ends its comparison with `total compared: 26348 | total mismatches: 0`. `gen_ext_tables.py` rewrites `ext_summary.tex` and `ext_by_model.tex` byte-identically. The analysis resamples the 18 models 10,000 times (seed 42) and takes a fraction of a second.

## Where each table and figure comes from

| Manuscript item | Generator | Reads | Produced by |
| --- | --- | --- | --- |
| Table 1 (block overview) | written by hand | data/item_bank/item_bank.jsonl | counts and token limits checked against the bank |
| Table 2 (per-model main-bank scores) | gen_tables.py | RUN_REPORT.json | stored file, see limitations |
| Table 3 (intermediate-state and multi-target summary) | gen_ext_tables.py | ext/analysis.json, ext/accuracy_by_model.csv, BFCL_INPUTS_20261004.csv (model names and order) | analyze_ext.py |
| Table 4 (E/F domain scores) | gen_tables.py | ef_comparison_20261004/pooled_accuracy.csv | run_ef_comparison.py |
| Table 5 (response-protocol scores) | gen_tables.py | budget/by_model.csv | score_budget.py |
| Table 6 (BFCL associations) | gen_tables.py | bfcl_comparison_20261004/associations.csv, paired_differences.csv | run_bfcl_comparison.py |
| Table 7 (protocol BFCL associations) | gen_tables.py | bfcl_protocol_pearson_20261005/kendall_protocols.csv, budget/bfcl_associations.csv | run_bfcl_protocol_pearson.py, score_budget.py |
| Figure 1 | gen_fig_framework.py | protocol_pairs.csv, stability_20261004/depth_pooled.csv, BFCL_INPUTS_20261004.csv | report_protocol_pairs.py, report_depth_stability.py |
| Figure 2 (probe overview) | gen_fig_overview.py | none (schematic) | none |
| Figure 3 | gen_fig_main_results.py | BFCL_INPUTS_20261004.csv, stability_20261004/depth_pooled.csv, budget/summary.json | report_depth_stability.py, score_budget.py |
| Figure 4 | gen_fig_ext.py | ext/analysis.json | analyze_ext.py |
| Supplementary: figures/fig_bfcl.pdf (BFCL association plot) | gen_fig_bfcl.py | bfcl_comparison_20261004/associations.csv | run_bfcl_comparison.py |
| Section 3.1 prompt | gen_documentation.py | data/item_bank/item_bank.jsonl | build_item_bank.py |
| Table A1 | gen_documentation.py | EF_SCORING_DIFFERENCES_20261004.csv | stored file, see limitations |
| Supplementary: tables/criterion_inputs.tex (BFCL inputs) | gen_documentation.py | BFCL_INPUTS_20261004.csv | prepare_bfcl_panel.py |
| Supplementary: tables/local_revisions.tex (pinned local repositories and revisions) | gen_documentation.py | external/model_metadata/model_freeze_19.csv | static metadata |
| Supplementary: tables/high_subset_associations.tex, high_subset_scores.tex (high-BFCL subgroup) | gen_high_subset_tables.py | bfcl_high_subset_20261004/analysis.json | report_bfcl_high_subset.py |
| Supplementary: tables/controls.tex (rank-adjusted associations) | gen_tables.py | bfcl_controls_20261004/associations.csv, bfcl_task_controls_20261004/associations.csv | run_bfcl_controls.py, run_bfcl_task_controls.py |
| Table A2 | gen_tables.py | bfcl_protocol_pearson_20261005/kendall_protocols.csv, pearson.csv | run_bfcl_protocol_pearson.py |
| Table A3 | gen_tables.py | ef_comparison_20261004/associations.csv | run_ef_comparison.py |
| Supplementary: tables/ef_pooled.tex (complete pooled block and domain scores) | gen_tables.py | ef_comparison_20261004/pooled_accuracy.csv | run_ef_comparison.py |
| Supplementary: tables/ef_domains_by_model.tex (per-model domain scores) | gen_tables.py | ef_comparison_20261004/accuracy_by_domain.csv | run_ef_comparison.py |
| Supplementary: tables/protocol_by_model.tex (per-model protocol diagnostics, summarized in Appendix F) | gen_tables.py | protocol_conditions.csv, protocol_pair_summary.csv | report_protocol_pairs.py |
| Supplementary: tables/protocol.tex (pooled protocol outcomes) | gen_tables.py | protocol_conditions.csv, protocol_pair_summary.csv | report_protocol_pairs.py |
| Supplementary: tables/stability_pairs.tex (pairwise seed/wording correlations) | gen_tables.py | stability_20261004/seed_*.csv, prompt_*.csv | report_seed_prompt_stability.py |
| Supplementary: tables/stability_by_model.tex (per-model seed and wording scores) | gen_tables.py | stability_20261004/seed_*.csv, prompt_*.csv | report_seed_prompt_stability.py |
| Supplementary: tables/depth_by_model.tex (per-model depth scores) | gen_tables.py | stability_20261004/depth_by_model.csv | report_depth_stability.py |
| Table A4 (per-model intermediate-state and multi-target scores) | gen_ext_tables.py | ext/analysis.json, ext/accuracy_by_model.csv, BFCL_INPUTS_20261004.csv (model names and order) | analyze_ext.py |
| Supplementary: tables/full_controls.tex (all 54 A/C/D associations) | gen_tables.py | bfcl_controls_20261004/associations.csv | run_bfcl_controls.py |
| Supplementary: tables/full_control_differences.tex (all 36 paired differences) | gen_tables.py | bfcl_controls_20261004/paired_differences.csv | run_bfcl_controls.py |
| Supplementary: tables/family_diagnostics.tex (leave one provider out) | gen_tables.py | bfcl_comparison_20261004/leave_one_family_out.csv | run_bfcl_comparison.py |
| Supplementary: tables/protocol_pairs.tex (paired gains and losses per model) | gen_tables.py | protocol_pair_summary.csv | report_protocol_pairs.py |
| Section 3.1 (median updates to the queried entity by K) and Section 4.1 (core accuracy by those updates) | report_relevant_updates.py | data/scored/scored_items.csv, data/item_bank/item_bank.jsonl | score_bank.py |
| Section 4.6 (core minus arithmetic without the four truncation-dominated models) | run_bfcl_truncation_sensitivity.py | BFCL_INPUTS_20261004.csv | prepare_bfcl_panel.py |
| Section 4.6 (the 15 BFCL subcategory scores) | run_bfcl_subcategories.py | BFCL_INPUTS_20261004.csv, external/bfcl_snapshot/bfcl_criterion_19.csv | prepare_bfcl_panel.py, build_criterion_csv.py |
| Supplementary: figures/depth.pdf | gen_depth.py | stability_20261004/depth_pooled.csv | report_depth_stability.py |
| Supplementary: figures/protocol.pdf | gen_protocol.py | protocol_conditions.csv, BFCL_INPUTS_20261004.csv | report_protocol_pairs.py |

All paths in the "Reads" column are under `analysis/results/` unless they start with `data/` or `external/`.

## Models

Ten models were served locally with vLLM 0.30.0 in bfloat16 at temperature 0. The local repositories and frozen revision prefixes are in `analysis/paper/tables/local_revisions.tex`; complete commit identifiers are retained in `external/model_metadata/model_freeze_19.csv` and `analysis/paper/model_provenance.json`.

| Label | Repository |
| --- | --- |
| granite-4.0-350m | ibm-granite/granite-4.0-350m |
| granite-3.1-8b-instruct | ibm-granite/granite-3.1-8b-instruct |
| llama-3.2-1b-instruct | meta-llama/Llama-3.2-1B-Instruct |
| llama-3.2-3b-instruct | meta-llama/Llama-3.2-3B-Instruct |
| llama-3.3-70b-instruct | meta-llama/Llama-3.3-70B-Instruct |
| falcon3-1b-instruct | tiiuae/Falcon3-1B-Instruct |
| falcon3-3b-instruct | tiiuae/Falcon3-3B-Instruct |
| falcon3-7b-instruct | tiiuae/Falcon3-7B-Instruct |
| falcon3-10b-instruct | tiiuae/Falcon3-10B-Instruct |
| qwen3-4b-instruct-2507 | Qwen/Qwen3-4B-Instruct-2507 |

Eight models were called through OpenRouter, each pinned to its original provider with no fallback. Seven used temperature 0. GPT-5.2 was sent without a temperature parameter and with reasoning effort `none`.

The execution records retain rendered prompts and inference backends, together with API endpoint snapshots, requested aliases, original providers, response model fields, usage, and reasoning metadata.

| Label | Request model | Provider |
| --- | --- | --- |
| gpt-5.2-2025-12-11 | openai/gpt-5.2 | OpenAI |
| gpt-4.1-2025-04-14 | openai/gpt-4.1 | OpenAI |
| gpt-4.1-mini-2025-04-14 | openai/gpt-4.1-mini | OpenAI |
| claude-haiku-4-5-20251001 | anthropic/claude-haiku-4.5 | Anthropic |
| claude-sonnet-4-5-20250929 | anthropic/claude-sonnet-4.5 | Anthropic |
| claude-opus-4-5-20251101 | anthropic/claude-opus-4.5 | Anthropic |
| command-a-03-2025 | cohere/command-a | Cohere |
| command-r7b-12-2024 | cohere/command-r7b-12-2024 | Cohere |

The same settings were used for blocks K and L, with one difference: Llama-3.3-70B was not served locally for these blocks but called through OpenRouter as `meta-llama/llama-3.3-70b-instruct`, pinned to the provider Novita (endpoint listed as bf16), at temperature 0, with the system message described in step 8.

`data/responses/` holds text generated by these third-party models. The models themselves are not redistributed.

## Third-party model licenses and hosted services

The evaluated local models retain their original licenses. This package does not redistribute third-party model weights. Frozen repository identifiers and revisions are recorded in `external/model_metadata/model_freeze_19.csv`.

| Evaluated local models | Original license | Official source |
| --- | --- | --- |
| Granite 4.0 350M and Granite 3.1 8B Instruct | Apache License 2.0 | [Granite 4.0 model card](https://huggingface.co/ibm-granite/granite-4.0-350m/blob/bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2/README.md), [Granite 3.1 model card](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct/blob/4009206d5fc95d2e65a7b7633e159d6e97e25d35/README.md) |
| Qwen3 4B Instruct 2507 | Apache License 2.0 | [Frozen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/README.md) |
| Llama 3.2 1B and 3B Instruct | Llama 3.2 Community License Agreement | [License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE), [acceptable use policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md) |
| Llama 3.3 70B Instruct | Llama 3.3 Community License Agreement | [License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE), [acceptable use policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/USE_POLICY.md) |
| Falcon3 1B, 3B, 7B, and 10B Instruct | TII Falcon-LLM License 2.0 | [License terms](https://falconllm.tii.ae/falcon-terms-and-conditions.html), [frozen model card](https://huggingface.co/tiiuae/Falcon3-1B-Instruct/blob/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf/README.md) |

Hosted inference used OpenRouter. Relevant service terms are the [OpenRouter Terms of Service](https://openrouter.ai/terms), [OpenAI terms linked by OpenRouter](https://openai.com/policies/row-terms-of-use/), [OpenAI Services Agreement for API use](https://openai.com/policies/services-agreement/), [Anthropic Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms), [Cohere Terms of Use](https://cohere.com/terms-of-use), and [Novita Terms of Service](https://novita.ai/legal/terms-of-service). Provider links were checked against OpenRouter's [provider directory](https://openrouter.ai/providers). The Llama 70B hosted runs used Novita, as recorded in their execution metadata. Hosted-service terms are distinct from licenses for downloadable weights. In particular, the public Command A and Command R7B weight licenses do not establish permission for the hosted API runs.

The Cohere terms linked by OpenRouter include a benchmarking restriction. Whether a separate agreement or permission applies to the hosted Cohere runs has not been established. This inventory identifies the terms and does not certify such an exception.

## Local computational budget

The ten local runs of the main evaluation took 3,156 seconds of inference in total, from 39 to 783 seconds per model. Their jobs ran for 4,384 seconds of wall time including model loading and preflight checks, or 1.54 GPU-hours counting each MIG instance as one device and both GPUs of the 70B job. These totals exclude failed or stopped runs and separate test and diagnostic jobs.

The nine local runs of blocks K and L added 1,208 seconds of job wall time (0.34 GPU-hours), excluding the three first runs on the plural-instruction bank (392 seconds). The hosted Llama-3.3-70B run is not counted. The nine local runs of the large-budget protocol added 928 seconds of job wall time (0.26 GPU-hours); its hosted Llama-3.3-70B run is not counted.

The figures above use a device-count convention and describe the final local runs only. The full recorded October account, including pilots, diagnostics, failed or stopped attempts and superseded runs, is **1.868889 A100 4g.40gb MIG instance-hours** and **1.835000 RTX PRO 6000 whole-GPU-hours** across 42 started allocations. Three additional jobs were cancelled before allocation. These units are not added together. Small models used 40 GB MIG instances on NVIDIA A100 80 GB GPUs. The main-bank 70B model used two NVIDIA RTX PRO 6000 GPUs with 96 GB each and tensor parallelism of two.

Across all saved October API usage records, returned costs total **US$5.6422230425**, with **1,793,678 input tokens** and **183,360 output tokens**. This includes preflight and diagnostic calls and the subsequently retired control block. The total uses 9,889 usage records. Nine saved HTTP errors lack usage, as do early authentication failures and some internal retries. This is a sum of returned usage fields, not an invoice total. Provider GPU-hours and parallelism are unavailable. Offline CPU work and compute before these October runs were not timed in this account and are not treated as zero.

Anonymous allocation and usage ledgers, category summaries, exact totals and scope limits are in `data/compute/`. To reproduce the totals without model calls, run `python data/compute/recompute.py`.

## Provenance of the BFCL scores

`external/bfcl_snapshot/score/` holds unmodified score files from the public BFCL result archive, `https://github.com/HuanzhiMao/BFCL-Result`.

| Folder | Upstream path | Upstream commit that added it |
| --- | --- | --- |
| `score/2025-12-16/` | `2025-12-16/score/` | 6830ed13035c0cfee9aa7a9a0ffed70f10b3dd50 |
| `score/2025-11-03/` | `2025-11-03/score/` | 9e6a78acbb576c0789c915bd4d33dce64b358bc5 |
| `score/2025-08-26/` | `2025-08-26/score/` | 33e1b9906f807894c5eee8285b75bfc00f60ded0 |

The files were downloaded on 2 October 2026. On 4 October 2026 all 15 files were byte-identical to the upstream repository at commit cadd76a8dc62984e7bf15ecaf358f070bae745cc. The paper uses the [2025-12-16 snapshot](https://github.com/HuanzhiMao/BFCL-Result/tree/main/2025-12-16) (function-calling mode) as its criterion; the two earlier snapshots are included because `build_criterion_csv.py` reads them. `bfcl_criterion_19.csv` and `analysis/results/BFCL_INPUTS_20261004.csv` hold values extracted from these files. The latter also records the model mapping and parameter-count availability; `analysis/paper/model_provenance.json` retains the model provenance. No BFCL evaluation was rerun for this work. The upstream licence is Apache License 2.0; see `NOTICE` and `external/bfcl_snapshot/LICENSE`.

## Limitations of this package

- `analysis/results/RUN_REPORT.json` (Table 2), `EF_SCORING_DIFFERENCES_20261004.csv` (Table A1), `EF_SCORING_BY_MODEL_20261004.csv` and `analysis/results/ext/comparison.json` are stored files; the scripts that wrote them are not included. Every cell of Table 2, all seven rows of Table A1 and every value of `comparison.json` were recomputed from the response files and match.
- The commands in step 6 were not rerun for this package. Descriptive text strings in the scripts and in the stored outputs were edited for this package after the outputs were computed; the numeric values in the stored outputs are the originally computed ones.
- The scripts write Markdown reports (for example `REPORT.md`) next to their outputs. These reports are not included, except `analysis/results/ext/REPORT.md`.
- Three further bank checks (regeneration against earlier scripts, template comparison, runner stub) need earlier experiment scripts and are not included. The same holds for three further checks of the K/L bank (comparison with a draft bank that was never run, comparison with the first prompts of these item designs, output-token counts under the models' tokenizers).
- The generator modules under `code/generator/ext/` write prompts with a JSON answer instruction. `build_ext_bank.py` imports them, replaces the answer instruction, and checks the items against the bank fingerprints stored in the two manifests. The modules are used here only to build the bank. In the copies, module and manifest file names and experiment labels were renamed. Because the original label entered the seed derivation, the twenty derived seeds of each generator are listed as constants. The two manifests are cut down to the fields that the builder reads: the experiment label and the bank fingerprint.
- The K/L bank checks were written while all L items carried the plural instruction; one pattern was added to their parser so that it reads the singular question of the final bank.
- Stored JSON outputs record paths as `/path/to/repo/...`, which refers to the working tree in which they were produced, not to this package layout. The scripts write absolute paths into their outputs when rerun.
- The scripts under `code/runner/local/` are the launch scripts as run on one SLURM cluster. Paths, node names and job numbers are replaced by placeholders (`/path/to/repo`, `/path/to/venv`, `/path/to/conda_env`, `/path/to/hf_cache`, `node-a`, `node-b`, `Job N`), so they document the procedure and need adapting before use. Step labels such as 5A to 5F refer to the run protocol that `procedure.sh` implements; the smoke conditions S1 to S6 are defined in `eval_smoke.py`. The protocol text itself is not included.
- In the rendered-prompt records (`*.rendered_prompt*.txt` under `data/run_records/`), the current date that the chat templates of Llama 3.2 and Granite 3.1 insert is replaced by `DD Mon YYYY` or `Month DD, YYYY`.
- The scripts under `code/runner/api/` and `code/runner/local/report_runs.py` also keep the directory layout of the original working tree. Running the API commands sends paid requests and needs an OpenRouter key; this is not needed for any step above. The header comments of `api_run_ext.sh` and `api_run_70b_hosted.sh` were copied from `api_run_commands.sh` and still describe the main run.
- Item fields such as `legacy_scoring` and the column `correct_submitted` name scoring functions of earlier experiment scripts. Those scripts are not included; the rules they refer to are reimplemented in `analysis/score_bank.py`.

## Licences

- Code (`code/`, the scripts in `analysis/` and `external/bfcl_snapshot/build_criterion_csv.py`): MIT License, see `LICENSE`.
- Data, figures and documentation produced for this work: Creative Commons Attribution 4.0 International, see `LICENSE-DATA`.
- Model-generated text in `data/`: third-party model output, provided under CC BY 4.0 to the extent that the authors hold rights in it, see `NOTICE`.
- BFCL score files: Apache License 2.0, see `NOTICE` and `external/bfcl_snapshot/LICENSE`.
