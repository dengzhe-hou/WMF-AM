# Compute accounting for the October 2026 experiments

All numbers below are calculated from saved scheduler records or response usage fields. No inference was repeated for this accounting. This account covers the present study's October runs, including preparation, diagnostics, failures and replaced runs, rather than only the results retained in the paper.

## Local inference

The 42 started allocations ran from October 3 to October 5. Three further jobs were cancelled while still pending and consumed no allocated GPU time. Each allocation is counted once, using its elapsed seconds multiplied by its actual allocated device count. Time includes model loading, preflight, execution and shutdown. It measures resource allocation rather than active GPU kernel time.

| Run group | A100 4g.40gb MIG instance-hours | RTX PRO 6000 whole-GPU-hours |
| --- | ---: | ---: |
| Final main evaluation | 0.895000 | 0.645556 |
| Final intermediate-state and multi-target queries | 0.335556 | 0 |
| Final large-budget protocol | 0.257778 | 0 |
| Full-pipeline pilot | 0.150278 | 0 |
| Failed or stopped main attempts | 0.050278 | 1.060556 |
| Echo diagnostics | 0.071111 | 0 |
| Communication diagnostics | 0 | 0.128889 |
| Superseded first query-bank runs | 0.108889 | 0 |
| **All recorded study allocations** | **1.868889** | **1.835000** |

The exact totals are 6,728 MIG instance-seconds and 6,606 whole-GPU-seconds. The two columns are not equivalent resource units and should not be added as undifferentiated GPU-hours. Each small-model allocation used one 40 GB MIG instance on an NVIDIA A100 80GB PCIe. Each of the four started whole-GPU allocations reserved two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition devices. The successful 70B run records 97,887 MiB visible memory per device, tensor parallelism of two, and checks that both devices held model processes and weights.

The earlier 1.54 figure for the successful main jobs is reproduced by adding 0.895000 MIG instance-hours to 0.645556 whole-GPU-hours. The 0.34 and 0.26 figures refer only to the successful local query-extension and large-budget jobs. Those figures were partial and used a mixed device-count convention. The separate columns above provide the complete recorded October allocation account. The underlying main jobs included a subsequently retired control block. Its compute remains included here even though its answers are not in the final evaluation.

`local_compute.csv` lists all 45 records using anonymous row identifiers. `compute_summary.json` gives exact seconds and unrounded totals. The scheduler's actual allocated GPU counts determine the multiplicity. Source manifests and GPU reports establish the hardware type. Pending jobs have zero allocated devices, regardless of what their scripts requested.

## Hosted inference

The original API response folders contain 9,898 saved records, of which 9,889 have returned usage fields. Across these fields the sum is **US$5.6422230425**, with **1,793,678 input tokens** and **183,360 output tokens**. This includes all recorded main and extra-query runs, the large-budget comparison, preflight and hosted-deployment checks, and the retired control block. `api_compute.csv` separates these categories. `api_usage_records.csv` gives the anonymous per-record usage ledger, including empty usage cells for recorded HTTP errors. Run `python recompute.py` to independently sum both the local and API ledgers.

The 9 records lacking usage are saved HTTP-error outcomes. Early authentication failures and some internal retry attempts also lack separate usage records. Costs are response-derived sums, not verified invoice totals. Original run folders are counted once, with identical saved JSON records deduplicated directly. Provider response IDs were not retained. Genuine repeated probes with distinct timestamps remain in the count. Provider hardware, parallelism and GPU-hours are not visible to the client, so API costs and token counts are not converted to GPU-hours.

## Scope limits

This account covers the October experiments rather than reconstructing compute for the earlier exploratory work. Historical compute for the earlier exploratory work is not estimated here. Model pretraining and the public BFCL evaluations were not rerun. Offline CPU work for item generation, scoring, analysis, environment setup, packaging and manuscript preparation was not timed. No aggregate CPU-hours are therefore claimed. These unavailable quantities are not silently treated as zero.
