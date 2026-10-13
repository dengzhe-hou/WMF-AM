#!/usr/bin/env python3
"""Core, arithmetic and cancellation associations with every BFCL V4 score column (2026-10-11).

Reproduce from any directory: python analysis/run_bfcl_subcategories.py
The three main criteria and the 15 subcategory scores of the 2025-12-16 snapshot, all 18 models.
Kendall tau-b with the kernels of run_bfcl_comparison.py and Pearson r with those of
run_bfcl_protocol_pearson.py, on the same 10,000 paired model resamples (seed 42).
"""

import csv

import numpy as np
from scipy.stats import pearsonr

from analyze import clean, kendall, model_weights, sign_matrix, taub
from run_bfcl_comparison import INPUT, N_BOOT, OUT, ROOT, SEED, summary, write_csv
from run_bfcl_protocol_pearson import weighted_pearson

CRITERION = ROOT / "external/bfcl_snapshot/bfcl_criterion_19.csv"
MAIN = ("overall", "multi_turn", "agentic")
SUB = ("non_live", "live", "irrelevance", "relevance", "mt_base", "mt_miss_func", "mt_miss_param",
       "mt_long_context", "web_search", "ws_base", "ws_no_snippet", "memory", "mem_kv", "mem_vector", "mem_rec_sum")


def main():
    with INPUT.open(encoding="utf-8", newline="") as handle:
        panel = list(csv.DictReader(handle))
    with CRITERION.open(encoding="utf-8", newline="") as handle:
        crit = {r["model"]: r for r in csv.DictReader(handle)}
    assert len(panel) == 18 and all(r["model"] in crit for r in panel)
    x = {b: np.array([int(r[f"{b}_correct"]) for r in panel]) for b in "ACD"}
    weights = model_weights(len(panel), N_BOOT, SEED)
    rows = []
    for col in MAIN + SUB:
        y = np.array([float(crit[r["model"]][f"{col}_1216"]) for r in panel])
        if col in MAIN:
            assert np.array_equal(y, [float(r[f"bfcl_{col}_1216"]) for r in panel])
        kpoint = {b: clean(kendall(x[b], y)) for b in "ACD"}
        kboot = {b: taub(sign_matrix(x[b])[None], sign_matrix(y)[None], weights) for b in "ACD"}
        ppoint = {b: float(pearsonr(x[b], y).statistic) for b in "ACD"}
        pboot = {b: weighted_pearson(x[b].astype(float), y, weights) for b in "ACD"}
        for stat, point, boot in (("kendall_tau_b", kpoint, kboot), ("pearson_r", ppoint, pboot)):
            for term in ("A", "C", "D", "A-C", "A-D"):
                if "-" in term:
                    a, c = term.split("-")
                    estimate, draws = point[a] - point[c], boot[a] - boot[c]
                else:
                    estimate, draws = point[term], boot[term]
                rows.append({"statistic": stat, "criterion": col, "kind": "main" if col in MAIN else "subcategory",
                             "term": term, "n_models": len(panel), "estimate": estimate, **summary(draws)})
    write_csv(OUT / "subcategory_associations.csv", rows)
    sub = [r for r in rows if r["kind"] == "subcategory" and r["statistic"] == "kendall_tau_b" and "-" in r["term"]]
    below = sorted(r["criterion"] for r in sub if r["term"] == "A-C" and r["ci95_high"] < 0)
    above = sorted(f"{r['criterion']} {r['term']}" for r in sub if r["ci95_low"] > 0)
    print("Kendall, 15 subcategories: core minus arithmetic intervals below zero:", ", ".join(below))
    print("Kendall, 15 subcategories: difference intervals above zero:", ", ".join(above) or "none")


if __name__ == "__main__":
    main()
