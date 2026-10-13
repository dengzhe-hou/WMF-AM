#!/usr/bin/env python3
"""First BFCL comparison (specified 2026-10-04): A/C/D, three criteria, all 18 models.

Reproduce from any directory: python analysis/run_bfcl_comparison.py
Uses the existing tau-b and paired model-bootstrap kernels, not analyze.main().
No permutation p, control adjustment, model selection, or new model inference.
"""

import csv
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from analyze import clean, kendall, model_weights, percentile_ci, sign_matrix, taub

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
INPUT = PANEL / "BFCL_INPUTS_20261004.csv"
OUT = PANEL / "bfcl_comparison_20261004"
CRITERIA = ("overall", "multi_turn", "agentic")
BLOCKS = ("A", "C", "D")
N_BOOT = 10_000
SEED = 42


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "undefined" if value is None else f"{value:.3f}"


def interval(row):
    return f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"


def summary(values):
    ci, valid = percentile_ci(values)
    return {"ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
            "boot_valid": valid, "boot_undefined": N_BOOT - valid}


def main():
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [r["model"] for r in rows]
    assert len(names) == len(set(names)) == 18
    families = sorted({r["family"] for r in rows})
    assert len(families) == 7
    for block, n in (("A", 60), ("C", 90), ("D", 100)):
        assert all(int(r[f"{block}_n"]) == n for r in rows)
    # Each block has the same denominator across models. Numerators preserve its
    # accuracy ordering and exact ties without rounding the published fractions.
    x = {block: np.array([int(r[f"{block}_correct"]) for r in rows]) for block in BLOCKS}
    y = {col: np.array([float(r[f"bfcl_{col}_1216"]) for r in rows]) for col in CRITERIA}
    assert all(np.isfinite(v).all() for v in [*x.values(), *y.values()])
    weights = model_weights(len(rows), N_BOOT, SEED)
    associations, differences, leave_out = [], [], []
    for col in CRITERIA:
        boot, points = {}, {}
        for block in BLOCKS:
            points[block] = clean(kendall(x[block], y[col]))
            boot[block] = taub(sign_matrix(x[block])[None], sign_matrix(y[col])[None], weights)
            associations.append({"criterion": col, "block": block, "n_models": len(rows),
                                 "tau_b": points[block], **summary(boot[block])})
        for control in ("C", "D"):
            delta = (points["A"] - points[control]
                     if points["A"] is not None and points[control] is not None else None)
            differences.append({"criterion": col, "comparison": f"A-{control}",
                                "n_models": len(rows), "delta_tau_b": delta,
                                **summary(boot["A"] - boot[control])})
        for family in families:
            keep = np.array([r["family"] != family for r in rows])
            point = {block: clean(kendall(x[block][keep], y[col][keep])) for block in BLOCKS}
            leave_out.append({"criterion": col, "family_left_out": family,
                              "n_models": int(keep.sum()),
                              **{f"tau_{block}": point[block] for block in BLOCKS},
                              **{f"delta_A_{control}": point["A"] - point[control]
                                 if point["A"] is not None and point[control] is not None else None
                                 for control in ("C", "D")}})

    metadata = {
        "analysis_status": "exploratory; selected after model scores were visible",
        "input": str(INPUT), "scored_source": str(ROOT / "data/scored/scored_items.csv"),
        "criterion_source": str(ROOT / "external/bfcl_snapshot/bfcl_criterion_19.csv"),
        "criterion_snapshot": "2025-12-16", "criteria_parallel_no_primary": list(CRITERIA),
        "models_in_input_order": names, "n_models": len(names),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "family_definition": "Existing HF organization / API provider labels; not verified training-lineage clusters",
        "statistic": "Kendall tau-b; ties retained",
        "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                      "draws": "choice(n_models, size=n_models, replace=True) once per resample",
                      "pairing": "Same model draws for every predictor and criterion; compute difference within each draw",
                      "interval": "2.5th and 97.5th percentiles of defined draws; pointwise, not simultaneous",
                      "undefined_draws": "Excluded only from that statistic's interval; count retained; no model exclusion"},
        "leave_one_family_out": "Point statistics only; main estimates always retain all 18 models",
        "limits": ["Model resampling does not include item sampling uncertainty",
                   "Model panel is not a random sample of all LLMs",
                   "Model bootstrap does not remove within-family dependence; LOFO is a numerical sensitivity check",
                   "Different block sizes and ceiling/ties remain; associations are not causal mechanism evidence",
                   "Public BFCL and current probes are different evaluations"],
        "remaining": ["Single-turn and parameter controls; specific adjustment methods still pending",
                      "E/F final scoring and remaining analysis specifications"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "associations.csv", associations)
    write_csv(OUT / "paired_differences.csv", differences)
    write_csv(OUT / "leave_one_family_out.csv", leave_out)
    (OUT / "analysis.json").write_text(json.dumps({"meta": metadata, "associations": associations,
        "paired_differences": differences, "leave_one_family_out": leave_out}, indent=2,
        ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    lines = ["# First BFCL comparison for 18 models", "",
             "2026-10-04. Three BFCL columns are reported in parallel. All 18 models are included; no model is excluded for score, format, or S3 status. "
             "The specification was fixed after model scores were visible, so the analysis is exploratory.", "",
             "A = core block, 60 items; C = single-step arithmetic, 90 items; D = cancellation control, 100 items. All three use the fixed strict numeric scoring. "
             "The statistic is Kendall τ-b with all ties retained. Intervals are 95% percentile intervals from 10,000 paired model bootstrap draws, seed=42. "
             "A/C/D and BFCL use the same model draws, and differences are computed within each resample; intervals are not adjusted for multiple comparisons.", "",
             "## Association of each block with BFCL", "",
             "| BFCL column | Task | n | τ-b | 95% interval | Valid/undefined resamples |",
             "| --- | --- | ---: | ---: | --- | --- |"]
    for r in associations:
        lines.append(f"| {r['criterion']} | {r['block']} | {r['n_models']} | {fmt(r['tau_b'])} | "
                     f"{interval(r)} | {r['boot_valid']}/{r['boot_undefined']} |")
    lines += ["", "## Direct comparison of association strength", "",
              "A positive difference means the association of A is stronger, a negative one that the control association is stronger. An interval that spans 0 means the direction of the difference is not clearly supported by this interval; "
              "it does not prove that the two are equal.", "",
              "| BFCL column | Difference | n | Δτ-b | 95% interval | Valid/undefined resamples |",
              "| --- | --- | ---: | ---: | --- | --- |"]
    for r in differences:
        lines.append(f"| {r['criterion']} | {r['comparison']} | {r['n_models']} | {fmt(r['delta_tau_b'])} | "
                     f"{interval(r)} | {r['boot_valid']}/{r['boot_undefined']} |")
    lines += ["", "## Numerical sensitivity to leaving out one family", "",
              "Groups follow the existing HF organization / API provider labels, 7 groups in total; after each omission only point statistics are reported, which do not replace the full 18-model panel.", "",
              "| BFCL column | Group left out | Remaining n | τ(A) | τ(C) | τ(D) | Δ(A−C) | Δ(A−D) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in leave_out:
        lines.append(f"| {r['criterion']} | {r['family_left_out']} | {r['n_models']} | " +
                     " | ".join(fmt(r[k]) for k in ("tau_A", "tau_C", "tau_D", "delta_A_C", "delta_A_D")) + " |")
    lines += ["", "## All models in the input", "",
              "BFCL scores keep the 0–100 scale of the public table. A/C/D are given as correct count and denominator for each model.", "",
              "| Model | Group | A | C | D | Overall | Multi-Turn | Agentic |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['family']} | " +
                     " | ".join(f"{r[b + '_correct']}/{r[b + '_n']}" for b in BLOCKS) + " | " +
                     " | ".join(r[f"bfcl_{c}_1216"] for c in CRITERIA) + " |")
    lines += ["", "## Scope of interpretation and follow-up work", "",
              "Intervals reflect model resampling only and not item sampling; the 18 models are not a random sample of all LLMs, and models of the same provider are dependent. "
              "Leaving out one family cannot prove the absence of family confounding. Format and truncation issues under the fixed protocol remain and are not removed by this analysis.", "",
              "This step has no permutation p-value, automatic significance, or claim threshold; no column of the three is picked as the main result. "
              "Single-turn BFCL and parameter-count controls remain to follow, with their specific statistics still pending. E/F scoring and other open analyses are not selected by this step. "
              "The manuscript was not modified; no model call or cost was added.", "",
              "Reproduce: `python3 analysis/run_bfcl_comparison.py`. The input is BFCL_INPUTS_20261004.csv in the parent directory; "
              "analysis.json keeps full-precision numbers, model order, sampling method, and software versions; the three CSV files store the associations, the paired differences, and the leave-one-family-out results."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"associations": associations, "paired_differences": differences}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
