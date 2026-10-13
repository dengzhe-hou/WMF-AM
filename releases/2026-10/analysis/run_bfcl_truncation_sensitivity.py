#!/usr/bin/env python3
"""Sensitivity of the core-minus-arithmetic BFCL difference to truncation-dominated models (2026-10-11).

Reproduce from any directory: python analysis/run_bfcl_truncation_sensitivity.py
Same kernels, criteria, resampling and seed as run_bfcl_comparison.py, on the models that truncated at most half
of their 60 core responses. This analysis was added after the main results and a review round; it is not
pre-specified. The main estimates keep all 18 models.
"""

import csv
import json
import platform

import numpy as np
import scipy

from analyze import clean, kendall, model_weights, percentile_ci, sign_matrix, taub
from run_bfcl_comparison import CRITERIA, INPUT, N_BOOT, OUT, SEED, summary, write_csv


def main():
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18 and all(int(r["A_n"]) == 60 for r in rows)
    left_out = [r["model"] for r in rows if int(r["A_truncated"]) > int(r["A_n"]) / 2]
    keep = [r for r in rows if r["model"] not in left_out]
    x = {block: np.array([int(r[f"{block}_correct"]) for r in keep]) for block in ("A", "C")}
    weights = model_weights(len(keep), N_BOOT, SEED)
    out = []
    for col in CRITERIA:
        y = np.array([float(r[f"bfcl_{col}_1216"]) for r in keep])
        points = {block: clean(kendall(x[block], y)) for block in x}
        boot = {block: taub(sign_matrix(x[block])[None], sign_matrix(y)[None], weights) for block in x}
        out.append({"criterion": col, "comparison": "A-C", "n_models": len(keep),
                    "tau_A": points["A"], "tau_C": points["C"], "delta_tau_b": points["A"] - points["C"],
                    **summary(boot["A"] - boot["C"])})
    write_csv(OUT / "truncation_sensitivity.csv", out)
    meta = {"analysis_status": "sensitivity analysis added after the main results; not pre-specified",
            "rule": "leave out models with more than half of their 60 core responses truncated",
            "models_left_out": left_out, "n_models": len(keep),
            "statistic": "Kendall tau-b; ties retained",
            "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                          "draws": "choice(n_models, size=n_models, replace=True) once per resample",
                          "interval": "2.5th and 97.5th percentiles of defined draws; pointwise, not simultaneous"},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    (OUT / "truncation_sensitivity.json").write_text(json.dumps({"meta": meta, "paired_differences": out}, indent=2,
                                                                ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for r in out:
        print(r["criterion"], f"{r['delta_tau_b']:.3f}", f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
