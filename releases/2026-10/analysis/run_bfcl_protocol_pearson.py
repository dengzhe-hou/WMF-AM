#!/usr/bin/env python3
"""BFCL associations under protocols H and I, and a Pearson sensitivity for A, C, and D.

Added on 2026-10-05 at the author's decision. Same 18 models and the same 10,000 paired
model-bootstrap draws (seed 42) as run_bfcl_comparison.py. Kendall tau-b stays the primary
statistic. Before writing, the script recomputes A's tau-b with these draws and requires an
exact match with the stored bfcl_comparison_20261004 results, and it checks every Pearson
point estimate against scipy.stats.pearsonr and one resample against repeated rows.

Reproduce from any directory: python analysis/run_bfcl_protocol_pearson.py
"""

import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import pearsonr

from analyze import clean, kendall, model_weights, percentile_ci, sign_matrix, taub

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis/results"
INPUT = PANEL / "BFCL_INPUTS_20261004.csv"
REPORT = PANEL / "RUN_REPORT.json"
STORED = PANEL / "bfcl_comparison_20261004/associations.csv"
OUT = PANEL / "bfcl_protocol_pearson_20261005"
CRITERIA = ("overall", "multi_turn", "agentic")
N_BOOT = 10_000
SEED = 42


def summary(values):
    ci, valid = percentile_ci(values)
    return {"ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
            "boot_valid": valid, "boot_undefined": N_BOOT - valid}


def weighted_pearson(x, y, weights):
    """Pearson r for each resample, with model counts as frequency weights (= repeated rows)."""
    w = weights.astype(float)
    total = w.sum(axis=1)
    mx = (w @ x) / total
    my = (w @ y) / total
    dx = x[None, :] - mx[:, None]
    dy = y[None, :] - my[:, None]
    cov = (w * dx * dy).sum(axis=1)
    var = (w * dx * dx).sum(axis=1) * (w * dy * dy).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(var > 0, cov / np.sqrt(var), np.nan)
    return r


def main():
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    report = {m["model"]: m for m in json.loads(REPORT.read_text(encoding="utf-8"))["models"]}
    blocks = [report[r["run_label"]]["blocks"] for r in rows]
    x = {"A": np.array([int(r["A_correct"]) for r in rows]),
         "C": np.array([int(r["C_correct"]) for r in rows]),
         "D": np.array([int(r["D_correct"]) for r in rows])}
    for b, n in (("H", 60), ("I", 60)):
        assert all(bl[b]["denominator"] == n for bl in blocks)
        x[b] = np.array([bl[b]["numerator"] for bl in blocks])
    assert all(bl["A"]["numerator"] == a for bl, a in zip(blocks, x["A"]))
    y = {col: np.array([float(r[f"bfcl_{col}_1216"]) for r in rows]) for col in CRITERIA}
    weights = model_weights(len(rows), N_BOOT, SEED)
    with STORED.open(encoding="utf-8", newline="") as handle:
        stored = {(r["criterion"], r["block"]): r for r in csv.DictReader(handle)}

    kendall_rows, pearson_rows = [], []
    for col in CRITERIA:
        boot, point = {}, {}
        for b in ("A", "H", "I"):
            point[b] = clean(kendall(x[b], y[col]))
            boot[b] = taub(sign_matrix(x[b])[None], sign_matrix(y[col])[None], weights)
            row = {"statistic": "kendall_tau_b", "criterion": col, "term": b, "n_models": 18,
                   "estimate": point[b], **summary(boot[b])}
            if b == "A":
                ref = stored[col, "A"]
                for key, rkey in (("estimate", "tau_b"), ("ci95_low", "ci95_low"), ("ci95_high", "ci95_high")):
                    assert abs(row[key] - float(ref[rkey])) < 1e-12, (col, key)
            kendall_rows.append(row)
        for b in ("H", "I"):
            kendall_rows.append({"statistic": "kendall_tau_b", "criterion": col, "term": f"{b}-A",
                                 "n_models": 18, "estimate": point[b] - point["A"],
                                 **summary(boot[b] - boot["A"])})

        pboot, ppoint = {}, {}
        for b in ("A", "C", "D"):
            ppoint[b] = float(pearsonr(x[b], y[col]).statistic)
            pboot[b] = weighted_pearson(x[b].astype(float), y[col], weights)
            ones = weighted_pearson(x[b].astype(float), y[col], np.ones((1, 18), dtype=int))[0]
            assert abs(ones - ppoint[b]) < 1e-12, (col, b)
            draw = weights[0]
            rep = pearsonr(np.repeat(x[b], draw), np.repeat(y[col], draw)).statistic
            assert abs(pboot[b][0] - rep) < 1e-12, (col, b, "resample 0")
            pearson_rows.append({"statistic": "pearson_r", "criterion": col, "term": b, "n_models": 18,
                                 "estimate": ppoint[b], **summary(pboot[b])})
        for c in ("C", "D"):
            pearson_rows.append({"statistic": "pearson_r", "criterion": col, "term": f"A-{c}",
                                 "n_models": 18, "estimate": ppoint["A"] - ppoint[c],
                                 **summary(pboot["A"] - pboot[c])})

    metadata = {
        "purpose": "BFCL associations of the protocol variants H and I, and a Pearson sensitivity analysis",
        "decision": "Added 2026-10-05 at the author's decision, after the point estimates had been computed; Kendall tau-b remains the primary statistic",
        "scores": "correct counts; A, C, D from BFCL_INPUTS; H and I from RUN_REPORT.json (I uses its last-integer rule)",
        "models_in_input_order": [r["model"] for r in rows],
        "counts_in_input_order": {b: v.tolist() for b, v in x.items()},
        "checks": ["A tau-b recomputed with the same draws equals bfcl_comparison_20261004 (|diff| < 1e-12)",
                   "Pearson point estimates equal scipy.stats.pearsonr (|diff| < 1e-12)",
                   "Weighted Pearson of resample 0 equals pearsonr on repeated rows (|diff| < 1e-12)"],
        "bootstrap": {"n": N_BOOT, "seed": SEED, "rng": "numpy.random.RandomState",
                      "draws": "identical to run_bfcl_comparison.py; differences computed within each draw",
                      "interval": "2.5th and 97.5th percentiles of defined draws; pointwise"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    OUT.mkdir(exist_ok=True)
    for name, data in (("kendall_protocols.csv", kendall_rows), ("pearson.csv", pearson_rows)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (OUT / "analysis.json").write_text(json.dumps({"meta": metadata, "kendall_protocols": kendall_rows,
        "pearson": pearson_rows}, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for r in kendall_rows + pearson_rows:
        print(f"{r['statistic']:14s} {r['criterion']:10s} {r['term']:4s} {r['estimate']:.3f} "
              f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] valid={r['boot_valid']}")


if __name__ == "__main__":
    main()
