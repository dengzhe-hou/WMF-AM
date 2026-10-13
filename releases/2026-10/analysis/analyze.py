#!/usr/bin/env python3
"""WMF-AM analysis: block scores against the BFCL criterion.

    python analyze.py --scored scored/scored_items.csv --out-dir results --g-rule RULE
    python analyze.py --scored scored/scored_items.csv --out-dir results --g-rule RULE \
        --drop-set floor_a=Model1,Model2 --drop-set floor_b=Model3 \
        --exclude-models MiniCPM3-4B

--g-rule {first_int,last_int,strict} is required and has no default: the integer
rule for block G (paraphrase stability) is not decided here.
first_int = the submitted rule (wmf_am_paraphrase.py L175-177), last_int = the
last integer, strict = scheme one. score_bank.py writes all three columns.

Input: scored_items.csv from score_bank.py, bfcl_criterion_19.csv and
model_freeze_19.csv. Output: analysis.json (every number) and
analysis_summary.md (tables). No column is treated as primary: BFCL Overall and
the five sub-scores are computed the same way and listed side by side.

X is the block A accuracy under scoring scheme one (correct_strict).

Statistics
  tau          Kendall tau-b.
  ci95         percentile bootstrap over models, as bootstrap_kendall_ci in
               code/consolidate_n28.py (RandomState(seed).choice per resample,
               undefined resamples dropped).
  ci95_family  the same, resampling whole model families.
  perm_p       two-sided permutation test on the model labels. n <= 9: all n!
               permutations, p = share with |S*| >= |S|. Larger n: Monte Carlo,
               p = (count + 1) / (n_perm + 1).
  partial      "Kendall tau-b on rank residuals", the method of
               consolidate_n28.partial_kendall: average ranks of X and of Y are
               each regressed linearly on the ranks of the control, and tau-b is
               taken between the two residual vectors. This is not Kendall's
               partial tau formula. With two controls the ranks are regressed on
               both control ranks (extension, not in the submitted code).
               Because only ranks of the control enter, parameter count and log
               parameter count give the same number.
               Correction to the submitted function: residuals that are equal in
               exact arithmetic count as ties here (difference below 1e-9). The
               submitted function adds 1e-12 to the slope's denominator, which
               leaves such pairs about 1e-14 apart, and scipy then orders them.
               `tau_submitted_code` is the value of the submitted function,
               unchanged, on the same vectors; `differs_from_submitted_code`
               says whether the two disagree.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, rankdata

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent / "external"

MAIN_Y = ["overall", "non_live", "live", "irrelevance", "multi_turn", "agentic"]
SUB_Y = ["mt_base", "mt_miss_func", "mt_miss_param", "mt_long_context",
         "web_search", "ws_base", "ws_no_snippet", "memory", "mem_kv", "mem_vector", "mem_rec_sum"]
SNAPSHOTS = {"2025-12-16": "1216", "2025-11-03": "1103", "2025-08-26": "0826"}
MAIN_SNAPSHOT = "2025-12-16"
SINGLE_TURN = ("non_live", "live")          # Z1 = their mean
# Parameter counts missing from model_freeze_19.csv (Hugging Face API, 2026-10-02).
EXTRA_PARAMS = {"Command A": 111_057_580_032, "Command R7B": 8_028_033_024}
TEMPLATE_ORDER = ["original", "formal", "casual", "minimal", "verbose"]
G_RULE_COLUMN = {"first_int": "correct_first_int", "last_int": "correct_last_int", "strict": "correct_strict"}
RESIDUAL_TIE = 1e-9
EXACT_PERM_MAX_N = 9
MIN_N = 4                                   # fewer models: no statistic is reported
THRESHOLD = 0.90


# ── Kendall tau-b kernel (batched, with integer case weights) ────────────────

def sign_matrix(values: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """(..., n) -> (..., n, n) of sign(v_i - v_j); |difference| <= tol is a tie."""
    diff = values[..., :, None] - values[..., None, :]
    signs = np.sign(diff).astype(np.int8)
    if tol:
        signs[np.abs(diff) <= tol] = 0
    return signs


def taub(sx: np.ndarray, sy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """tau-b for each row of `weights` (B, n). A weight c means the model occurs
    c times in that sample. sx, sy: (B, n, n) or (1, n, n) sign matrices."""
    w = weights.astype(float)
    pair = w[:, :, None] * w[:, None, :]
    total = w.sum(1)
    n0 = total * (total - 1) / 2
    same_model = (w * (w - 1)).sum(1) / 2          # pairs of copies of one model: tied in x and y
    diagonal = (w * w).sum(1)
    numerator = (pair * (sx * sy)).sum((1, 2)) / 2
    ties_x = ((pair * (sx == 0)).sum((1, 2)) - diagonal) / 2 + same_model
    ties_y = ((pair * (sy == 0)).sum((1, 2)) - diagonal) / 2 + same_model
    dx, dy = n0 - ties_x, n0 - ties_y
    defined = (dx > 0) & (dy > 0)
    tau = np.full(len(w), np.nan)
    tau[defined] = numerator[defined] / np.sqrt(dx[defined]) / np.sqrt(dy[defined])
    return np.clip(tau, -1.0, 1.0)


def weighted_rank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average ranks of `values` in each weighted sample; shape (B, n)."""
    less = (values[None, :] < values[:, None]).astype(float)     # [i, j]: v_j < v_i
    equal = (values[None, :] == values[:, None]).astype(float)
    return weights @ less.T + (weights @ equal.T + 1) / 2


def rank_residual(ranks: np.ndarray, controls: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ranks - controls @ beta, beta from weighted least squares on centred ranks.
    The 1e-12 ridge is the 1e-12 of consolidate_n28.partial_kendall."""
    total = weights.sum(1, keepdims=True)
    centred = controls - (weights[:, :, None] * controls).sum(1, keepdims=True) / total[:, :, None]
    target = ranks - (weights * ranks).sum(1, keepdims=True) / total
    gram = np.einsum("bi,bip,biq->bpq", weights, centred, centred) + 1e-12 * np.eye(controls.shape[2])
    moment = np.einsum("bi,bip,bi->bp", weights, centred, target)
    beta = np.linalg.solve(gram, moment[:, :, None])[:, :, 0]
    return ranks - np.einsum("bip,bp->bi", controls, beta)


def signs_for(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray], weights: np.ndarray):
    """Sign matrices of x and y, or of their rank residuals when controls are given."""
    if not controls:
        return sign_matrix(x)[None], sign_matrix(y)[None]
    w = weights.astype(float)
    control_ranks = np.stack([weighted_rank(z, w) for z in controls], axis=2)
    return (sign_matrix(rank_residual(weighted_rank(x, w), control_ranks, w), RESIDUAL_TIE),
            sign_matrix(rank_residual(weighted_rank(y, w), control_ranks, w), RESIDUAL_TIE))


def kendall(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray] | None = None) -> float:
    ones = np.ones((1, len(x)))
    sx, sy = signs_for(np.asarray(x, float), np.asarray(y, float), controls or [], ones)
    return float(taub(sx, sy, ones)[0])


def submitted_partial_kendall(x, y, z) -> float:
    """consolidate_n28.partial_kendall L124-131, unchanged (statistic only)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(r, c):
        cm = c - np.mean(c)
        beta = np.dot(cm, r - np.mean(r)) / (np.dot(cm, cm) + 1e-12)
        return r - beta * c
    return float(kendalltau(resid(rx, rz), resid(ry, rz)).statistic)


# ── Resampling ───────────────────────────────────────────────────────────────

def model_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Counts per model for each resample; same index stream as bootstrap_kendall_ci."""
    rng = np.random.RandomState(seed)
    return np.stack([np.bincount(rng.choice(n, size=n, replace=True), minlength=n) for _ in range(n_boot)])


def family_weights(families: list[str], n_boot: int, seed: int) -> np.ndarray:
    names = sorted(set(families))
    member = np.array([[family == name for family in families] for name in names], dtype=int)   # (F, n)
    rng = np.random.RandomState(seed)
    draws = np.stack([np.bincount(rng.choice(len(names), size=len(names), replace=True), minlength=len(names))
                      for _ in range(n_boot)])
    return draws @ member


def percentile_ci(values: np.ndarray) -> tuple[list[float] | None, int]:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return None, 0
    return [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))], int(len(valid))


def permutation_p(sx: np.ndarray, sy: np.ndarray, n_perm: int, seed: int) -> tuple[float, str]:
    """Two-sided p for the pair statistic S; the tau-b denominator does not change under permutation."""
    n = sx.shape[0]
    sx_flat = sx.astype(np.int32).ravel()
    sy_flat = sy.ravel()
    observed = abs(int(sx_flat @ sy_flat.astype(np.int32)))
    exact = n <= EXACT_PERM_MAX_N
    if exact:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int32)
    else:
        rng = np.random.default_rng(seed)
        perms = rng.permuted(np.tile(np.arange(n, dtype=np.int32), (n_perm, 1)), axis=1)
    count = 0
    for start in range(0, len(perms), 20000):
        p = perms[start:start + 20000]
        permuted = sy_flat[(p[:, :, None] * n + p[:, None, :]).reshape(len(p), -1)]
        count += int((np.abs(permuted.astype(np.int32) @ sx_flat) >= observed).sum())
    if exact:
        return count / len(perms), f"exact, {n}! = {len(perms)} permutations"
    return (count + 1) / (n_perm + 1), f"Monte Carlo, {n_perm} permutations, seed {seed}"


class Config:
    def __init__(self, n_boot: int, n_perm: int, seed: int):
        self.n_boot, self.n_perm, self.seed = n_boot, n_perm, seed


def clean(value: float) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


def tau_report(x, y, cfg: Config, controls: list | None = None, families: list[str] | None = None,
               inference: bool = True) -> dict[str, Any]:
    """tau-b (or tau-b on rank residuals) with bootstrap CI and permutation p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    controls = [np.asarray(z, float) for z in (controls or [])]
    n = len(x)
    out: dict[str, Any] = {"n": n}
    if n < MIN_N:
        out.update(tau=None, note=f"fewer than {MIN_N} models")
        return out
    ones = np.ones((1, n))
    sx, sy = signs_for(x, y, controls, ones)
    tau = clean(taub(sx, sy, ones)[0])
    out["tau"] = tau
    if len(controls) == 1:
        out["tau_submitted_code"] = clean(submitted_partial_kendall(x, y, controls[0]))
        both = tau is not None and out["tau_submitted_code"] is not None
        out["differs_from_submitted_code"] = bool(both and abs(tau - out["tau_submitted_code"]) > 1e-9)
    if tau is None:
        out["note"] = "undefined: one of the two vectors is constant"
        return out
    if not inference:
        return out
    weights = model_weights(n, cfg.n_boot, cfg.seed)
    out["ci95"], out["boot_valid"] = percentile_ci(taub(*signs_for(x, y, controls, weights), weights))
    if families is not None:
        weights = family_weights(families, cfg.n_boot, cfg.seed)
        out["ci95_family"], out["boot_valid_family"] = percentile_ci(taub(*signs_for(x, y, controls, weights), weights))
        out["n_families"] = len(set(families))
    out["perm_p"], out["perm_method"] = permutation_p(sx[0], sy[0], cfg.n_perm, cfg.seed)
    return out


def paired_difference(x1, x2, y, cfg: Config) -> dict[str, Any]:
    """tau(x1, y) - tau(x2, y) with a bootstrap CI from the same resamples."""
    x1, x2, y = (np.asarray(v, float) for v in (x1, x2, y))
    n = len(y)
    if n < MIN_N:
        return {"n": n, "delta": None, "note": f"fewer than {MIN_N} models"}
    t1, t2 = kendall(x1, y), kendall(x2, y)
    if math.isnan(t1) or math.isnan(t2):
        return {"n": n, "delta": None, "note": "undefined: a vector is constant"}
    weights = model_weights(n, cfg.n_boot, cfg.seed)
    sy = sign_matrix(y)[None]
    diff = taub(sign_matrix(x1)[None], sy, weights) - taub(sign_matrix(x2)[None], sy, weights)
    ci, valid = percentile_ci(diff)
    return {"n": n, "delta": t1 - t2, "ci95": ci, "boot_valid": valid}


# ── K-sweep fit (form, start values and bounds of consolidate_n28.sigmoid_fit) ─

def sigmoid(k, a, alpha, k_crit, offset):
    with np.errstate(over="ignore"):
        return a / (1.0 + np.exp(alpha * (k - k_crit))) + offset


def sigmoid_fit(k_values: list[float], accuracies: list[float]) -> dict[str, Any]:
    k, acc = np.asarray(k_values, float), np.asarray(accuracies, float)
    ss_tot = float(np.sum((acc - acc.mean()) ** 2))
    if ss_tot == 0:
        return {"fit": False, "note": "flat curve: accuracy is the same at every K"}
    lower, upper = [0, 0.01, 0.1, -0.5], [2, 5, 200, 0.5]
    try:
        popt, _ = curve_fit(sigmoid, k, acc, p0=[1.0, 0.5, 10.0, 0.0], bounds=(lower, upper), maxfev=10000)
    except RuntimeError as exc:             # curve_fit's "no convergence" signal
        return {"fit": False, "note": str(exc)}
    ss_res = float(np.sum((acc - sigmoid(k, *popt)) ** 2))
    return {"fit": True, "a": float(popt[0]), "alpha": float(popt[1]), "K_crit": float(popt[2]),
            "offset": float(popt[3]), "R2": 1 - ss_res / (ss_tot + 1e-12),
            "K_crit_at_bound": bool(min(popt[2] - lower[2], upper[2] - popt[2]) < 1e-6)}


# ── Loading ──────────────────────────────────────────────────────────────────

def number(text: str) -> float | None:
    return float(text) if text.strip() else None


def load_models(criterion_csv: Path, freeze_csv: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Model table keyed by the criterion file's model name, and name aliases."""
    with criterion_csv.open(encoding="utf-8", newline="") as handle:
        criterion = list(csv.DictReader(handle))
    with freeze_csv.open(encoding="utf-8", newline="") as handle:
        freeze = {row["model"].lower(): row for row in csv.DictReader(handle)}
    models: dict[str, dict[str, Any]] = {}
    alias: dict[str, str] = {}
    for row in criterion:
        name = row["model"]
        frozen = freeze[name.lower()]
        handler = row["model_id_sent_by_handler"]
        family = handler.split("/")[0] if row["kind"] == "open" else frozen["openrouter_slug"].split("/")[0]
        raw_params = frozen["params"].strip()
        approx = raw_params.startswith("~")
        params = float(raw_params.lstrip("~").split()[0]) if raw_params else EXTRA_PARAMS.get(name)
        scores = {}
        for snapshot, suffix in SNAPSHOTS.items():
            scores[snapshot] = {col: number(row[f"{col}_{suffix}"]) for col in MAIN_Y + SUB_Y if f"{col}_{suffix}" in row}
        models[name] = {"kind": row["kind"], "family": family.lower(), "params": params,
                        "params_approximate": approx, "bfcl": scores}
        for key in (name, handler.split("/")[-1]):
            alias[key.lower()] = name
    if len(models) != len(criterion):
        sys.exit(f"{criterion_csv}: duplicate model names")
    return models, alias


def resolve(names: list[str], alias: dict[str, str]) -> list[str]:
    unknown = [name for name in names if name.lower() not in alias]
    if unknown:
        sys.exit(f"unknown model name(s): {unknown}. Known: {sorted(set(alias.values()))}")
    return [alias[name.lower()] for name in names]


def load_scored(path: Path, alias: dict[str, str]) -> tuple[dict[str, list[dict[str, str]]], bool]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    absent = [c for c in (*G_RULE_COLUMN.values(), "truncated") if c not in reader.fieldnames]
    if absent:
        sys.exit(f"{path}: no column(s) {absent}; it was written by an older score_bank.py, score again")
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        name = resolve([row["model"]], alias)[0]
        by_model[name].append(row)
        labels[name].add(row["model"])
    # run label and leaderboard name are both accepted, so two output files can name one model
    clash = {name: sorted(found) for name, found in labels.items() if len(found) > 1}
    if clash:
        sys.exit(f"{path}: more than one label for the same model (their items would be pooled): {clash}")
    synthetic = {row["synthetic"] for row in rows}
    if len(synthetic) != 1:
        sys.exit(f"{path}: synthetic and real rows mixed")
    return by_model, synthetic.pop() == "1"


def accuracy(rows: list[dict[str, str]], keep: Callable[[dict[str, str]], bool],
             column: str = "correct_strict") -> float | None:
    """Share correct among scored rows that pass `keep`. Missing rows are left out."""
    hits = [int(row[column]) for row in rows if row["status"] == "scored" and keep(row)]
    return sum(hits) / len(hits) if hits else None


# ── Analysis ─────────────────────────────────────────────────────────────────

class Analysis:
    def __init__(self, models: dict[str, dict[str, Any]], scored: dict[str, list[dict[str, str]]], cfg: Config):
        self.models, self.scored, self.cfg = models, scored, cfg
        self.names = [name for name in models if name in scored]       # criterion file order

    def block(self, block: str, column: str = "correct_strict", **where: str) -> dict[str, float | None]:
        def keep(row):
            return row["block"] == block and all(row[key] == value for key, value in where.items())
        return {name: accuracy(self.scored[name], keep, column) for name in self.names}

    def bfcl(self, column: str, snapshot: str = MAIN_SNAPSHOT) -> dict[str, float | None]:
        return {name: self.models[name]["bfcl"][snapshot].get(column) for name in self.names}

    def single_turn(self, snapshot: str = MAIN_SNAPSHOT) -> dict[str, float | None]:
        a, b = (self.bfcl(col, snapshot) for col in SINGLE_TURN)
        return {name: None if a[name] is None or b[name] is None else (a[name] + b[name]) / 2 for name in self.names}

    def params(self) -> dict[str, float | None]:
        return {name: None if self.models[name]["params"] is None else math.log10(self.models[name]["params"])
                for name in self.names}

    def report(self, x: dict, y: dict, controls: list[dict] | None = None, keep: list[str] | None = None,
               inference: bool = True, family_ci: bool = False) -> dict[str, Any]:
        """tau_report on the models that have every value."""
        vectors = [x, y] + (controls or [])
        names = [name for name in (keep or self.names) if all(v[name] is not None for v in vectors)]
        families = [self.models[name]["family"] for name in names] if family_ci else None
        out = tau_report([x[n] for n in names], [y[n] for n in names], self.cfg,
                         [[z[n] for n in names] for z in (controls or [])], families, inference)
        return out

    def association(self, x: dict, keep: list[str] | None = None, snapshot: str = MAIN_SNAPSHOT,
                    columns: list[str] | None = None) -> dict[str, Any]:
        """X against each BFCL column: tau, and tau on rank residuals for each control set."""
        z1, z2 = self.single_turn(snapshot), self.params()
        available = self.models[self.names[0]]["bfcl"][snapshot]
        out = {}
        for col in columns or [c for c in MAIN_Y + SUB_Y if c in available]:
            y = self.bfcl(col, snapshot)
            entry = {"tau": self.report(x, y, keep=keep, family_ci=True),
                     "partial_single_turn": self.report(x, y, [z1], keep),
                     "partial_log_params": self.report(x, y, [z2], keep),
                     "partial_single_turn_and_log_params": self.report(x, y, [z1, z2], keep)}
            if col in SINGLE_TURN:
                note = "this column is one of the two components of the single-turn control"
                entry["partial_single_turn"]["component_note"] = note
                entry["partial_single_turn_and_log_params"]["component_note"] = note
            out[col] = entry
        return out

    def leave_one_family_out(self, x: dict, y: dict, controls: list[dict] | None = None) -> dict[str, Any]:
        per_family = {}
        for family in sorted({self.models[name]["family"] for name in self.names}):
            keep = [name for name in self.names if self.models[name]["family"] != family]
            result = self.report(x, y, controls, keep, inference=False)
            per_family[family] = {"n": result["n"], "tau": result["tau"]}
        taus = [entry["tau"] for entry in per_family.values() if entry["tau"] is not None]
        return {"per_family": per_family, "range": [min(taus), max(taus)] if taus else None}


def summarize(values: dict[str, float | None]) -> dict[str, Any]:
    present = [v for v in values.values() if v is not None]
    return {"n_models": len(present), "mean": float(np.mean(present)) if present else None,
            "min": min(present) if present else None, "max": max(present) if present else None}


def at_or_above(values: dict[str, float | None]) -> dict[str, Any]:
    present = {name: v for name, v in values.items() if v is not None}
    hit = [name for name, v in present.items() if v >= THRESHOLD]
    return {"threshold": THRESHOLD, "n_at_or_above": len(hit), "n_models": len(present),
            "mean": float(np.mean(list(present.values()))) if present else None,
            "below": {name: v for name, v in present.items() if v < THRESHOLD}}


def pairwise(an: Analysis, vectors: dict[str, dict]) -> dict[str, Any]:
    """tau and permutation p for every pair of conditions (same models)."""
    pairs = {}
    for a, b in itertools.combinations(vectors, 2):
        result = an.report(vectors[a], vectors[b])
        pairs[f"{a}|{b}"] = {key: result.get(key) for key in ("n", "tau", "perm_p", "perm_method", "note")
                             if key in result}
    taus = [p["tau"] for p in pairs.values() if p["tau"] is not None]
    return {"pairs": pairs, "n_pairs": len(pairs), "mean_tau": float(np.mean(taus)) if taus else None,
            "n_pairs_tau_positive": sum(t > 0 for t in taus),
            "n_pairs_p_below_05": sum(p.get("perm_p") is not None and p["perm_p"] < 0.05 for p in pairs.values())}


def ablations(an: Analysis, x: dict, g_rule: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    criteria = {col: an.bfcl(col) for col in MAIN_Y}
    rows_of = an.scored

    # K=1 control (block C) and CoT wrapper (block I)
    out["k1_control"] = at_or_above(an.block("C"))
    out["cot_wrapper"] = at_or_above(an.block("I"))

    # yoked control (block D)
    yoked = an.block("D")
    out["yoked_control"] = {"accuracy": summarize(yoked), "against_criterion": {}}
    for col, y in criteria.items():
        names = [n for n in an.names if None not in (x[n], yoked[n], y[n])]
        out["yoked_control"]["against_criterion"][col] = {
            "tau_yoked": an.report(yoked, y),
            "tau_wmf_am": an.report(x, y, inference=False)["tau"],
            "delta_wmf_am_minus_yoked": paired_difference([x[n] for n in names], [yoked[n] for n in names],
                                                          [y[n] for n in names], an.cfg)}

    # non-arithmetic blocks, under both scoring rules (Q17 open)
    out["nonarithmetic_direct"], out["nonarithmetic_cumulative"] = {}, {}
    for label, column in (("strict", "correct_strict"), ("submitted", "correct_submitted")):
        out["nonarithmetic_direct"][label] = summarize(an.block("E", column))
        f = an.block("F", column)
        domains = sorted({row["domain"] for row in rows_of[an.names[0]] if row["block"] == "F"})
        ks = sorted({row["k"] for row in rows_of[an.names[0]] if row["block"] == "F"}, key=int)
        high_k = {name: accuracy(rows_of[name], lambda r: r["block"] == "F" and int(r["k"]) >= 5, column)
                  for name in an.names}
        perfect = [name for name, v in high_k.items() if v == 1.0]
        out["nonarithmetic_cumulative"][label] = {
            "accuracy": summarize(f),
            "mean_accuracy_by_k": {k: summarize(an.block("F", column, k=k))["mean"] for k in ks},
            "tau_with_block_A": an.report(x, f, family_ci=True),
            "tau_with_block_A_leave_one_family_out": an.leave_one_family_out(x, f),
            "tau_with_block_A_by_domain": {d: an.report(x, an.block("F", column, domain=d)) for d in domains},
            "tau_with_criterion": {col: an.report(f, y) for col, y in criteria.items()},
            "models_perfect_at_k_ge_5": {"n": len(perfect), "n_models": sum(v is not None for v in high_k.values())},
        }

    # paraphrase stability (block G), under the rule given by --g-rule
    present = {row["template"] for row in rows_of[an.names[0]] if row["block"] == "G"}
    order = [t for t in TEMPLATE_ORDER if t in present]
    templates = {t: an.block("G", G_RULE_COLUMN[g_rule], template=t) for t in order}
    out["paraphrase"] = {"g_rule": g_rule, "column": G_RULE_COLUMN[g_rule],
                         "mean_accuracy_by_template": {t: summarize(v)["mean"] for t, v in templates.items()},
                         "mean_truncated_rate_by_template": {
                             t: summarize(an.block("G", "truncated", template=t))["mean"] for t in order},
                         **pairwise(an, templates)}

    # chat template (block H against block A)
    chat = an.block("H")
    out["chat_template"] = {"mean_accuracy_bare": summarize(x)["mean"], "mean_accuracy_chat": summarize(chat)["mean"],
                            "tau_bare_chat": an.report(x, chat)}

    # multi-seed reliability inside block A
    seeds = sorted({row["seed"] for row in rows_of[an.names[0]] if row["block"] == "A"}, key=int)
    by_seed = {seed: an.block("A", seed=seed) for seed in seeds}
    spread = [float(np.std([by_seed[s][n] for s in seeds], ddof=1)) for n in an.names
              if all(by_seed[s][n] is not None for s in seeds)]
    out["multiseed"] = {"mean_sd_across_seeds": float(np.mean(spread)) if spread else None, **pairwise(an, by_seed)}

    # K-sweep (blocks A and B)
    k_levels = sorted({int(row["k"]) for row in rows_of[an.names[0]] if row["block"] in ("A", "B")})
    at_k = {k: {name: accuracy(rows_of[name], lambda r, k=k: r["block"] in ("A", "B") and int(r["k"]) == k)
                for name in an.names} for k in k_levels}
    fits, k_crit = {}, {}
    for name in an.names:
        curve = [(k, at_k[k][name]) for k in k_levels if at_k[k][name] is not None]
        fits[name] = {"accuracy_by_k": {str(k): v for k, v in curve}}
        if len(curve) < len(k_levels):
            fits[name].update(fit=False, note="K level without scored items")
        else:
            fits[name].update(sigmoid_fit([k for k, _ in curve], [v for _, v in curve]))
        k_crit[name] = fits[name].get("K_crit")
    fitted = [f for f in fits.values() if f["fit"]]
    out["k_sweep"] = {
        "k_levels": k_levels, "per_model": fits,
        "mean_accuracy_by_k": {str(k): summarize(at_k[k])["mean"] for k in k_levels},
        "n_models_fitted": len(fitted), "n_models_r2_above_090": sum(f["R2"] > 0.90 for f in fitted),
        "n_models_k_crit_at_bound": sum(f["K_crit_at_bound"] for f in fitted),
        "k_crit_with_criterion": {col: an.report(k_crit, y) for col, y in criteria.items()},
        "accuracy_at_k_with_criterion": {str(k): {col: an.report(at_k[k], y) for col, y in criteria.items()}
                                         for k in k_levels},
    }
    return out


# ── Markdown ─────────────────────────────────────────────────────────────────

def f3(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def cell(report: dict[str, Any]) -> str:
    """'tau [lo, hi], p=..., n=..' for one tau_report."""
    if report.get("tau") is None:
        return f"n/a (n={report['n']})"
    text = f3(report["tau"])
    if report.get("ci95"):
        text += f" [{f3(report['ci95'][0])}, {f3(report['ci95'][1])}]"
    if report.get("perm_p") is not None:
        text += f", p={report['perm_p']:.4f}"
    text += f", n={report['n']}"
    if report.get("differs_from_submitted_code"):
        text += f" (submitted code: {f3(report['tau_submitted_code'])})"
    return text


def association_table(table: dict[str, Any]) -> list[str]:
    lines = ["| BFCL column | tau-b [95% CI], permutation p | given single-turn | given log params | given both |",
             "|---|---|---|---|---|"]
    for col, entry in table.items():
        lines.append(f"| {col} | {cell(entry['tau'])} | {cell(entry['partial_single_turn'])} | "
                     f"{cell(entry['partial_log_params'])} | {cell(entry['partial_single_turn_and_log_params'])} |")
    return lines


def markdown(result: dict[str, Any]) -> str:
    meta, abl = result["meta"], result["ablations"]
    lines = ["# WMF-AM against BFCL: analysis summary", ""]
    if meta["synthetic"]:
        lines += ["**SYNTHETIC DATA. These numbers come from generated responses, not from models.**", ""]
    if meta["incomplete"]:
        lines += [f"**INCOMPLETE DATA.** {json.dumps(meta['incomplete'], ensure_ascii=False)}", ""]
    lines += [f"Models: {meta['n_models']}. Excluded by option: {meta['excluded_models'] or 'none'}.",
              f"X = block A accuracy, scoring scheme one. Bootstrap {meta['n_boot']}, permutations {meta['n_perm']}, seed {meta['seed']}.",
              "Given columns are Kendall tau-b on rank residuals (see analysis.json meta.methods). "
              "Where the submitted function gives another value on the same vectors, it is shown in brackets. "
              "All BFCL columns are computed the same way; none is designated in advance.",
              f"X: min {f3(meta['x']['min'])}, max {f3(meta['x']['max'])}, models at 1.0: {meta['x']['n_at_ceiling']}.", "",
              f"## All models, BFCL {MAIN_SNAPSHOT}", ""]
    lines += association_table(result["association"])
    lines += ["", "Family-clustered 95% CI of tau-b:", "", "| BFCL column | CI | families |", "|---|---|---|"]
    for col in MAIN_Y:
        tau = result["association"][col]["tau"]
        ci = tau.get("ci95_family")
        lines.append(f"| {col} | {'n/a' if not ci else f'[{f3(ci[0])}, {f3(ci[1])}]'} | {tau.get('n_families', 'n/a')} |")
    for name, table in result["strata"].items():
        lines += ["", f"## Stratum: {name}", ""] + association_table({c: table[c] for c in MAIN_Y})
    lines += ["", "## Leave one family out (range of tau-b)", "", "| BFCL column | tau range | given single-turn range |", "|---|---|---|"]
    for col in MAIN_Y:
        lofo = result["leave_one_family_out"][col]
        rng = lambda r: "n/a" if r is None else f"{f3(r[0])} to {f3(r[1])}"
        lines.append(f"| {col} | {rng(lofo['tau']['range'])} | {rng(lofo['partial_single_turn']['range'])} |")
    for snapshot, table in result["snapshots"].items():
        lines += ["", f"## Snapshot {snapshot}", ""] + association_table(table)
    for name, entry in result["drop_sets"].items():
        lines += ["", f"## Without set '{name}' ({', '.join(entry['dropped'])})", ""]
        lines += association_table({c: entry["association"][c] for c in MAIN_Y})
    lines += ["", "## X under the submitted (lenient) integer rule", "", "| BFCL column | tau-b [95% CI], permutation p |", "|---|---|"]
    lines += [f"| {col} | {cell(rep)} |" for col, rep in result["x_submitted_rule"].items()]

    lines += ["", "## Ablations", "", "| Item | Value |", "|---|---|"]
    k1, cot = abl["k1_control"], abl["cot_wrapper"]
    lines.append(f"| K=1 control: models at or above {THRESHOLD:.2f} | {k1['n_at_or_above']}/{k1['n_models']} (mean {f3(k1['mean'])}) |")
    lines.append(f"| CoT wrapper: models at or above {THRESHOLD:.2f} | {cot['n_at_or_above']}/{cot['n_models']} (mean {f3(cot['mean'])}) |")
    lines.append(f"| Yoked control: mean accuracy | {f3(abl['yoked_control']['accuracy']['mean'])} |")
    for label in ("strict", "submitted"):
        direct, cumulative = abl["nonarithmetic_direct"][label], abl["nonarithmetic_cumulative"][label]
        lines.append(f"| Non-arithmetic direct (E), {label} rule: mean accuracy | {f3(direct['mean'])} |")
        lines.append(f"| Non-arithmetic cumulative (F), {label} rule: mean accuracy | {f3(cumulative['accuracy']['mean'])} |")
        lines.append(f"| tau(block F, block A), {label} rule | {cell(cumulative['tau_with_block_A'])} |")
    para = abl["paraphrase"]
    lines.append(f"| Paraphrase ({para['g_rule']} rule): mean pairwise tau over {para['n_pairs']} pairs | {f3(para['mean_tau'])}; "
                 f"{para['n_pairs_tau_positive']} positive, {para['n_pairs_p_below_05']} with p<0.05 |")
    lines.append(f"| Paraphrase ({para['g_rule']} rule): mean accuracy by template | " +
                 ", ".join(f"{t} {f3(v)}" for t, v in para["mean_accuracy_by_template"].items()) + " |")
    lines.append("| Paraphrase: mean share truncated (finish_reason length) by template | " +
                 ", ".join(f"{t} {f3(v)}" for t, v in para["mean_truncated_rate_by_template"].items()) + " |")
    lines.append(f"| Chat template: tau(bare, chat) | {cell(abl['chat_template']['tau_bare_chat'])} |")
    seeds = abl["multiseed"]
    lines.append(f"| Multi-seed: mean SD across seeds; mean cross-seed tau | {f3(seeds['mean_sd_across_seeds'])}; {f3(seeds['mean_tau'])} |")
    sweep = abl["k_sweep"]
    lines.append(f"| K-sweep: models fitted; R2 > 0.90; K_crit at a bound | {sweep['n_models_fitted']}; "
                 f"{sweep['n_models_r2_above_090']}; {sweep['n_models_k_crit_at_bound']} |")

    lines += ["", "Against each BFCL column:", "",
              "| BFCL column | tau(yoked) | tau(WMF-AM) - tau(yoked) [95% CI] | tau(block F, strict) | tau(block F, submitted) | tau(K_crit) |",
              "|---|---|---|---|---|---|"]
    for col in MAIN_Y:
        yoked = abl["yoked_control"]["against_criterion"][col]
        delta = yoked["delta_wmf_am_minus_yoked"]
        delta_text = ("n/a" if delta["delta"] is None or not delta["ci95"]
                      else f"{f3(delta['delta'])} [{f3(delta['ci95'][0])}, {f3(delta['ci95'][1])}]")
        lines.append(f"| {col} | {cell(yoked['tau_yoked'])} | {delta_text} | "
                     f"{cell(abl['nonarithmetic_cumulative']['strict']['tau_with_criterion'][col])} | "
                     f"{cell(abl['nonarithmetic_cumulative']['submitted']['tau_with_criterion'][col])} | "
                     f"{cell(sweep['k_crit_with_criterion'][col])} |")

    lines += ["", "Mean accuracy by K (blocks A and B), and tau of accuracy at that K with each BFCL column:", "",
              "| K | mean accuracy | " + " | ".join(MAIN_Y) + " |", "|---|---|" + "---|" * len(MAIN_Y)]
    for k in sweep["k_levels"]:
        taus = " | ".join(f3(sweep["accuracy_at_k_with_criterion"][str(k)][col]["tau"]) for col in MAIN_Y)
        lines.append(f"| {k} | {f3(sweep['mean_accuracy_by_k'][str(k)])} | {taus} |")

    lines += ["", "## Per model (strict rule in every block, G included)", "",
              "| Model | kind | family | A | C | D | E | F | G | H | I | B | K_crit | R2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, entry in result["per_model"].items():
        acc = entry["accuracy_strict"]
        fit = sweep["per_model"][name]
        lines.append(f"| {name} | {entry['kind']} | {entry['family']} | " +
                     " | ".join(f3(acc[b]) for b in "ACDEFGHIB") +
                     f" | {f3(fit.get('K_crit'))} | {f3(fit.get('R2'))} |")
    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_drop_set(text: str) -> tuple[str, list[str]]:
    name, _, members = text.partition("=")
    if not name or not members:
        raise argparse.ArgumentTypeError("use NAME=model1,model2")
    return name, [m.strip() for m in members.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", type=Path, required=True, help="scored_items.csv from score_bank.py")
    parser.add_argument("--criterion", type=Path, default=DRAFT / "bfcl_snapshot" / "bfcl_criterion_19.csv")
    parser.add_argument("--models", type=Path, default=DRAFT / "model_metadata" / "model_freeze_19.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-models", nargs="+", default=[], metavar="MODEL",
                        help="leave these models out of every analysis")
    parser.add_argument("--drop-set", action="append", default=[], type=parse_drop_set, metavar="NAME=M1,M2",
                        help="sensitivity analysis without these models; repeat for several sets")
    parser.add_argument("--g-rule", choices=sorted(G_RULE_COLUMN), required=True,
                        help="integer rule for block G (paraphrase stability); no default, the choice is open")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="run although models or items are missing; the output is marked INCOMPLETE")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    models, alias = load_models(args.criterion, args.models)
    scored, synthetic = load_scored(args.scored, alias)
    excluded = resolve(args.exclude_models, alias)
    drop_sets = {name: resolve(members, alias) for name, members in args.drop_set}
    for name in excluded:
        models.pop(name)
        scored.pop(name, None)

    incomplete: dict[str, Any] = {}
    absent = [name for name in models if name not in scored]
    if absent:
        incomplete["models_without_output"] = absent
    missing = {name: sum(row["status"] == "missing" for row in rows) for name, rows in scored.items()}
    if any(missing.values()):
        incomplete["missing_items_per_model"] = {name: count for name, count in missing.items() if count}
    if incomplete and not args.allow_incomplete:
        sys.exit(f"incomplete data: {json.dumps(incomplete, ensure_ascii=False)}\n"
                 "Finish the runs, or pass --allow-incomplete to analyse what is there (output is marked).")

    cfg = Config(args.n_boot, args.n_perm, args.seed)
    an = Analysis(models, scored, cfg)
    x = an.block("A")
    z1 = an.single_turn()

    result: dict[str, Any] = {"meta": {
        "synthetic": synthetic, "incomplete": incomplete, "n_models": len(an.names), "models": an.names,
        "excluded_models": excluded, "n_boot": args.n_boot, "n_perm": args.n_perm, "seed": args.seed,
        "criterion_snapshot": MAIN_SNAPSHOT, "g_rule": args.g_rule,
        "x": {**summarize(x), "n_at_ceiling": sum(v == 1.0 for v in x.values()),
              "n_distinct_values": len({v for v in x.values() if v is not None})},
        "methods": {
            "x": "block A accuracy (K=3,5,7; 60 items), correct_strict = scoring scheme one",
            "tau": "Kendall tau-b",
            "ci95": "percentile bootstrap over models",
            "ci95_family": "percentile bootstrap over model families",
            "perm_p": f"two-sided label permutation; exact for n <= {EXACT_PERM_MAX_N}, else Monte Carlo (count+1)/(n_perm+1)",
            "partial": "Kendall tau-b on rank residuals (method of code/consolidate_n28.py partial_kendall; "
                       "not Kendall's partial tau formula); two controls: ranks regressed on both control ranks; "
                       "residuals equal in exact arithmetic are ties (the submitted function orders them by a "
                       "1e-14 artefact; its value is kept as tau_submitted_code)",
            "single_turn_control": "mean of BFCL Non-Live and Live, same snapshot",
            "param_control": "log10 parameter count; only models with a count (rank-based, so log changes nothing)",
            "paraphrase": f"block G accuracy per template under --g-rule {args.g_rule} "
                          f"(column {G_RULE_COLUMN[args.g_rule]}); pairwise tau-b between templates",
            "k_sweep_fit": "a / (1 + exp(alpha * (K - K_crit))) + offset; p0 = [1, 0.5, 10, 0]; "
                           "bounds [0, 0.01, 0.1, -0.5] to [2, 5, 200, 0.5]; R2 = 1 - SSres / (SStot + 1e-12)",
        }}}
    strict = {b: an.block(b) for b in "ABCDEFGHI"}
    submitted = {b: an.block(b, "correct_submitted") for b in "ABCDEFGHI"}
    result["per_model"] = {name: {
        "kind": models[name]["kind"], "family": models[name]["family"], "params": models[name]["params"],
        "params_approximate": models[name]["params_approximate"],
        "single_turn": z1[name], "bfcl": {col: models[name]["bfcl"][MAIN_SNAPSHOT][col] for col in MAIN_Y},
        "accuracy_strict": {b: strict[b][name] for b in strict},
        "accuracy_submitted": {b: submitted[b][name] for b in submitted},
    } for name in an.names}

    result["association"] = an.association(x)
    result["strata"] = {kind: an.association(x, [n for n in an.names if models[n]["kind"] == kind])
                        for kind in ("open", "api")}
    result["leave_one_family_out"] = {col: {"tau": an.leave_one_family_out(x, an.bfcl(col)),
                                            "partial_single_turn": an.leave_one_family_out(x, an.bfcl(col), [z1])}
                                      for col in MAIN_Y}
    result["snapshots"] = {snapshot: an.association(x, snapshot=snapshot) for snapshot in SNAPSHOTS if snapshot != MAIN_SNAPSHOT}
    result["drop_sets"] = {name: {"dropped": members,
                                  "association": an.association(x, [n for n in an.names if n not in members])}
                           for name, members in drop_sets.items()}
    x_submitted = an.block("A", "correct_submitted")
    result["x_submitted_rule"] = {col: an.report(x_submitted, an.bfcl(col)) for col in MAIN_Y}
    result["ablations"] = ablations(an, x, args.g_rule)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "analysis_summary.md").write_text(markdown(result), encoding="utf-8")
    print(f"models: {len(an.names)}   written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
