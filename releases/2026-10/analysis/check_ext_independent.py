#!/usr/bin/env python
"""Independent recomputation of the results of blocks K and L (18 models).

Written from a prose description of the scoring and statistics rules. It shares no
code with analysis/score_ext.py, analysis/analyze_ext.py or
analysis/score_bank.py, and those files were not opened while writing it.

It reads the item bank and the raw response files, rescores every item, recomputes
the per-model cells, pooled counts, means, paired differences and bootstrap
intervals, and compares them with data/scored/ext/scored_items.csv,
analysis/results/ext/accuracy_by_model.csv and analysis/results/ext/analysis.json.
It writes nothing; everything is printed.

Run from the package root: python analysis/check_ext_independent.py
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = ROOT / "data" / "item_bank" / "ext_bank.jsonl"
RES = ROOT / "data" / "responses" / "ext"
SCORED_CSV = ROOT / "data" / "scored" / "ext" / "scored_items.csv"
ACC_CSV = ROOT / "analysis" / "results" / "ext" / "accuracy_by_model.csv"
ANALYSIS_JSON = ROOT / "analysis" / "results" / "ext" / "analysis.json"
TOL = 1e-9
N_BOOT = 10000
SEED = 42

# ----------------------------------------------------------------------------
# Scoring rules
# ----------------------------------------------------------------------------
ENTITY_ID = re.compile(r"\b[EUV]\d+\b")
INTEGER = re.compile(r"-?\d+")
# A number written with thousands separators: 1-3 digits, then one or more ",ddd" groups.
THOUSANDS = re.compile(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)")

K_ERROR_ORDER = [
    ("final_value", "final_state"),
    ("next_checkpoint_value", "next_checkpoint"),
    ("previous_checkpoint_value", "previous_checkpoint"),
    ("initial_value", "initial_state"),
]


def k_integers(text, entity_fill=" ", separators="grouped"):
    t = ENTITY_ID.sub(entity_fill, text)
    if separators == "grouped":
        t = THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), t)
    elif separators == "any_comma_between_digits":
        t = re.sub(r"(?<=\d),(?=\d)", "", t)
    elif separators != "none":
        raise ValueError(separators)
    return [int(s) for s in INTEGER.findall(t)]


def l_integers(text, entity_fill=" "):
    return [int(s) for s in INTEGER.findall(ENTITY_ID.sub(entity_fill, text))]


def score_k(item, text, **kw):
    ints = k_integers(text, **kw)
    compliant = len(ints) == 1
    correct = compliant and ints[0] == item["gold"]
    if correct:
        etype = "correct"
    elif not compliant:
        etype = "noncompliant"
    else:
        etype = "other"
        for field, name in K_ERROR_ORDER:
            if item[field] is not None and ints[0] == item[field]:
                etype = name
                break
    return len(ints), int(compliant), int(correct), etype


def score_l(item, text, **kw):
    ints = l_integers(text, **kw)
    return len(ints), int(len(ints) == item["target_count"]), int(ints == item["gold"])


# ----------------------------------------------------------------------------
# Comparison bookkeeping
# ----------------------------------------------------------------------------
compared = Counter()
mismatches = []


def same(a, b):
    num = lambda x: isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
    if num(a) and num(b):
        return abs(float(a) - float(b)) < TOL
    return a == b


def check(category, name, theirs, mine):
    compared[category] += 1
    if not same(theirs, mine):
        mismatches.append((category, name, theirs, mine))


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


# ----------------------------------------------------------------------------
# 1. Bank
# ----------------------------------------------------------------------------
bank_rows = [json.loads(line) for line in BANK_PATH.open()]
bank = {r["item_id"]: r for r in bank_rows}
assert len(bank_rows) == 180 and len(bank) == 180, "bank must have 180 unique item ids"
for it in bank_rows:
    if it["block"] == "K":
        it["pos_flag"] = int(math.ceil(it["checkpoint_slot"] / 3) == it["query_after"])
        it["cell"] = f"K_q{it['query_after']}"
    else:
        it["cell"] = f"L_M{it['target_count']}_{it['condition']}"
CELLS = ["K_q1", "K_q2", "K_q3",
         "L_M1_tracking", "L_M1_readout", "L_M2_tracking", "L_M2_readout",
         "L_M4_tracking", "L_M4_readout"]
cell_sizes = Counter(it["cell"] for it in bank_rows)
print("== Bank ==")
print("items:", len(bank_rows), "| cell sizes:", dict(cell_sizes))
print("K items with position flag 1 / 0:",
      sum(it["pos_flag"] for it in bank_rows if it["block"] == "K"),
      "/", sum(1 - it["pos_flag"] for it in bank_rows if it["block"] == "K"))
print("L items where target_count != len(gold):",
      sum(1 for it in bank_rows if it["block"] == "L" and it["target_count"] != len(it["gold"])))

# ----------------------------------------------------------------------------
# 2. Response files
# ----------------------------------------------------------------------------
files = {}
for p in sorted((RES / "local").glob("*.jsonl")):
    files[p.name[: -len(".jsonl")]] = ("open", p)
for p in sorted((RES / "api").glob("*.jsonl")):
    if p.name.endswith(".runs.jsonl") or p.name.endswith(".timing.jsonl"):
        continue
    files[p.name[: -len(".jsonl")]] = ("api", p)
models = sorted(files)
print("\n== Response files ==")
print("models found:", len(models), "| open:", sum(1 for v in files.values() if v[0] == "open"),
      "| api:", sum(1 for v in files.values() if v[0] == "api"))
assert len(models) == 18, "expected 18 models"

# ----------------------------------------------------------------------------
# 3. File checks and final-record selection
# ----------------------------------------------------------------------------
print("\n== File checks ==")
final = {}
file_problems = []
n_records_total = 0
n_prompt_compared = 0
bank_bytes = BANK_PATH.read_bytes()
for m in models:
    kind, path = files[m]
    recs = [json.loads(line) for line in path.open()]
    n_records_total += len(recs)
    by_item = defaultdict(list)
    for r in recs:
        by_item[r["item_id"]].append(r)
    extra_ids = sorted(set(by_item) - set(bank))
    fin = {}
    for iid, rs in by_item.items():
        good = [r for r in rs if r["error"] is None]
        if good and iid in bank:
            fin[iid] = good[-1]
    missing = sorted(set(bank) - set(fin))
    n_err = sum(1 for r in recs if r["error"] is not None)
    n_multi = sum(1 for rs in by_item.values() if len(rs) > 1)
    wrong_label = sum(1 for r in recs if r.get("label") != m)
    prompt_diff = [iid for iid, r in fin.items() if r["prompt"] != bank[iid]["prompt"]]
    cap_diff = [iid for iid, r in fin.items() if r["max_tokens"] != bank[iid]["max_tokens"]]
    gold_diff = [iid for iid, r in fin.items() if r["gold"] != bank[iid]["gold"]]
    n_prompt_compared += len(fin)
    line = (f"{m}: records={len(recs)} final={len(fin)} missing={len(missing)} extra_ids={len(extra_ids)} "
            f"error_records={n_err} items_with_several_records={n_multi} wrong_label={wrong_label} "
            f"prompt_diff={len(prompt_diff)} max_tokens_diff={len(cap_diff)} gold_diff={len(gold_diff)}")
    if kind == "open":
        copy = ROOT / "data" / "run_records" / "ext" / "local" / m / "ext_bank.jsonl"
        identical = copy.read_bytes() == bank_bytes
        line += f" bank_copy_identical={identical}"
        if not identical:
            file_problems.append((m, "bank copy differs"))
    print(line)
    for label, lst in [("missing", missing), ("extra_ids", extra_ids), ("prompt_diff", prompt_diff),
                       ("max_tokens_diff", cap_diff), ("gold_diff", gold_diff)]:
        if lst:
            file_problems.append((m, label, lst))
    if wrong_label:
        file_problems.append((m, "wrong_label", wrong_label))
    final[m] = fin
print("total records:", n_records_total, "| prompts compared with bank:", n_prompt_compared)
print("file problems:", file_problems if file_problems else "none")
n_missing_trials = sum(180 - len(final[m]) for m in models)
print("missing trials:", n_missing_trials)
if n_missing_trials:
    sys.exit("Missing trials found; the description does not define the statistics for that case. Stopping.")

# ----------------------------------------------------------------------------
# 4. Scoring
# ----------------------------------------------------------------------------
scored = {}
for m in models:
    for it in bank_rows:
        rec = final[m][it["item_id"]]
        text = rec["raw_response"]
        row = {"truncated": int(rec["finish_reason"] == "length"),
               "finish_reason": rec["finish_reason"], "raw_response": text,
               "k_error_type": "", "pos_flag": ""}
        if it["block"] == "K":
            row["n_integers"], row["compliant"], row["correct"], row["k_error_type"] = score_k(it, text)
            row["pos_flag"] = it["pos_flag"]
        else:
            row["n_integers"], row["compliant"], row["correct"] = score_l(it, text)
        scored[(m, it["item_id"])] = row

# ----------------------------------------------------------------------------
# 5. Statistics
# ----------------------------------------------------------------------------
def per_model(select, field="correct"):
    """Per-model (sum of field, number of items) over bank items passing select."""
    sums, ns = [], []
    for m in models:
        vals = [scored[(m, it["item_id"])][field] for it in bank_rows if select(it)]
        sums.append(sum(vals))
        ns.append(len(vals))
    return np.array(sums), np.array(ns)


cell_correct, cell_n, cell_compliant, cell_trunc, acc = {}, {}, {}, {}, {}
for c in CELLS:
    sel = lambda it, c=c: it["cell"] == c
    cell_correct[c], cell_n[c] = per_model(sel)
    cell_compliant[c], _ = per_model(sel, "compliant")
    cell_trunc[c], _ = per_model(sel, "truncated")
    acc[c] = cell_correct[c] / cell_n[c]

agree_correct, agree_n = per_model(lambda it: it["block"] == "K" and it["pos_flag"] == 1)
disag_correct, disag_n = per_model(lambda it: it["block"] == "K" and it["pos_flag"] == 0)
acc_agree = agree_correct / agree_n
acc_disag = disag_correct / disag_n

rng = np.random.RandomState(SEED)
boot_idx = np.stack([rng.choice(18, size=18, replace=True) for _ in range(N_BOOT)])


def summary(v):
    boot_means = v[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean_over_models": float(np.mean(v)), "ci95": [float(lo), float(hi)],
            "min": float(np.min(v)), "max": float(np.max(v))}


def cell_block(c):
    d = summary(acc[c])
    d.update(correct=int(cell_correct[c].sum()), answered=int(cell_n[c].sum()),
             compliant=int(cell_compliant[c].sum()), truncated=int(cell_trunc[c].sum()))
    return d


response_types = {}
for q in (1, 2, 3):
    cnt = Counter(scored[(m, it["item_id"])]["k_error_type"]
                  for m in models for it in bank_rows if it["cell"] == f"K_q{q}")
    response_types[f"q{q}"] = dict(cnt)

mine = {
    "meta": {"n_models": len(models), "models": models, "missing_trials": n_missing_trials},
    "K": {
        "accuracy_by_query_position": {f"q{q}": cell_block(f"K_q{q}") for q in (1, 2, 3)},
        "paired_differences": {
            "q2_minus_q1": summary(acc["K_q2"] - acc["K_q1"]),
            "q3_minus_q2": summary(acc["K_q3"] - acc["K_q2"]),
            "q3_minus_q1": summary(acc["K_q3"] - acc["K_q1"]),
        },
        "response_types": response_types,
        "by_position_guess": {
            "agrees": dict(summary(acc_agree), correct=int(agree_correct.sum()), answered=int(agree_n.sum())),
            "disagrees": dict(summary(acc_disag), correct=int(disag_correct.sum()), answered=int(disag_n.sum())),
            "agrees_minus_disagrees": summary(acc_agree - acc_disag),
        },
    },
    "L": {
        "accuracy": {f"M{k}_{cond}": cell_block(f"L_M{k}_{cond}")
                     for k in (1, 2, 4) for cond in ("tracking", "readout")},
        "readout_minus_tracking": {f"M{k}": summary(acc[f"L_M{k}_readout"] - acc[f"L_M{k}_tracking"])
                                   for k in (1, 2, 4)},
        "tracking_M4_minus_M1": summary(acc["L_M4_tracking"] - acc["L_M1_tracking"]),
        "readout_M4_minus_M1": summary(acc["L_M4_readout"] - acc["L_M1_readout"]),
    },
}

print("\n== Recomputed per-model correct counts (out of 20 per cell) ==")
print("model," + ",".join(CELLS))
for i, m in enumerate(models):
    print(m + "," + ",".join(str(int(cell_correct[c][i])) for c in CELLS))

print("\n== Recomputed statistics (pooled correct | mean over models | 95% interval) ==")
for path, d in [(f"K {q}", mine["K"]["accuracy_by_query_position"][q]) for q in ("q1", "q2", "q3")] + \
               [(f"L {c}", d) for c, d in mine["L"]["accuracy"].items()]:
    print(f"{path}: {d['correct']}/{d['answered']} | {d['mean_over_models']:.6f} | "
          f"[{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}]")
diffs = [("K " + k, v) for k, v in mine["K"]["paired_differences"].items()] + \
        [("L readout_minus_tracking " + k, v) for k, v in mine["L"]["readout_minus_tracking"].items()] + \
        [("L tracking_M4_minus_M1", mine["L"]["tracking_M4_minus_M1"]),
         ("L readout_M4_minus_M1", mine["L"]["readout_M4_minus_M1"]),
         ("K position agrees", mine["K"]["by_position_guess"]["agrees"]),
         ("K position disagrees", mine["K"]["by_position_guess"]["disagrees"]),
         ("K position agrees_minus_disagrees", mine["K"]["by_position_guess"]["agrees_minus_disagrees"])]
for name, d in diffs:
    print(f"{name}: {d['mean_over_models']:.6f} | [{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}]")
print("K response types:", json.dumps(response_types, sort_keys=True))
print("python", sys.version.split()[0], "| numpy", np.__version__)

# ----------------------------------------------------------------------------
# 6. Compare with scored_items.csv
# ----------------------------------------------------------------------------
with SCORED_CSV.open(newline="") as f:
    their_rows = list(csv.DictReader(f))
their_keys = [(r["model"], r["item_id"]) for r in their_rows]
print("\n== scored_items.csv ==")
print("rows:", len(their_rows), "| unique (model, item_id):", len(set(their_keys)),
      "| keys only in theirs:", len(set(their_keys) - set(scored)),
      "| keys only in mine:", len(set(scored) - set(their_keys)))
for r in their_rows:
    key = (r["model"], r["item_id"])
    if key not in scored:
        continue
    s = scored[key]
    tag = f"{key[0]} / {key[1]}"
    check("item.correct", tag, int(r["correct"]), s["correct"])
    check("item.compliant", tag, int(r["compliant"]), s["compliant"])
    check("item.truncated", tag, int(r["truncated"]), s["truncated"])
    check("item.n_integers", tag, int(r["n_integers"]), s["n_integers"])
    check("item.raw_response", tag, r["raw_response"], s["raw_response"])
    check("item.finish_reason", tag, r["finish_reason"], s["finish_reason"])
    if r["block"] == "K":
        check("item.k_error_type", tag, r["k_error_type"], s["k_error_type"])
        check("item.position_guess_correct", tag, int(r["position_guess_correct"]), s["pos_flag"])
    else:
        check("item.k_error_type(L rows blank)", tag, r["k_error_type"], "")
        check("item.position_guess_correct(L rows blank)", tag, r["position_guess_correct"], "")

# ----------------------------------------------------------------------------
# 7. Compare with accuracy_by_model.csv
# ----------------------------------------------------------------------------
with ACC_CSV.open(newline="") as f:
    acc_rows = {r["model"]: r for r in csv.DictReader(f)}
print("\n== accuracy_by_model.csv ==")
print("rows:", len(acc_rows), "| model set equal to mine:", set(acc_rows) == set(models))
k_cells = [c for c in CELLS if c.startswith("K")]
l_cells = [c for c in CELLS if c.startswith("L")]
for i, m in enumerate(models):
    r = acc_rows[m]
    for c in CELLS:
        check("model.cell_count", f"{m} / {c}", r[c], f"{int(cell_correct[c][i])}/{int(cell_n[c][i])}")
    check("model.K_compliant", m, int(r["K_compliant"]), int(sum(cell_compliant[c][i] for c in k_cells)))
    check("model.K_truncated", m, int(r["K_truncated"]), int(sum(cell_trunc[c][i] for c in k_cells)))
    check("model.L_compliant", m, int(r["L_compliant"]), int(sum(cell_compliant[c][i] for c in l_cells)))
    check("model.L_truncated", m, int(r["L_truncated"]), int(sum(cell_trunc[c][i] for c in l_cells)))
    check("model.answered", m, int(r["answered"]), len(final[m]))

# ----------------------------------------------------------------------------
# 8. Compare with analysis.json
# ----------------------------------------------------------------------------
theirs = json.loads(ANALYSIS_JSON.read_text())
ft, fm = flatten(theirs), flatten(mine)
NOT_COMPARED_PREFIXES = ("meta.status", "meta.bootstrap.", "meta.versions.", "K.by_position_guess.definition")


def category(key):
    if key.startswith("meta."):
        return "analysis.meta"
    if ".response_types." in key:
        return "analysis.pooled_count(response_types)"
    if key.endswith(".mean_over_models"):
        return "analysis.mean"
    if ".ci95[" in key:
        return "analysis.interval_bound"
    if key.endswith(".min") or key.endswith(".max"):
        return "analysis.min_max"
    return "analysis.pooled_count"


skipped = [k for k in ft if k.startswith(NOT_COMPARED_PREFIXES)]
only_theirs = [k for k in ft if k not in fm and k not in skipped]
only_mine = [k for k in fm if k not in ft]
for k in ft:
    if k in fm:
        check(category(k), k, ft[k], fm[k])
print("\n== analysis.json ==")
print("leaves in theirs:", len(ft), "| leaves in mine:", len(fm))
print("descriptive leaves not compared:", skipped)
print("leaves only in theirs:", only_theirs)
print("leaves only in mine:", only_mine)
print("largest absolute difference over the float leaves compared:",
      max(abs(ft[k] - fm[k]) for k in ft if k in fm and isinstance(ft[k], float)))
for k in only_theirs:
    mismatches.append(("analysis.key_only_in_theirs", k, ft[k], None))
for k in only_mine:
    mismatches.append(("analysis.key_only_in_mine", k, None, fm[k]))

# ----------------------------------------------------------------------------
# 9. Result of the comparison
# ----------------------------------------------------------------------------
print("\n== Comparison summary ==")
for cat in sorted(compared):
    n_bad = sum(1 for x in mismatches if x[0] == cat)
    print(f"{cat}: compared={compared[cat]} mismatches={n_bad}")
print("total compared:", sum(compared.values()), "| total mismatches:", len(mismatches))
for cat, name, a, b in mismatches:
    print(f"MISMATCH [{cat}] {name}: theirs={a!r} mine={b!r}")

# ----------------------------------------------------------------------------
# 10. How much the ambiguous parts of the rule description matter in these files
# ----------------------------------------------------------------------------
print("\n== Sensitivity of my reading of the rules (counts over all 3,240 final responses) ==")
n_entity_present = n_fill_k = n_fill_l = 0
n_entity_changes_count = 0
n_grouped_pattern = n_any_comma = 0
n_sep_any_vs_grouped = n_sep_none_vs_grouped = 0
n_decimal_k = 0
for m in models:
    for it in bank_rows:
        text = final[m][it["item_id"]]["raw_response"]
        has_entity = bool(ENTITY_ID.search(text))
        n_entity_present += has_entity
        if has_entity and len(INTEGER.findall(text)) != len(INTEGER.findall(ENTITY_ID.sub(" ", text))):
            n_entity_changes_count += 1
        if it["block"] == "K":
            base = score_k(it, text)
            n_fill_k += score_k(it, text, entity_fill="") != base
            n_sep_any_vs_grouped += score_k(it, text, separators="any_comma_between_digits") != base
            n_sep_none_vs_grouped += score_k(it, text, separators="none") != base
            stripped = ENTITY_ID.sub(" ", text)
            n_grouped_pattern += bool(THOUSANDS.search(stripped))
            n_any_comma += bool(re.search(r"\d,\d", stripped))
            n_decimal_k += bool(re.search(r"\d\.\d", stripped))
        else:
            n_fill_l += score_l(it, text, entity_fill="") != score_l(it, text)
print("responses containing at least one entity identifier:", n_entity_present)
print("responses where removing entity identifiers changes the number of integers:", n_entity_changes_count)
print("K responses scored differently if identifiers are deleted instead of replaced by a space:", n_fill_k)
print("L responses scored differently if identifiers are deleted instead of replaced by a space:", n_fill_l)
print("K responses containing a grouped thousands pattern (d{1,3}(,ddd)+):", n_grouped_pattern)
print("K responses containing any digit,digit:", n_any_comma)
print("K responses scored differently if every comma between two digits is removed:", n_sep_any_vs_grouped)
print("K responses scored differently with no separator handling at all:", n_sep_none_vs_grouped)
print("K responses containing digit.digit (a decimal point splits into two integers under the rule):", n_decimal_k)

all_text = [(m, it, final[m][it["item_id"]]["raw_response"]) for m in models for it in bank_rows]
print("responses with a lowercase e/u/v identifier (not removed by the rule):",
      sum(bool(re.search(r"\b[euv]\d+\b", t)) for _, _, t in all_text))
print("responses with an E/U/V+digits sequence glued to another word character (not removed by the rule):",
      sum(len(re.findall(r"[EUV]\d+", t)) != len(ENTITY_ID.findall(t)) for _, _, t in all_text))
print("responses with a Unicode minus or dash directly before a digit (not read as a minus sign):",
      sum(bool(re.search(r"[−–—]\d", t)) for _, _, t in all_text))
print("responses with a non-ASCII digit:",
      sum(any(ch.isdigit() and not ch.isascii() for ch in t) for _, _, t in all_text))
print("responses in which a negative integer is parsed: K =",
      sum(any(v < 0 for v in k_integers(t)) for _, it, t in all_text if it["block"] == "K"),
      "| L =", sum(any(v < 0 for v in l_integers(t)) for _, it, t in all_text if it["block"] == "L"))

print("\n== Additional counts (facts only) ==")
for blk in ("K", "L"):
    rows = [(scored[(m, it["item_id"])], t) for m, it, t in all_text if it["block"] == blk]
    print(f"block {blk}: responses={len(rows)}"
          f" correct={sum(s['correct'] for s, _ in rows)}"
          f" truncated={sum(s['truncated'] for s, _ in rows)}"
          f" correct_and_truncated={sum(s['correct'] and s['truncated'] for s, _ in rows)}"
          f" with_entity_identifier={sum(bool(ENTITY_ID.search(t)) for _, t in rows)}"
          f" correct_with_entity_identifier={sum(s['correct'] and bool(ENTITY_ID.search(t)) for s, t in rows)}")
