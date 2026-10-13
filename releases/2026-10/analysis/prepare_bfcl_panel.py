#!/usr/bin/env python3
"""Join the completed 18-model scores to frozen BFCL inputs; no correlations.

Run from any directory: python analysis/prepare_bfcl_panel.py
The only omitted frozen model is MiniCPM3-4B.
E/F rules, primary criterion and inferential methods are not selected here.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/results"
EXTRA_PARAMS = {"Command A": 111_057_580_032, "Command R7B": 8_028_033_024}
DENOMINATORS = {"A": 60, "C": 90, "D": 100}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    criterion = read_csv(ROOT / "external/bfcl_snapshot/bfcl_criterion_19.csv")
    frozen = {r["model"].lower(): r for r in read_csv(ROOT / "external/model_metadata/model_freeze_19.csv")}
    expected = [r for r in criterion if r["model"].lower() != "minicpm3-4b"]
    aliases = {}
    for r in expected:
        for alias in (r["model"], r["model_id_sent_by_handler"].split("/")[-1]):
            aliases[alias.lower()] = r["model"]
    grouped = defaultdict(list)
    for r in read_csv(ROOT / "data/scored/scored_items.csv"):
        grouped[aliases[r["model"].lower()]].append(r)
    assert set(grouped) == {r["model"] for r in expected}, "Model panel mismatch"

    panel = []
    for criterion_row in expected:
        name = criterion_row["model"]
        rows = grouped[name]
        assert len(rows) == len({r["item_id"] for r in rows}) == 840, name
        assert all(r["status"] == "scored" and r["synthetic"] == "0" for r in rows), name
        model = frozen[name.lower()]
        params = int(model["params"]) if model["params"] else EXTRA_PARAMS.get(name)
        family = (criterion_row["model_id_sent_by_handler"] if model["kind"] == "open"
                  else model["openrouter_slug"]).split("/")[0].lower()
        labels = {r["model"] for r in rows}
        assert len(labels) == 1, name
        entry = {
            "model": name, "run_label": next(iter(labels)), "kind": model["kind"],
            "family": family, "params": params,
            "params_source": ("model_freeze_19.csv" if model["params"] else
                              "Hugging Face API 2026-10-02" if params else "unavailable"),
        }
        for block, denominator in DENOMINATORS.items():
            block_rows = [r for r in rows if r["block"] == block]
            assert len(block_rows) == denominator, (name, block)
            correct = sum(int(r["correct_strict"]) for r in block_rows)
            entry.update({
                f"{block}_n": denominator, f"{block}_correct": correct,
                f"{block}_accuracy": correct / denominator,
                f"{block}_one_integer": sum(r["n_integers"] == "1" for r in block_rows),
                f"{block}_bare": sum(r["format"] == "bare" for r in block_rows),
                f"{block}_truncated": sum(int(r["truncated"]) for r in block_rows),
            })
        for column in ("overall", "multi_turn", "agentic", "non_live", "live"):
            entry[f"bfcl_{column}_1216"] = float(criterion_row[f"{column}_1216"])
        entry["bfcl_single_turn_mean_1216"] = (entry["bfcl_non_live_1216"] + entry["bfcl_live_1216"]) / 2
        entry["bfcl_handler_id"] = criterion_row["model_id_sent_by_handler"]
        entry["bfcl_mode"] = criterion_row["mode"]
        entry["bfcl_source"] = criterion_row["src_overall_1216"]
        panel.append(entry)

    with (OUT / "BFCL_INPUTS_20261004.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(panel[0]))
        writer.writeheader()
        writer.writerows(panel)
    metadata = {
        "purpose": "Descriptive input alignment only; no association statistic or primary-column selection",
        "n_models": len(panel), "source_rows": sum(map(len, grouped.values())),
        "excluded_as_already_approved": ["MiniCPM3-4B"],
        "family_definition": "Existing analysis.py convention: HF organization for local; provider for API",
        "family_counts": dict(Counter(r["family"] for r in panel)),
        "kind_counts": dict(Counter(r["kind"] for r in panel)),
        "parameter_count_available": sum(r["params"] is not None for r in panel),
        "parameter_count_missing": [r["model"] for r in panel if r["params"] is None],
        "bfcl_snapshot": "2025-12-16; frozen source scores on a 0-100 scale",
        "probe_scale": "A/C/D accuracy 0-1; all denominators retained",
        "pending": ["E/F final scoring", "BFCL primary column or parallel reporting", "inference and strata specification"],
    }
    (OUT / "BFCL_INPUTS_20261004.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
