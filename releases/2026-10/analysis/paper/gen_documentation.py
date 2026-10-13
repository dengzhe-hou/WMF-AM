"""Typeset frozen prompt, score disagreements and model provenance; no inference."""
import csv
import json
from pathlib import Path

from gen_tables import appendix_table, esc, read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANEL = ROOT / "results"


def main():
    bank = [json.loads(line) for line in (ROOT.parent / "data/item_bank/item_bank.jsonl").read_text().splitlines()]
    item = next(row for row in bank if row["block"] == "A")
    prompt = "\n\n".join(esc(part).replace("\n", r"\newline ") for part in item["prompt"].split("\n\n"))
    (HERE / "tables/prompt_example.tex").write_text(
        "\\begin{quote}\\small\n" + prompt + "\n\\end{quote}\nThe reference answer is " + str(item["gold"]) + ".\n")
    names = {"falcon3-1b-instruct": "Falcon3 1B", "falcon3-3b-instruct": "Falcon3 3B",
             "llama-3.2-3b-instruct": "Llama 3.2 3B", "command-r7b-12-2024": "Command R7B"}
    changes = read("EF_SCORING_DIFFERENCES_20261004.csv")
    appendix_table("scoring_differences.tex", "tab:scoring-differences",
                    "All seven scoring disagreements on the assignment items and the set and count items. "
                    "Item identifiers match those in the released item bank. [] denotes the empty set (no permissions). "
                    "Parser 1 accepts each response and Parser 2 rejects it. "
                    "All responses stopped normally. The redundant negative description of an empty set is "
                    "semantically compatible with the reference answer, despite failing Parser 2's format.",
                    ["Model and item", "Reference answer", "Complete response"],
                    [[esc(names[r["model"]]) + r"\newline " + esc(r["item_id"]), esc(r["gold"]), esc(r["raw_response"])] for r in changes],
                    spec=r"p{3.1cm}p{2.9cm}p{7.8cm}")
    inputs = read("BFCL_INPUTS_20261004.csv")
    rows = []
    for row in inputs:
        params = f"{int(row['params']) / 1e9:.3f}" if row["params"] else "Unavailable"
        rows.append([esc(row["model"]), params] + [f"{float(row[key]):.2f}" for key in
                    ("bfcl_overall_1216", "bfcl_multi_turn_1216", "bfcl_agentic_1216", "bfcl_single_turn_mean_1216")])
    appendix_table("criterion_inputs.tex", "tab:criterion-inputs",
                    "All external criterion inputs. BFCL scores are percentages from the official 16 December 2025 FC snapshot. "
                    "Single-turn is the mean of Non-Live and Live. Public parameter counts, shown in billions, come from "
                    "model metadata frozen on 2 October 2026. Public counts are available only for the ten local models and the "
                    "two Cohere models. Single-turn BFCL and parameter counts enter only the rank-adjusted analyses in the "
                    "supplementary material.",
                    ["Model", "Parameters (B)", "Overall", "Multi-Turn", "Agentic", "Single-turn"], rows)
    with (ROOT.parent / "external/model_metadata/model_freeze_19.csv").open() as handle:
        freeze = {r["model"].lower(): r for r in csv.DictReader(handle)}
    revisions, manifest = [], []
    for row in inputs:
        if row["kind"] != "open":
            continue
        frozen = freeze[row["run_label"]]
        sha = frozen["hf_sha"]
        repo = frozen["hf_repo"]
        repo_display = r"\path{" + repo + "}"
        if repo == "ibm-granite/granite-3.1-8b-instruct":
            repo_display = r"\shortstack[l]{\path{ibm-granite/}\\\path{granite-3.1-8b-instruct}}"
        revisions.append([repo_display,
                          r"\href{https://huggingface.co/" + repo + "/tree/" + sha + r"}{\texttt{" + sha[:12] + "}}"])
        manifest.append({"model": row["run_label"], "hf_repo": repo, "revision": sha, "params": int(row["params"])})
    appendix_table("local_revisions.tex", "tab:local-revisions",
                    "Pinned repositories for all ten local models. Revision prefixes link to the complete frozen commit. "
                    "The provenance file in the supplementary material retains full commit identifiers and exact parameter counts.",
                    ["Repository", "Revision"], revisions, wide=False, colsep=1,
                    spec=r"@{}>{\raggedright\arraybackslash}p{\dimexpr\columnwidth-65pt\relax}@{\hspace{8pt}}>{\raggedleft\arraybackslash}p{57pt}@{}")
    (HERE / "model_provenance.json").write_text(json.dumps({
        "local": manifest, "all_criterion_inputs": inputs,
        "parser_names": {"Parser 1": "correct_submitted", "Parser 2": "correct_strict (E/F only)"},
        "source_files": ["external/model_metadata/model_freeze_19.csv", "analysis/results/BFCL_INPUTS_20261004.csv"],
    }, indent=2) + "\n")
    assert len(manifest) == 10 and len(inputs) == 18 and len(changes) == 7


if __name__ == "__main__":
    main()
