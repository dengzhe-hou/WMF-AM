#!/usr/bin/env python3
"""Summarize local-run facts; never score responses or run analysis.

Run after jobs end and after analysis/score_bank.py has exported scored_items.csv:
    python localrun/report_runs.py
    python localrun/report_runs.py --out-dir /path/to/report
    python localrun/report_runs.py \\
        --run-dir llama-3.3-70b-instruct=results/open_retry_20261003/llama-3.3-70b-instruct \\
        --scored-dir results/open_retry_20261003/scored \\
        --out-dir results/open_retry_20261003

Defaults: results/open/<label>/ and results/open/scored/scored_items.csv.
--run-dir LABEL=PATH selects a different source directory for that model only;
it can be repeated. Select a new --out-dir to preserve an earlier report.
Writes RUN_REPORT.md and RUN_REPORT.json. Existing job directories alone trigger
read-only sacct queries. Full accounting is saved to <label>/logs/sacct.txt only
when sacct confirms a terminal state and an actual end time. Active or unknown
jobs are never marked complete. Synthetic inputs are visibly marked as tests.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
N_ITEMS = 840
BLOCKS = "ABCDEFGHI"
TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_raw(path: Path, label: str) -> tuple[dict, list[str], bool]:
    final, notes, synthetic = {}, [], False
    if not path.exists():
        return final, notes, synthetic
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if not line.endswith("\n"):
                notes.append("the last raw line is incomplete and is not counted as finished")
                break
            raise
        if row["label"] != label:
            raise ValueError(f"{path}: unexpected label {row['label']!r}")
        synthetic |= bool(row.get("synthetic"))
        if row["error"] is None:
            item_id = row["item_id"]
            if item_id in final:
                raise ValueError(f"{path}: duplicate successful item {item_id}")
            final[item_id] = row
    if len(final) > N_ITEMS:
        raise ValueError(f"{path}: more than {N_ITEMS} successful items")
    return final, notes, synthetic


def command(args: list[str]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode:
        return None, f"exit {result.returncode}: {result.stderr.strip()}"
    return result.stdout, None


def accounting(run_dir: Path, manifest: dict, state: dict) -> dict:
    local = manifest.get("local_run", {})
    job_id = local.get("slurm_job_id") or state.get("slurm_job_id")
    array_id = local.get("slurm_array") or state.get("slurm_array")
    control = run_dir / "logs" / "scontrol_job.txt"
    if control.exists() and (not job_id or not array_id):
        saved = control.read_text()
        found = re.search(r"\bJobId=(\d+(?:_\d+)?)\b", saved)
        if found and not job_id:
            job_id = found[1]
        array_job = re.search(r"\bArrayJobId=(\d+)\b", saved)
        array_task = re.search(r"\bArrayTaskId=(\d+)\b", saved)
        if array_job and array_task and not array_id:
            array_id = f"{array_job[1]}_{array_task[1]}"
    info = {"job_id": str(job_id) if job_id else None, "state": None,
            "ended": False, "sacct_line": None, "sacct_file": None}
    if not job_id:
        info["note"] = "no job id found; job end cannot be confirmed"
        return info
    # sacct prints array aliases even when queried with SLURM_JOB_ID.
    query_id = str(array_id) if re.fullmatch(r"\d+_\d+", str(array_id)) else str(job_id)
    output, error = command(["sacct", "-X", "-j", query_id, "-n", "-P",
                             "-o", "JobID,State,End"])
    if error:
        info["note"] = f"job state query failed: {error}"
        return info
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    row = next((row for row in rows if row[0] in {str(job_id), query_id}), None)
    if not row:
        info["note"] = "sacct has no state record for this job"
        return info
    info["state"] = row[1]
    info["end"] = row[2]
    info["ended"] = (row[1].split()[0].rstrip("+") in TERMINAL
                     and row[2] not in {"", "Unknown", "None"})
    if not info["ended"]:
        info["note"] = "job end not yet confirmed; final sacct not saved"
        return info
    output, error = command(["sacct", "-j", query_id, "-o",
                             "JobID,Elapsed,TotalCPU,MaxRSS,ReqMem,AllocCPUS", "-P"])
    if error:
        info["note"] = f"final sacct query failed: {error}"
        return info
    rows = list(csv.DictReader(io.StringIO(output), delimiter="|"))
    row = next((row for row in rows if row["JobID"] in {str(job_id), query_id}), None)
    if not row:
        info["note"] = "final sacct has no resource record for this job"
        return info
    path = run_dir / "logs" / "sacct.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(output, encoding="utf-8")
    info["sacct_file"] = str(path.resolve())
    info["sacct_line"] = "|".join(row[key] for key in
                                   ("JobID", "Elapsed", "TotalCPU", "MaxRSS", "ReqMem", "AllocCPUS"))
    return info


def duration(manifest: dict, state: dict) -> tuple[float | None, str | None]:
    for source, data in (("manifest", manifest), ("manifest_state", state)):
        if data.get("run_seconds") is not None:
            return float(data["run_seconds"]), f"{source}.run_seconds"
        start, end = data.get("run_started_utc"), data.get("run_finished_utc")
        if start and end:
            seconds = (datetime.fromisoformat(end.replace("Z", "+00:00"))
                       - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
            if seconds < 0:
                raise ValueError(f"{source}: run ends before it starts")
            return seconds, f"{source}: run_finished_utc - run_started_utc"
    return None, None


def fraction(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def summarize(label: str, results_dir: Path, scored: list[dict], run_dir: Path | None = None) -> dict:
    source_override = run_dir
    run_dir = run_dir if run_dir is not None else results_dir / label
    manifest = read_json(run_dir / "manifest" / f"{label}.json")
    state = read_json(run_dir / "logs" / f"manifest_state_{label}.json")
    smoke = read_json(run_dir / "manifest" / f"{label}.smoke_result.json")
    conditions = smoke.get("smoke_result", manifest.get("smoke_result", {}))
    final, notes, synthetic = read_raw(run_dir / "out" / f"{label}.jsonl", label)
    if source_override is not None:
        notes.append(f"source directory set by --run-dir: {run_dir.resolve()}")
    result = {"model": label, "finished": len(final), "missing": N_ITEMS - len(final),
              "status": "incomplete" if run_dir.exists() else "not_started",
              "smoke": {f"S{i}": conditions.get(f"S{i}", "unknown") for i in range(1, 7)},
              "compliance_ALL_INT": None, "blocks_correct_strict": dict.fromkeys(BLOCKS),
              "truncated": sum(row.get("finish_reason") == "length" for row in final.values()),
              "synthetic": synthetic, "notes": notes}
    result["run_seconds"], result["run_seconds_source"] = duration(manifest, state)
    result["accounting"] = accounting(run_dir, manifest, state) if run_dir.exists() else {}
    if result["accounting"].get("note"):
        notes.append(result["accounting"]["note"])
    if result["accounting"].get("ended") and result["accounting"].get("state") != "COMPLETED":
        notes.append(f"Slurm state {result['accounting']['state']}")
    if (run_dir / "REPORT.md").exists():
        notes.append(f"[original record of the failed or unfinished run]({(run_dir / 'REPORT.md').resolve()})")
    if scored:
        result["synthetic"] |= any(row.get("synthetic") == "1" for row in scored)
        done = [row for row in scored if row["status"] == "scored"]
        # Detect stale CSVs after a running model has produced more records.
        if (len(scored) != N_ITEMS or len({row["item_id"] for row in scored}) != N_ITEMS
                or {row["item_id"] for row in done} != set(final)):
            notes.append("the scored CSV does not match the items currently finished in raw, so scores are not reported; rerun score_bank.py")
        else:
            numeric = [row for row in done if row["answer_type"] == "int"]
            result["compliance_ALL_INT"] = fraction(
                sum(int(row["n_integers"]) == 1 for row in numeric), len(numeric))
            for block in BLOCKS:
                group = [row for row in done if row["block"] == block]
                result["blocks_correct_strict"][block] = fraction(
                    sum(int(row["correct_strict"]) for row in group), len(group))
    elif final:
        notes.append("no scored CSV yet")
    exception_path = run_dir / "manifest" / f"{label}.smoke_exception.json"
    exception = read_json(exception_path)
    s3_exception = (
        label in {"granite-4.0-350m", "llama-3.2-1b-instruct"}
        and exception.get("waived_conditions") == ["S3"]
        and exception.get("smoke_result") == result["smoke"]
        and result["smoke"] == {f"S{i}": "fail" if i == 3 else "pass" for i in range(1, 7)}
    )
    if s3_exception:
        notes.append(f"S3-only exception applied on 2026-10-03; the original S3=fail is retained. Exception record: {exception_path.resolve()}")
    if (len(final) == N_ITEMS and manifest.get("runner_exit_code") == 0
            and result["accounting"].get("ended")
            and result["accounting"].get("state") == "COMPLETED"):
        if all(value == "pass" for value in result["smoke"].values()):
            result["status"] = "complete"
        elif s3_exception:
            result["status"] = "complete_with_s3_exception"
    return result


def show_rate(value: dict | None) -> str:
    if not value or value["rate"] is None:
        return "not scored" if value is None else "no finished item (0/0)"
    return f"{value['rate']:.1%} ({value['numerator']}/{value['denominator']})"


def markdown(report: dict) -> str:
    generated_utc = datetime.fromisoformat(report["generated_utc"]).astimezone(ZoneInfo("UTC"))
    lines = ["# WMF-AM local model run facts", "", f"Generated (UTC): {generated_utc:%Y-%m-%d %H:%M:%S}", "",
             "840 items in total. finished counts raw records with error=null only; missing items do not enter the accuracy denominator.",
             "ALL_INT covers answer_type=int only; compliant means exactly one integer, other text allowed; block I is not included.",
             "A–I use correct_strict; E/F are still candidate scores; I is CoT and takes the last integer.",
             "Truncated counts cover all finished items. Duration refers to the full run only, excluding loading, smoke test, and shutdown.",
             "sacct column order: JobID, Elapsed, TotalCPU, MaxRSS, ReqMem, AllocCPUS; the full output includes job steps, see the link.", ""]
    if report["synthetic"]:
        lines += ["**SYNTHETIC TEST ONLY: this report contains synthetic test data and is not a real model result.**", ""]
    columns = ["Model", "Status", "finished/missing", "Smoke S1–S6", "ALL_INT compliant"]
    columns += list(BLOCKS) + ["Truncated", "Run seconds", "sacct"]
    lines += ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in report["models"]:
        smoke = ", ".join(f"{key}={value}" for key, value in row["smoke"].items())
        acct = row["accounting"]
        sacct = acct.get("sacct_line", "not available") or "not available"
        if acct.get("sacct_file"):
            sacct += f" [full output]({acct['sacct_file']})"
        cells = [row["model"], row["status"], f"{row['finished']}/{row['missing']}", smoke,
                 show_rate(row["compliance_ALL_INT"])]
        cells += [show_rate(row["blocks_correct_strict"][block]) for block in BLOCKS]
        cells += [str(row["truncated"]), str(row["run_seconds"]) if row["run_seconds"] is not None else "not available", sacct]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    notes = [f"- {row['model']}: {'; '.join(row['notes'])}" for row in report["models"] if row["notes"]]
    if notes:
        lines += [""] + notes
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "open")
    parser.add_argument("--scored-dir", type=Path, help="default: RESULTS_DIR/scored")
    parser.add_argument("--out-dir", type=Path, help="default: RESULTS_DIR")
    parser.add_argument("--run-dir", action="append", default=[], metavar="LABEL=PATH",
                        help="override one model's source directory; may be repeated")
    parser.add_argument("--models", type=Path, default=ROOT / "frozen" / "bundle" / "models_open.tsv")
    args = parser.parse_args()
    scored_dir, out_dir = args.scored_dir or args.results_dir / "scored", args.out_dir or args.results_dir
    with args.models.open(encoding="utf-8", newline="") as handle:
        labels = [row["label"] for row in csv.DictReader(handle, delimiter="\t") if row["label"] != "minicpm3-4b"]
    run_dirs = {}
    for value in args.run_dir:
        label, separator, path = value.partition("=")
        if not separator or not path or label not in labels:
            parser.error("--run-dir requires a listed model LABEL=PATH")
        run_dirs[label] = Path(path)
    scored = defaultdict(list)
    path = scored_dir / "scored_items.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                scored[row["model"]].append(row)
    models = [summarize(label, args.results_dir, scored[label], run_dirs.get(label)) for label in labels]
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "n_items": N_ITEMS,
              "synthetic": any(row["synthetic"] for row in models), "models": models}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RUN_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "RUN_REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {out_dir / 'RUN_REPORT.md'} and {out_dir / 'RUN_REPORT.json'}")


if __name__ == "__main__":
    main()
