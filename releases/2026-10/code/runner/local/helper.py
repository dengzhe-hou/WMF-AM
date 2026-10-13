#!/usr/bin/env python3
"""Small checks shared by the local-run scripts. Standard library only.

Subcommands
  lookup LABEL               print "repo<TAB>revision<TAB>trust_remote_code" from models_open.tsv
  present LABEL              offline: exit 0 if the pinned snapshot holds config, tokenizer and every weight shard
  complete LABEL             online (tree API, GET): exit 0 if every file the download selects is in the snapshot
  access LABEL...            HEAD request per repo (token from the HF token file, never printed);
                             prints the HTTP status; exit 0 if all are readable, 3 if any is not
  bank BANK_JSONL            exit 0 if the bank has 840 items and G matches option A
  table LABEL...             size table from the HF tree API (GET only), what is already present,
                             what would be downloaded; nothing is downloaded

Paths: models_open.tsv of the frozen bundle (env MODELS_TSV overrides; the jobs use the copy unpacked
from the bundle tarball); HF cache $HF_HUB_CACHE, default /path/to/hf_cache.
"""
from __future__ import annotations

import fnmatch
import http.client
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

MODELS_TSV = Path(os.environ.get("MODELS_TSV", "/path/to/repo/frozen/bundle/models_open.tsv"))
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/path/to/hf_cache"))
TOKEN_FILE = Path.home() / ".cache/huggingface/token"

# What vLLM and the step-C tokenizer read. Same patterns as the hf download command
# in download_models.sh. "original/*" (second copy of the Llama weights) is excluded.
INCLUDE = ["*.safetensors", "*.json", "*.jinja", "tokenizer.model", "merges.txt"]
EXCLUDE = ["original/*"]

N_BANK = 840
G_MAX_TOKENS = 16   # decided 2026-10-03: option A; minimal gets the final-number instruction


def models() -> dict[str, dict[str, str]]:
    rows = [line.rstrip("\n").split("\t") for line in MODELS_TSV.read_text().splitlines() if line.strip()]
    head = rows[0]
    return {r[head.index("label")]: dict(zip(head, r)) for r in rows[1:]}


def model(label: str) -> dict[str, str]:
    table = models()
    if label not in table:
        sys.exit(f"unknown label {label!r}; labels in {MODELS_TSV}: {', '.join(table)}")
    return table[label]


def snapshot_dir(m: dict[str, str]) -> Path:
    return HF_CACHE / ("models--" + m["hf_repo"].replace("/", "--")) / "snapshots" / m["revision"]


def missing_files(m: dict[str, str], expected: list[str] | None = None) -> list[str]:
    """Files missing from the pinned snapshot. Default (offline): config, tokenizer, every weight shard.
    With `expected` (from the tree API): exactly those files."""
    snap = snapshot_dir(m)
    if not snap.is_dir():
        return [f"{snap} (no snapshot at this revision)"]
    if expected is not None:
        return [f for f in expected if not (snap / f).exists()]
    need = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    index = snap / "model.safetensors.index.json"
    if index.exists():
        need += sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (snap / "model.safetensors").exists():
        need.append("model.safetensors")
    else:
        need.append("model.safetensors or model.safetensors.index.json")
    return [f for f in need if not (snap / f).exists()]   # exists() follows the symlink to the blob


def token() -> str:
    return TOKEN_FILE.read_text().strip()


def head_status(repo: str, rev: str) -> int:
    """HEAD on config.json at the pinned revision; no redirect is followed."""
    conn = http.client.HTTPSConnection("huggingface.co", timeout=30)
    conn.request("HEAD", f"/{repo}/resolve/{rev}/config.json", headers={"Authorization": f"Bearer {token()}"})
    status = conn.getresponse().status
    conn.close()
    return status


def tree(repo: str, rev: str) -> list[tuple[str, int]]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    files = []
    headers = {"Authorization": f"Bearer {token()}"}
    while url:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = json.loads(resp.read())
            if not page:
                break
            files.extend((d["path"], d["size"]) for d in page if d["type"] == "file")
            next_page = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
            url = next_page.group(1) if next_page else None
    return files


def selected(path: str) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in INCLUDE) and not any(fnmatch.fnmatch(path, p) for p in EXCLUDE)


def cmd_lookup(label: str) -> int:
    m = model(label)
    print(f"{m['hf_repo']}\t{m['revision']}\t{m['trust_remote_code']}")
    return 0


def cmd_present(label: str) -> int:
    miss = missing_files(model(label))
    if miss:
        print(f"{label}: not present at the pinned revision; missing: {', '.join(miss)}")
        return 1
    print(f"{label}: present at {snapshot_dir(model(label))}")
    return 0


def cmd_complete(label: str) -> int:
    m = model(label)
    expected = [p for p, _ in tree(m["hf_repo"], m["revision"]) if selected(p)]
    miss = missing_files(m, expected)
    if miss:
        print(f"{label}: download incomplete; missing: {', '.join(miss)}")
        return 1
    print(f"{label}: all {len(expected)} selected files present")
    return 0


def cmd_access(labels: list[str]) -> int:
    bad = []
    for label in labels:
        m = model(label)
        status = head_status(m["hf_repo"], m["revision"])
        ok = status in (200, 302, 307)
        print(f"{label}\t{m['hf_repo']}\tHTTP {status}\t{'access OK' if ok else 'NO ACCESS (gated request pending or refused)'}")
        if not ok:
            bad.append(label)
    return 3 if bad else 0


def cmd_bank(path: str) -> int:
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    g = [it for it in items if it.get("block") == "G"]
    g_tokens = sorted({it.get("max_tokens") for it in g})
    problems = []
    if len(items) != N_BANK:
        problems.append(f"{len(items)} items, expected {N_BANK}")
    if len(g) != 150 or g_tokens != [G_MAX_TOKENS]:
        problems.append(f"block G has {len(g)} items with max_tokens {g_tokens}, expected 150 items with [{G_MAX_TOKENS}] "
                        "(stale bank from before the 2026-10-03 G decision?)")
    minimal = [it for it in g if it.get("template") == "minimal"]
    suffix = "\n\nRespond with ONLY the final number."
    if len(minimal) != 30 or any(not it["prompt"].endswith(suffix) or not it.get("new_text") for it in minimal):
        problems.append("option A requires 30 minimal items with the declared final-number instruction")
    if problems:
        print(f"BANK CHECK FAILED for {path}: " + "; ".join(problems))
        return 1
    print(f"bank check OK: {path}: {len(items)} items, block G 150 items at max_tokens {G_MAX_TOKENS}")
    return 0


def cmd_table(labels: list[str]) -> int:
    print(f"{'label':26} {'repo':36} {'rev':8} {'files':>5} {'GB sel.':>8} {'GB repo':>8}  status")
    total_get = 0
    for label in labels:
        m = model(label)
        files = tree(m["hf_repo"], m["revision"])
        sel = [(p, s) for p, s in files if selected(p)]
        gb_sel = sum(s for _, s in sel) / 1e9
        gb_all = sum(s for _, s in files) / 1e9
        if not missing_files(m):
            status = "present, skipped"
        else:
            status = "to download"
            total_get += sum(s for _, s in sel)
        print(f"{label:26} {m['hf_repo']:36} {m['revision'][:8]} {len(sel):5d} {gb_sel:8.2f} {gb_all:8.2f}  {status}")
        dropped = [p for p, _ in files if not selected(p)]
        print(f"{'':26} not downloaded: {', '.join(dropped) if dropped else '-'}")
    print(f"total to download for these labels: {total_get / 1e9:.2f} GB")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "lookup":
        return cmd_lookup(args[0])
    if cmd == "present":
        return cmd_present(args[0])
    if cmd == "complete":
        return cmd_complete(args[0])
    if cmd == "access":
        return cmd_access(args)
    if cmd == "bank":
        return cmd_bank(args[0])
    if cmd == "table":
        return cmd_table(args)
    sys.exit(f"unknown subcommand {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main())
