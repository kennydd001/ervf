from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

RELEVANT_PREFIXES = ("reports/", "agents/", "pro_research/")
RELEVANT_ROOT_RE = re.compile(r"^(?:S100|PRO|RESULT|VERDICT|BASELINE|RESEARCH|EUREKA|PATH|POST_).*", re.I)
TEXT_SUFFIXES = {".json", ".md", ".yaml", ".yml", ".py", ".ps1", ".txt"}
MAX_SCAN_BYTES = 2 << 20

NANO_RE = re.compile(
    r"(?:NVIDIA-Nemotron-3-Nano|Nemotron[-_ ]?3[-_ ]?Nano|"
    r"models[\\/]+nemotron_3_5_lightning(?!_v35))", re.I,
)
LIGHTNING_RE = re.compile(
    r"(?:NVIDIA-Nemotron-3\.5-Lightning|Nemotron[-_ ]?3\.5[-_ ]?Lightning|"
    r"nemotron_3_5_lightning_v35)", re.I,
)
DEEPSEEK_RE = re.compile(r"DeepSeek[-_ ]?V2[-_ ]?Lite", re.I)
TECH_FAIL_RE = re.compile(r"(?:technical_failure|technical failure|instrumentation complete:\s*false)", re.I)
MODEL_DIR_RE = re.compile(
    r"(?:model_dir|checkpoint|model(?:_name|_id)?)[\"' :=]+([^\"'\r\n,}]+)", re.I,
)


def run(repo: Path, args: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    if proc.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def list_remote_branches(repo: Path) -> list[tuple[str, str]]:
    out = run(repo, [
        "for-each-ref", "--format=%(refname:short)\t%(objectname)",
        "refs/remotes/origin",
    ])
    rows = []
    for line in out.splitlines():
        if not line.strip(): continue
        ref, sha = line.split("\t", 1)
        if ref == "origin/HEAD": continue
        rows.append((ref.removeprefix("origin/"), sha))
    return sorted(rows)


def ls_tree(repo: Path, ref: str) -> list[dict[str, Any]]:
    out = run(repo, ["ls-tree", "-r", "-l", ref], timeout=300)
    rows = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) < 4: continue
        size = None if parts[3] == "-" else int(parts[3])
        rows.append({"mode": parts[0], "type": parts[1], "blob": parts[2], "bytes": size, "path": path})
    return rows


def relevant(path: str, size: int | None) -> bool:
    p = Path(path)
    if p.suffix.lower() not in TEXT_SUFFIXES: return False
    if size is not None and size > MAX_SCAN_BYTES: return False
    if path.startswith(RELEVANT_PREFIXES): return True
    return "/" not in path and bool(RELEVANT_ROOT_RE.match(path))


def show_blob(repo: Path, blob: str) -> str:
    return run(repo, ["cat-file", "-p", blob], timeout=60)


def parse_json_summary(text: str) -> dict[str, Any]:
    try: obj = json.loads(text)
    except json.JSONDecodeError: return {}
    if not isinstance(obj, dict): return {}
    return {
        "kind": obj.get("kind"), "status": obj.get("status"),
        "decision": obj.get("decision"),
        "s100_single_achieved": obj.get("s100_single_achieved"),
        "model_dir": ((obj.get("environment") or {}).get("model_dir")
                      if isinstance(obj.get("environment"), dict)
                      else obj.get("model_dir")),
    }


def classify_evidence(evidence: dict[str, Any]) -> str:
    if evidence["deepseek_hits"] and not (evidence["nano_hits"] or evidence["lightning_hits"]):
        return "different_model_deepseek"
    if evidence["nano_hits"] and evidence["lightning_hits"]: return "mixed_or_conflicting"
    if evidence["nano_hits"]: return "nano_evidence"
    if evidence["lightning_hits"]: return "lightning_claim_unverified"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not args.no_fetch:
        run(repo, ["fetch", "--all", "--prune"], timeout=600)

    static_path = repo / "pro_research" / "lightning_migration" / "BRANCH_LEDGER.json"
    static = {}
    if static_path.exists():
        data = json.loads(static_path.read_text(encoding="utf-8"))
        static = {x["branch"]: x for x in data.get("branches", [])}

    blob_cache: dict[str, tuple[str, dict[str, Any]]] = {}
    branches = []
    for branch, head in list_remote_branches(repo):
        files = ls_tree(repo, f"origin/{branch}")
        scan = [x for x in files if relevant(x["path"], x["bytes"])]
        evidence = {
            "nano_hits": 0, "lightning_hits": 0, "deepseek_hits": 0,
            "technical_failure_hits": 0, "model_references": [],
            "json_kinds": Counter(), "json_statuses": Counter(),
            "scanned_files": 0, "scanned_unique_blobs": 0,
            "relevant_paths": [x["path"] for x in scan],
        }
        for item in scan:
            blob = item["blob"]
            if blob in blob_cache:
                text, summary = blob_cache[blob]
            else:
                text = show_blob(repo, blob)
                summary = parse_json_summary(text) if item["path"].endswith(".json") else {}
                blob_cache[blob] = (text, summary)
                evidence["scanned_unique_blobs"] += 1
            evidence["scanned_files"] += 1
            evidence["nano_hits"] += len(NANO_RE.findall(text))
            evidence["lightning_hits"] += len(LIGHTNING_RE.findall(text))
            evidence["deepseek_hits"] += len(DEEPSEEK_RE.findall(text))
            evidence["technical_failure_hits"] += len(TECH_FAIL_RE.findall(text))
            evidence["model_references"].extend(MODEL_DIR_RE.findall(text)[:20])
            if summary.get("kind"): evidence["json_kinds"][summary["kind"]] += 1
            if summary.get("status"): evidence["json_statuses"][str(summary["status"])] += 1
            if summary.get("model_dir"): evidence["model_references"].append(str(summary["model_dir"]))

        evidence["model_references"] = sorted(set(evidence["model_references"]))[:200]
        evidence["json_kinds"] = dict(evidence["json_kinds"])
        evidence["json_statuses"] = dict(evidence["json_statuses"])
        evidence["automatic_provenance_class"] = classify_evidence(evidence)
        frozen = static.get(branch)
        branches.append({
            "branch": branch, "actual_head": head,
            "frozen_head": frozen.get("head") if frozen else None,
            "head_matches_frozen_audit": bool(frozen and frozen.get("head") == head),
            "frozen_migration_action": frozen.get("migration_action") if frozen else None,
            "frozen_rationale": frozen.get("rationale") if frozen else None,
            "tree_file_count": len(files), "evidence": evidence,
        })
        print(f"{branch}: {head[:8]} {evidence['automatic_provenance_class']} files={len(files)} relevant={len(scan)}", flush=True)

    result = {
        "kind": "s100_all_remote_branch_audit",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": str(repo), "branch_count": len(branches),
        "unique_scanned_blobs": len(blob_cache), "branches": branches,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ALL_REMOTE_BRANCH_AUDIT.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "ALL_REMOTE_BRANCH_AUDIT.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["branch", "head", "head_matches_frozen", "automatic_provenance",
                         "frozen_action", "tree_files", "relevant_files", "nano_hits",
                         "lightning_hits", "deepseek_hits", "technical_failure_hits"])
        for row in branches:
            ev = row["evidence"]
            writer.writerow([row["branch"], row["actual_head"], row["head_matches_frozen_audit"],
                             ev["automatic_provenance_class"], row["frozen_migration_action"],
                             row["tree_file_count"], ev["scanned_files"], ev["nano_hits"],
                             ev["lightning_hits"], ev["deepseek_hits"], ev["technical_failure_hits"]])

    print(json.dumps({"branch_count": len(branches), "unique_scanned_blobs": len(blob_cache),
                      "json": str(json_path), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
