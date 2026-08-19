from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_EXPERIMENT_FIELDS = {
    "id", "title", "legacy_branches", "legacy_artifacts",
    "legacy_result", "legacy_provenance", "transfer_class",
    "migration_decision", "wave", "dependencies", "lightning_test",
    "success_gate", "notes",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_registry(path: Path) -> dict[str, Any]:
    value = load(path)
    if value.get("storage") != "sharded":
        return value
    experiments = []
    for shard in value.get("shards") or []:
        shard_path = path.parent / shard["path"]
        shard_value = load(shard_path)
        rows = shard_value.get("experiments") or []
        if len(rows) != int(shard.get("count", -1)):
            raise ValueError(f"Registry shard count mismatch: {shard_path}")
        experiments.extend(rows)
    return {**value, "experiments": experiments}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ledger = load(args.ledger)
    registry = load_registry(args.registry)
    branches = ledger.get("branches") or []
    experiments = registry.get("experiments") or []
    errors: list[str] = []
    warnings: list[str] = []

    branch_names = [str(row.get("branch")) for row in branches]
    branch_set = set(branch_names)
    if ledger.get("branch_count") != len(branches):
        errors.append("branch_count does not match branches length")
    if len(branch_set) != len(branch_names):
        errors.append("duplicate branch names in ledger")
    if len(branches) != 42:
        errors.append(f"expected frozen 42-branch audit, got {len(branches)}")
    for row in branches:
        branch = row.get("branch")
        head = row.get("head")
        if not branch:
            errors.append("branch ledger row without branch name")
        if not isinstance(head, str) or not SHA_RE.fullmatch(head):
            errors.append(f"invalid immutable head for {branch}: {head}")
        if not row.get("migration_action"):
            errors.append(f"missing migration_action for {branch}")

    ids = [str(row.get("id")) for row in experiments]
    id_set = set(ids)
    if registry.get("experiment_count") != len(experiments):
        errors.append("experiment_count does not match experiments length")
    if len(id_set) != len(ids):
        errors.append("duplicate experiment IDs")

    allowed_decisions = set(registry.get("decision_values") or [])
    referenced_branches: set[str] = set()
    for row in experiments:
        exp_id = row.get("id", "<missing>")
        missing = REQUIRED_EXPERIMENT_FIELDS - set(row)
        if missing:
            errors.append(f"{exp_id}: missing fields {sorted(missing)}")
        try:
            int(row.get("wave"))
        except (TypeError, ValueError):
            errors.append(f"{exp_id}: wave is not an integer")
        decision = row.get("migration_decision")
        if allowed_decisions and decision not in allowed_decisions:
            errors.append(f"{exp_id}: unknown decision {decision}")
        for branch in row.get("legacy_branches") or []:
            referenced_branches.add(branch)
            if branch not in branch_set:
                errors.append(f"{exp_id}: unknown legacy branch {branch}")
        for dependency in row.get("dependencies") or []:
            if dependency not in id_set:
                errors.append(f"{exp_id}: unknown dependency {dependency}")
        if decision not in {"ARCHIVE"}:
            if not str(row.get("lightning_test") or "").strip():
                errors.append(f"{exp_id}: missing Lightning test")
            if not str(row.get("success_gate") or "").strip():
                errors.append(f"{exp_id}: missing success gate")

    uncovered = sorted(branch_set - referenced_branches)
    if uncovered:
        errors.append(f"branches not linked to an experiment: {uncovered}")
    duplicate_heads: dict[str, list[str]] = {}
    for row in branches:
        duplicate_heads.setdefault(row["head"], []).append(row["branch"])
    duplicate_heads = {
        sha: names for sha, names in duplicate_heads.items() if len(names) > 1
    }
    if duplicate_heads:
        warnings.append(
            "Some branch names intentionally share a head: "
            + json.dumps(duplicate_heads, sort_keys=True)
        )

    result = {
        "kind": "s100_lightning_registry_validation",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "branch_count": len(branches),
        "covered_branch_count": len(referenced_branches & branch_set),
        "experiment_count": len(experiments),
        "dependency_count": sum(
            len(row.get("dependencies") or []) for row in experiments
        ),
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
