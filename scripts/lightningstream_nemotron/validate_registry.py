"""Parse and summarise the LIGHTNINGSTREAM_NEMOTRON registry.

A registry that does not parse is not a registry, so this runs after every
append.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "reports" / "lightningstream_nemotron" / "EXPERIMENT_REGISTRY.yaml"


def main() -> int:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))

    print(f"registry   : {data['registry']}")
    print(f"phases     : {len(data['phases'])}")
    print(f"executions : {len(data['executions'])}")
    print()
    print(f"  {'phase':<40} {'outcome':<30} terminal state")
    print(f"  {'-' * 40} {'-' * 30} {'-' * 45}")
    for entry in data["executions"]:
        print(f"  {entry['phase']:<40} {str(entry['outcome']):<30} "
              f"{entry.get('terminal_state', '-')}")

    print()
    protected = {e["phase"]: e.get("protected_verdict") for e in data["executions"]}
    bad = [p for p, v in protected.items() if v != "PROTECTED_80B_INTACT"]
    print(f"protected verdicts all INTACT : {not bad}")
    if bad:
        print(f"  VIOLATIONS: {bad}")
        return 3

    verified = [(e["phase"], e.get("independent_verification_score"))
                for e in data["executions"] if e.get("independent_verification_score")]
    for phase, score in verified:
        print(f"independent verification      : {phase} {score}")

    print(f"forbidden hypotheses tracked  : {len(data['forbidden_hypotheses'])}")
    print(f"stop rules tracked            : {len(data['stop_rules'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
