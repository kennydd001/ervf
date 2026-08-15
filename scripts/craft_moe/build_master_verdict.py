from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moe_lab.craft_moe.master_verdict import (  # noqa: E402
    build_master_verdict,
    render_master_verdict,
)


JSON_PATH = ROOT / "reports" / "craft_moe" / "master_verdict.json"
REPORT_PATH = ROOT / "reports" / "craft_moe" / "CRAFT_MOE_MASTER_VERDICT.md"


def expected_outputs() -> dict[Path, str]:
    payload = build_master_verdict()
    return {
        JSON_PATH: json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        REPORT_PATH: render_master_verdict(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic CRAFT-MoE master verdict.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    outputs = expected_outputs()
    if args.check_only:
        failures = [
            f"missing or changed: {path.relative_to(ROOT)}"
            for path, expected in outputs.items()
            if not path.exists() or path.read_text(encoding="utf-8") != expected
        ]
        if failures:
            raise SystemExit("\n".join(failures))
        print("master verdict exact-control passed: 2 outputs")
        return

    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "append-only outputs already exist: "
            + ", ".join(str(path.relative_to(ROOT)) for path in existing)
        )
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    payload = build_master_verdict()
    print(
        json.dumps(
            {
                "project_status": payload["project_status"],
                "revolutionary_gates_satisfied": payload["revolutionary_v2_gate"]["satisfied_count"],
                "revolutionary_gates_total": payload["revolutionary_v2_gate"]["condition_count"],
                "verdict": payload["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
