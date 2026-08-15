from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moe_lab.craft_moe.novelty_audit import (  # noqa: E402
    build_matrix,
    render_prior_art,
    render_verdict,
)


OUTPUTS = {
    ROOT / "reports" / "craft_moe" / "novelty_matrix.json": "json",
    ROOT / "docs" / "CRAFT_MOE_PRIOR_ART.md": "prior_art",
    ROOT / "reports" / "craft_moe" / "NOVELTY_VERDICT.md": "verdict",
}


def expected_outputs() -> dict[Path, str]:
    matrix = build_matrix()
    return {
        path: (
            json.dumps(matrix, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            if kind == "json"
            else render_prior_art(matrix)
            if kind == "prior_art"
            else render_verdict(matrix)
        )
        for path, kind in OUTPUTS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic CRAFT-MoE novelty audit.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="compare the current outputs byte-for-byte with the deterministic render",
    )
    args = parser.parse_args()
    outputs = expected_outputs()
    if args.check_only:
        failures = []
        for path, expected in outputs.items():
            if not path.exists():
                failures.append(f"missing: {path.relative_to(ROOT)}")
            elif path.read_text(encoding="utf-8") != expected:
                failures.append(f"content mismatch: {path.relative_to(ROOT)}")
        if failures:
            raise SystemExit("\n".join(failures))
        print(f"novelty audit exact-control passed: {len(outputs)} outputs")
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
    matrix = build_matrix()
    print(
        json.dumps(
            {
                "claim_units": matrix["summary"]["claim_unit_count"],
                "mandatory_families": matrix["summary"]["mandatory_family_count"],
                "sources": matrix["summary"]["source_count"],
                "verdict": matrix["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

