from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.p1_verify import build_verification, render_verification


JSON_OUTPUT = ROOT / "reports/rsiv_moe/p1_verification_v2.json"
MD_OUTPUT = ROOT / "reports/rsiv_moe/P1_VERIFICATION_V2.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify sealed RSIV-MoE P1 artifacts.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def encoded_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    args = parse_args()
    payload = build_verification(ROOT)
    json_text = encoded_json(payload)
    markdown = render_verification(payload)
    if args.check_only:
        if JSON_OUTPUT.read_text(encoding="utf-8") != json_text:
            raise RuntimeError("p1_verification.json is not byte-exact")
        if MD_OUTPUT.read_text(encoding="utf-8") != markdown:
            raise RuntimeError("P1_VERIFICATION.md is not byte-exact")
        print(
            json.dumps(
                {
                    "all_required_checks_pass": payload["all_required_checks_pass"],
                    "checks": payload["checks"],
                    "passed": payload["passed"],
                    "failed": payload["failed"],
                    "warnings": payload["warnings"],
                },
                sort_keys=True,
            )
        )
    else:
        for path in (JSON_OUTPUT, MD_OUTPUT):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite: {path}")
        JSON_OUTPUT.write_text(json_text, encoding="utf-8")
        MD_OUTPUT.write_text(markdown, encoding="utf-8")
        print(json.dumps(payload, indent=2))
    if not payload["all_required_checks_pass"]:
        raise SystemExit(1)
