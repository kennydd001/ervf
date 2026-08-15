from __future__ import annotations

import argparse
import json

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.p1b_verify import (
    build_p1b_verification,
    render_p1b_verification,
)


JSON_OUTPUT = ROOT / "reports/rsiv_moe/p1b_verification.json"
MD_OUTPUT = ROOT / "reports/rsiv_moe/P1B_VERIFICATION.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RSIV-MoE P1B artifacts.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = build_p1b_verification(ROOT)
    json_text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    markdown = render_p1b_verification(payload)
    if args.check_only:
        if JSON_OUTPUT.read_text(encoding="utf-8") != json_text:
            raise RuntimeError("p1b_verification.json is not byte-exact")
        if MD_OUTPUT.read_text(encoding="utf-8") != markdown:
            raise RuntimeError("P1B_VERIFICATION.md is not byte-exact")
        print(json.dumps({key: payload[key] for key in ("checks", "passed", "failed", "warnings", "all_required_checks_pass")}, sort_keys=True))
    else:
        for path in (JSON_OUTPUT, MD_OUTPUT):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite: {path}")
        JSON_OUTPUT.write_text(json_text, encoding="utf-8")
        MD_OUTPUT.write_text(markdown, encoding="utf-8")
        print(json.dumps(payload, indent=2))
    if not payload["all_required_checks_pass"]:
        raise SystemExit(1)

