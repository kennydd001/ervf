from __future__ import annotations

import json

from moe_lab.reporting import ROOT


RESULT = ROOT / "reports/craft_moe/cache_span.json"


def main() -> None:
    if not RESULT.exists():
        raise FileNotFoundError(RESULT)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if not result.get("screen_positive", False):
        print("status=not_opened")
        print("reason=preregistered layer26 cache-span oracle gate failed")
        return
    raise RuntimeError(
        "positive-screen training requires a separate preregistration before implementation"
    )


if __name__ == "__main__":
    main()
