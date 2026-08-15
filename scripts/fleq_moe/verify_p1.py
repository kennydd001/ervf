from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from moe_lab.fleq_moe.p1_verify import verify
from moe_lab.reporting import ROOT


OUTPUT = ROOT / "reports/fleq_moe/p1_verification.json"
REPORT = ROOT / "reports/fleq_moe/P1_VERIFICATION.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    payload = verify(recompute_anchors=True)
    if not payload["verification_pass"]:
        failed = [name for name, passed in payload["checks"].items() if not passed]
        raise AssertionError(f"P1 verification failed: {failed}")
    if args.check_only:
        stored = json.loads(OUTPUT.read_text(encoding="utf-8"))
        assert stored["result_sha256"] == payload["result_sha256"]
        assert stored["checks"] == payload["checks"]
        print(json.dumps({"check_only": True, "verification_pass": True, "checks": payload["checks_total"]}))
    else:
        if OUTPUT.exists() or REPORT.exists():
            raise FileExistsError("refusing to overwrite P1 verification")
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# FLEQ-MoE P1 — onafhankelijke verificatie", "",
            f"Uitkomst: **PASS ({payload['checks_passed']}/{payload['checks_total']})**.", "",
            "De verifier heeft de lockselectie, 32 rapporthashes, 32 artifacthashes, alle verbeteringsformules, codebereiken, bitbudgetten en de vooraf vastgelegde gate onafhankelijk gecontroleerd. Voor de eerste geselecteerde expert van beide lagen zijn bovendien alle held-out metrics van 2-bit GPTQ/GSQ en ternary RTN/GSQ opnieuw uit de opgeslagen gewichten berekend.", "",
            "De bewezen conclusie is begrensd: P1 is `smoke_negative`, P2 is niet geautoriseerd en er is geen Eureka-claim. Dit bewijst geen algemene onmogelijkheid van low-entropy MoE-quantisatie.", "",
            "## Controles", "",
        ]
        lines.extend(f"- `{name}`: `{passed}`" for name, passed in payload["checks"].items())
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"verification_pass": True, "checks": payload["checks_total"]}))
