from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shared import PRO, REPO, environment, load_json, result_path, utc_now, write_json

OUT = result_path("PV2_00_PROVENANCE.json")
V6 = PRO / "results" / "PRO_V6_FULL_STACK.json"


def main() -> int:
    payload = {
        "kind": "pv2_provenance_budget",
        "status": "started",
        "started_utc": utc_now(),
        "base_expected": "5c699300da2d10552f5037426c1607119b2239b4",
        "environment": environment((
            HERE / "PREREGISTRATION.md",
            HERE / "POST_V6_REGISTRY.yaml",
            REPO / "agents" / "PATH_TO_100_TOKS.md",
        )),
    }
    if not V6.exists():
        payload.update(status="technical_failure", error=f"missing {V6}")
        write_json(OUT, payload)
        print(json.dumps({"status": payload["status"], "output": str(OUT)}, indent=2))
        return 2
    v6 = load_json(V6)
    p50 = float(v6["summary"]["v6_p50_ms"])
    payload["v6"] = {
        "source": str(V6),
        "source_status": v6.get("status"),
        "p50_ms": p50,
        "tok_s": 1000.0 / p50,
        "gates": v6.get("gates"),
    }
    payload["targets"] = {
        "E50": {"budget_ms": 20.0, "remaining_ms": p50 - 20.0},
        "E75": {"budget_ms": 1000.0 / 75.0, "remaining_ms": p50 - 1000.0 / 75.0},
        "E100_single": {"budget_ms": 10.0, "remaining_ms": p50 - 10.0},
    }
    payload["gates"] = {
        "v6_source_pass": v6.get("status") == "pass",
        "v6_all_recorded_gates_pass": all(bool(x) for x in v6.get("gates", {}).values()),
        "target_model_path_exists": (REPO / "models" / "nemotron_3_5_lightning_v35").exists(),
        "p50_matches_record_within_0_001ms": abs(p50 - 21.0923) <= 0.001,
    }
    payload["status"] = "pass" if all(payload["gates"].values()) else "gate_failed"
    payload["completed_utc"] = utc_now()
    write_json(OUT, payload)
    print(json.dumps({"status": payload["status"], "targets": payload["targets"],
                      "output": str(OUT)}, indent=2))
    return 0 if payload["status"] in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
