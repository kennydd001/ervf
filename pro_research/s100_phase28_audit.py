from __future__ import annotations

import json
import traceback

from common import utc_now, write_json_atomic
from s100_phase21_common import release
from s100_phase28_common import (
    RESULTS,
    Arm,
    make_arm,
    phase28_gate,
)

OUT = RESULTS / "S100_PHASE28_AUDIT.json"


def main() -> int:
    payload = {
        "kind": "s100_phase28_audit",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": (
            "real checkpoint pointer/resource audit; no throughput"
        ),
    }
    runtime = None

    try:
        phase28_gate()

        runtime, graph, keep = make_arm(
            128,
            Arm("direct_route"),
        )
        wrapper = graph.gmoe

        alignment = wrapper.alignment
        attributes = wrapper.mk.attributes()

        # Force CUDA/NVRTC function materialization before declaring the audit
        # complete. Actual graph capture is intentionally left to the measured
        # arm process.
        for function in wrapper.mk.f.values():
            _ = function.attributes

        payload.update(
            {
                "status": "measured",
                "alignment": alignment,
                "all_naturally_aligned_16": bool(
                    alignment[
                        "all_naturally_aligned_16"
                    ]
                ),
                "mirror_bytes_removed": int(
                    wrapper.freed_mirror_bytes
                ),
                "mirror_mib_removed": float(
                    wrapper.freed_mirror_bytes
                    / (1024.0 * 1024.0)
                ),
                "kernel_attributes": attributes,
                "V16_ARMS_ELIGIBLE": bool(
                    alignment[
                        "all_naturally_aligned_16"
                    ]
                ),
                "AUDIT_GREEN": bool(
                    wrapper.freed_mirror_bytes > 0
                ),
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "AUDIT_GREEN": False,
                "V16_ARMS_ELIGIBLE": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )
    finally:
        if runtime is not None:
            try:
                release(runtime)
            except Exception:
                pass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "all_naturally_aligned_16": payload.get(
                    "all_naturally_aligned_16"
                ),
                "mirror_mib_removed": payload.get(
                    "mirror_mib_removed"
                ),
                "V16_ARMS_ELIGIBLE": payload.get(
                    "V16_ARMS_ELIGIBLE"
                ),
                "AUDIT_GREEN": payload.get("AUDIT_GREEN"),
                "error": (
                    payload.get("error") or {}
                ).get("message"),
                "output": str(OUT),
            },
            indent=2,
        )
    )
    return 0 if payload.get("AUDIT_GREEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
