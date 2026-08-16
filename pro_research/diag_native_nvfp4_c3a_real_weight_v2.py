"""C3A-v2 driver: validate corrected scale layout, then rerun frozen C3A gates."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from common import REPO, environment_snapshot, require_gpu_free, utc_now, write_json_atomic

PREFLIGHT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_V2_LAYOUT_PREFLIGHT.json"
ERRATUM = REPO / "pro_research" / "S100_NATIVE_NVFP4_C3A_V2_LAYOUT_ERRATUM.md"


def main() -> int:
    payload = {
        "kind": "s100_native_nvfp4_c3a_v2_layout_preflight",
        "status": "started",
        "started_utc": utc_now(),
        "erratum": str(ERRATUM.relative_to(REPO)),
    }
    try:
        # Must happen before this process creates its own CUDA context.
        require_gpu_free()
        import torch
        import torch.nn.functional as F
        import native_nvfp4_c3a_lib as c3lib
        import native_nvfp4_c3a_layout_v2 as v2

        v2.install(c3lib)
        witness = v2.layout_witness(torch)
        smoke = v2.nonuniform_native_smoke(torch, F, F.ScalingType, F.SwizzleType, c3lib)
        payload.update({
            "revision": v2.REVISION,
            "legacy_commit": v2.LEGACY_COMMIT,
            "environment": environment_snapshot((Path(__file__), ERRATUM)),
            "layout_witness": witness,
            "nonuniform_native_smoke": smoke,
            "gates": {
                "V2_G1_row_block_major_byte_witness": bool(witness.get("passes")),
                "V2_G2_witness_discriminates_legacy_k_major": int(witness.get("legacy_k_major_byte_mismatches", 0)) > 0,
                "V2_G3_nonuniform_native_smoke_exact": bool(smoke.get("passes")),
            },
        })
        payload["status"] = "layout_v2_preflight_pass" if all(payload["gates"].values()) else "layout_v2_preflight_fail"
        payload["completed_utc"] = utc_now()
        write_json_atomic(PREFLIGHT, payload, archive=True)
        print(json.dumps({"status": payload["status"], "revision": payload["revision"],
                          "layout_witness": witness, "nonuniform_native_smoke": smoke,
                          "gates": payload["gates"], "output": str(PREFLIGHT)}, indent=2))
        if payload["status"] != "layout_v2_preflight_pass":
            return 2

        # The frozen C3A diagnostic is deliberately reused. Its run_family and
        # cold_timing functions resolve c3lib.repack_b_scale at runtime, so the
        # in-process patch above changes only the physical scale layout.  Skip
        # its second GPU-free probe because this process now owns the context.
        import diag_native_nvfp4_c3a_real_weight as base
        base.require_gpu_free = lambda: None
        return int(base.main())
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})
        write_json_atomic(PREFLIGHT, payload, archive=True)
        print(json.dumps({"status": payload["status"], "error": payload["error"], "output": str(PREFLIGHT)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
