"""C3A-v2 driver: validate corrected scale layout, then rerun frozen C3A gates."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from common import REPO, environment_snapshot, gpu_processes, run_text, utc_now, write_json_atomic

PREFLIGHT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_V2_LAYOUT_PREFLIGHT.json"
ERRATUM = REPO / "pro_research" / "S100_NATIVE_NVFP4_C3A_V2_LAYOUT_ERRATUM.md"


def require_gpu_idle_wddm() -> dict:
    """Fail on real competing compute work, but ignore the tiny ChatGPT WDDM GUI context.

    On the target Windows/WDDM machine, nvidia-smi reports the ChatGPT/Codex GUI
    process in --query-compute-apps with used_memory=[N/A]. common.require_gpu_free
    therefore treats a 12 MiB, 0%-utilization graphics context as a competing CUDA
    workload. That prevented C3A-v2 from starting at all.

    This exception is deliberately narrow: only ChatGPT.exe with non-numeric [N/A]
    memory is ignored. Every other compute-app line remains a blocker. We also gate
    total GPU memory and utilization before Torch creates this process' CUDA context.
    """
    raw = gpu_processes()
    ignored: list[str] = []
    blockers: list[str] = []
    for line in raw:
        low = line.lower()
        if "chatgpt.exe" in low and "[n/a]" in low:
            ignored.append(line)
        else:
            blockers.append(line)
    if blockers:
        raise RuntimeError(
            "Another process currently owns a CUDA context. Stop cleanly and retry; "
            "C3A-v2 will not kill it.\n  " + "\n  ".join(blockers)
        )

    snap = run_text([
        "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if snap.startswith("ERROR") or snap.startswith("rc="):
        raise RuntimeError(f"Unable to query GPU idle state: {snap}")
    first = snap.splitlines()[0]
    parts = [x.strip() for x in first.split(",")]
    if len(parts) < 2:
        raise RuntimeError(f"Unexpected nvidia-smi idle-state row: {first}")
    used_mib, util_pct = int(parts[0]), int(parts[1])
    if used_mib > 1024:
        raise RuntimeError(f"GPU memory already busy: {used_mib} MiB > 1024 MiB")
    if util_pct > 10:
        raise RuntimeError(f"GPU utilization already busy: {util_pct}% > 10%")
    return {
        "compute_app_lines": raw,
        "ignored_wddm_gui_contexts": ignored,
        "gpu_memory_used_mib": used_mib,
        "gpu_utilization_percent": util_pct,
        "policy": "ignore only ChatGPT.exe with [N/A] WDDM memory; block every other compute-app; require total memory <=1024 MiB and utilization <=10%",
    }


def main() -> int:
    payload = {
        "kind": "s100_native_nvfp4_c3a_v2_layout_preflight",
        "status": "started",
        "started_utc": utc_now(),
        "erratum": str(ERRATUM.relative_to(REPO)),
    }
    try:
        # Must happen before this process creates its own CUDA context.
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
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
                          "gpu_idle_preflight": payload["gpu_idle_preflight"],
                          "layout_witness": witness, "nonuniform_native_smoke": smoke,
                          "gates": payload["gates"], "output": str(PREFLIGHT)}, indent=2))
        if payload["status"] != "layout_v2_preflight_pass":
            return 2

        # The frozen C3A diagnostic is deliberately reused. Its run_family and
        # cold_timing functions resolve c3lib.repack_b_scale at runtime, so the
        # in-process patch above changes only the physical scale layout. Skip
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
