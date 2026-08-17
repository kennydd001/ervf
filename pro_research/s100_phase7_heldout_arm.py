
"""Evaluate exactly one frozen candidate in one fresh CUDA process."""
from __future__ import annotations
import argparse
import gc
import json
import traceback

from common import REPO, utc_now, write_json_atomic
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from s100_phase5_quality import evaluate
from s100_phase6_runtime import build_phase6_runtime
from s100_phase7_common import (
    load_frozen_candidates,
    public_spec,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    candidates = load_frozen_candidates()
    if args.candidate not in candidates:
        raise SystemExit(f"unknown frozen candidate {args.candidate}")
    spec = candidates[args.candidate]
    out = (
        REPO / "pro_research" / "results"
        / f"S100_PHASE7_HELDOUT_{args.candidate.upper()}.json"
    )
    payload = {
        "kind": "s100_phase7_heldout_arm",
        "status": "started",
        "candidate": args.candidate,
        "spec": public_spec(spec),
        "started_utc": utc_now(),
    }

    try:
        payload["gpu_idle_preflight"] = _require_gpu_idle_wddm()
        import cupy as cp

        bundle = build_phase6_runtime(
            layer_k=spec["layer_k"],
            alpha=spec["alpha"],
            backend="legacy",
        )
        result = evaluate(bundle, "heldout", True)
        payload.update(
            {
                "status": (
                    "v18_fidelity_candidate"
                    if result["official_pass"]
                    else "v18_fidelity_failed"
                ),
                **result,
                "completed_utc": utc_now(),
            }
        )
        bundle.restore_combined()
        bundle.restore_selective()
        del bundle
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )

    write_json_atomic(out, payload, archive=True)
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "status": payload.get("status"),
                "summary": payload.get("summary"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 2 if payload.get("status") == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
