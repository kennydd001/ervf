"""Phase42 smoke and paired measurement runner."""
from __future__ import annotations

import argparse
import json
import traceback

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase25_common import timed_h8_blocks
from s100_phase30e_measure import telemetry
from s100_phase40_h8_pipeline_kernels import Phase40H8PipelineKernels
from s100_phase42_h8_global_pipeline import make_overlap, make_parent, make_serial
from s100_phase42_range_dispatch_kernels import Phase42RangeDispatchKernels

RESULTS = REPO / "pro_research" / "results" / "s100_phase42"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("compile", "serial_smoke", "overlap_smoke", "base_a", "global_pipeline_b3", "base_b"),
        required=True,
    )
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    out = RESULTS / f"S100_PHASE42_{args.arm.upper()}_CTX{args.context}.json"
    payload = {
        "kind": "s100_phase42_measure",
        "status": "started",
        "arm": args.arm,
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "preregistration": "pro_research/S100_PHASE42_H8_GLOBAL_RANGE_PREREGISTRATION.md",
        "claim_boundary": "exact synchronous target-only H8",
    }
    rt = None
    try:
        if args.arm == "compile":
            payload.update({
                "status": "compiled",
                "kernel_resources": {
                    "dispatch": Phase42RangeDispatchKernels().resource_audit(),
                    "gather_down": Phase40H8PipelineKernels().resource_audit(),
                },
                "environment": environment_snapshot(),
                "completed_utc": utc_now(),
            })
        else:
            tokens = load_trace()["tokens"]
            steps = 8 * (int(args.blocks) + int(args.warmup))
            if int(args.context) + steps + 1 > len(tokens):
                raise RuntimeError("canonical trace too short")
            allocation_context = int(args.context) + steps
            if args.arm == "serial_smoke":
                rt, graph, keep = make_serial(allocation_context)
            elif args.arm in ("overlap_smoke", "global_pipeline_b3"):
                rt, graph, keep = make_overlap(allocation_context)
            else:
                rt, graph, keep = make_parent(allocation_context)
            capture = graph.setup_graph()
            before = telemetry()
            records, summary = timed_h8_blocks(
                rt, graph, tokens, int(args.context), int(args.blocks), int(args.warmup)
            )
            payload.update({
                "status": "measured",
                "allocation_context": allocation_context,
                "capture_info": capture,
                "records": records,
                "summary": summary,
                "tokens_exact": bool(summary.get("all_token_exact")),
                "target_only_tok_s": 8000.0 / float(summary["median_ms"]),
                "telemetry": {"before": before, "after": telemetry()},
                "keep_objects": len(keep),
                "environment": environment_snapshot((
                    REPO / "pro_research" / "s100_phase42_range_dispatch_kernels.py",
                    REPO / "pro_research" / "s100_phase42_h8_global_pipeline.py",
                )),
                "completed_utc": utc_now(),
            })
    except Exception as exc:
        payload.update({
            "status": "infeasible_vram" if "out of memory" in str(exc).lower() else "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    finally:
        if rt is not None:
            try:
                release(rt)
            except Exception:
                pass
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "arm": args.arm,
        "summary": payload.get("summary"),
        "kernel_resources": payload.get("kernel_resources"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") in {"compiled", "measured"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

