"""Phase43 B2/B4 full-pipeline geometry screen."""
from __future__ import annotations

import argparse
import json
import traceback

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase25_common import timed_h8_blocks
from s100_phase30e_measure import telemetry
from s100_phase32_common import make_candidate as make_phase32
from s100_phase42_h8_global_pipeline import make_overlap_geometry

RESULTS = REPO / "pro_research" / "results" / "s100_phase43"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("base_a", "global_b2", "global_b4", "base_b"), required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    out = RESULTS / f"S100_PHASE43_{args.arm.upper()}_CTX{args.context}.json"
    payload = {
        "kind": "s100_phase43_measure",
        "status": "started",
        "arm": args.arm,
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "preregistration": "pro_research/S100_PHASE43_H8_PIPELINE_GEOMETRY_PREREGISTRATION.md",
        "claim_boundary": "exact synchronous target-only H8 geometry screen",
    }
    rt = None
    try:
        tokens = load_trace()["tokens"]
        steps = 8 * (int(args.blocks) + int(args.warmup))
        allocation_context = int(args.context) + steps
        if allocation_context + 1 > len(tokens):
            raise RuntimeError("canonical trace too short")
        if args.arm == "global_b2":
            rt, graph, keep = make_overlap_geometry(allocation_context, 2)
            ranges = [[0, 24], [24, 48]]
        elif args.arm == "global_b4":
            rt, graph, keep = make_overlap_geometry(allocation_context, 4)
            ranges = [[0, 12], [12, 24], [24, 36], [36, 48]]
        else:
            rt, graph, keep = make_phase32(allocation_context, "dense_m8")
            ranges = None
        capture = graph.setup_graph()
        before = telemetry()
        records, summary = timed_h8_blocks(
            rt, graph, tokens, int(args.context), int(args.blocks), int(args.warmup)
        )
        payload.update({
            "status": "measured",
            "allocation_context": allocation_context,
            "ranges": ranges,
            "capture_info": capture,
            "records": records,
            "summary": summary,
            "tokens_exact": bool(summary.get("all_token_exact")),
            "target_only_tok_s": 8000.0 / float(summary["median_ms"]),
            "telemetry": {"before": before, "after": telemetry()},
            "keep_objects": len(keep),
            "environment": environment_snapshot(),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
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
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())

