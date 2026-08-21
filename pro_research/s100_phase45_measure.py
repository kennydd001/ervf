from __future__ import annotations

import argparse
import json
import traceback

from common import REPO, utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase24_common import timed_synth_blocks
from s100_phase30e_measure import telemetry
from s100_phase31_common import make_attention_head_direct_candidate
from s100_phase45_common import compile_audit, make_candidate


RESULTS = REPO / "pro_research" / "results" / "s100_phase45"
ARMS = ("parent", "p2_p4", "p4_p4", "p4_p8", "compile")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase45 persistent UP screen")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--tag", default="SCREEN")
    args = parser.parse_args()

    out = RESULTS / f"S100_PHASE45_{args.tag.upper()}_{args.arm.upper()}.json"
    payload = {
        "kind": "s100_phase45_persistent_up",
        "status": "started",
        "arm": args.arm,
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "frozen_parent": "codex/s100-phase31-critical-path@1046da1",
        "claim_boundary": "exact target-only H4; routed-UP CTA ownership only",
    }
    runtime = None
    graph = None
    try:
        if args.arm == "compile":
            payload.update(
                {
                    "status": "compiled",
                    "resource_audit": {
                        schedule: compile_audit(schedule)
                        for schedule in ("p2_p4", "p4_p4", "p4_p8")
                    },
                    "completed_utc": utc_now(),
                }
            )
        else:
            tokens = load_trace()["tokens"]
            if args.arm == "parent":
                runtime, graph, keep = make_attention_head_direct_candidate(
                    args.context, head_mode="m4"
                )
            else:
                runtime, graph, keep = make_candidate(args.context, args.arm)
            del keep
            capture_info = graph.setup_graph()
            before = telemetry()
            records, summary = timed_synth_blocks(
                runtime,
                graph,
                tokens,
                args.context,
                args.blocks,
                args.warmup,
            )
            after = telemetry()
            payload.update(
                {
                    "status": "measured",
                    "capture_info": capture_info,
                    "records": records,
                    "summary": summary,
                    "telemetry": {"before": before, "after": after},
                    "completed_utc": utc_now(),
                }
            )
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
    finally:
        graph = None
        if runtime is not None:
            release(runtime)

    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, payload, archive=True)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "status": payload.get("status"),
                "summary": payload.get("summary"),
                "resource_audit": payload.get("resource_audit"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") in ("compiled", "measured") else 2


if __name__ == "__main__":
    raise SystemExit(main())
