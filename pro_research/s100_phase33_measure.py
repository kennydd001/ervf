from __future__ import annotations

import argparse
import json
import traceback

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase25_common import timed_h8_blocks, timed_parent_h8_windows
from s100_phase30e_measure import telemetry
from s100_phase32_common import make_candidate as make_phase32_control
from s100_phase32_common import make_parent
from s100_phase33_common import ARMS, RESULTS, compile_audit, make_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase33 exact NVFP4 M8 measure")
    parser.add_argument(
        "--arm", choices=("parent", "phase32_control", *ARMS, "compile"),
        required=True,
    )
    parser.add_argument("--tag", default="SCREEN")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    out = RESULTS / f"S100_PHASE33_{args.tag.upper()}_{args.arm.upper()}_CTX{args.context}.json"
    payload = {
        "kind": "s100_phase33_measure",
        "status": "started",
        "arm": args.arm,
        "tag": args.tag,
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "claim_boundary": "fresh-process exact target-only H8; no drafter cost",
    }
    runtime = None
    try:
        if args.arm == "compile":
            payload.update(
                status="compiled",
                kernel_resources=compile_audit(),
                completed_utc=utc_now(),
            )
        else:
            tokens = load_trace()["tokens"]
            required = int(args.context) + 8 * (
                int(args.blocks) + int(args.warmup)
            ) + 1
            if required > len(tokens):
                raise RuntimeError(
                    f"canonical trace too short need={required} have={len(tokens)}"
                )
            if args.arm == "parent":
                runtime, graph, keep = make_parent(int(args.context))
                runner = timed_parent_h8_windows
            elif args.arm == "phase32_control":
                runtime, graph, keep = make_phase32_control(
                    int(args.context), "dense_m8"
                )
                runner = timed_h8_blocks
            else:
                runtime, graph, keep = make_candidate(int(args.context), args.arm)
                runner = timed_h8_blocks
            capture = graph.setup_graph()
            before = telemetry()
            records, summary = runner(
                runtime, graph, tokens, int(args.context),
                int(args.blocks), int(args.warmup),
            )
            after = telemetry()
            payload.update(
                {
                    "status": "measured",
                    "capture_info": capture,
                    "records": records,
                    "summary": summary,
                    "correctness_green": bool(summary.get("all_token_exact")),
                    "ms_per_useful_token": float(summary["median_ms"] / 8.0),
                    "target_only_tok_s": float(8000.0 / summary["median_ms"]),
                    "zero_cost_s100_ceiling_open": bool(summary["median_ms"] <= 80.0),
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
        if runtime is not None:
            try:
                release(runtime)
            except Exception:
                pass
    write_json_atomic(out, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "arm": args.arm,
                "tag": args.tag,
                "summary": payload.get("summary"),
                "kernel_resources": payload.get("kernel_resources"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") in {"compiled", "measured"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
