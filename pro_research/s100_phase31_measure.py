from __future__ import annotations

import argparse
import json
import traceback

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase24_common import timed_synth_blocks
from s100_phase30e_common import make_candidate as make_parent
from s100_phase30e_measure import telemetry
from s100_phase31_common import (
    RESULTS,
    compile_audit,
    make_attention_head_direct_candidate,
    make_attention_direct_candidate,
    make_candidate,
    make_dense_direct_candidate,
    make_group_down_candidate,
    make_staged_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase31 exact H4 measure")
    parser.add_argument(
        "--arm",
        choices=(
            "parent",
            "sink",
            "reduce_sink",
            "staged",
            "group_down",
            "attention_direct",
            "dense_direct",
            "attention_head_m4",
            "attention_head_m2",
            "compile",
        ),
        required=True,
    )
    parser.add_argument("--tag", default="SMOKE")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    out = RESULTS / f"S100_PHASE31_{args.tag.upper()}_{args.arm.upper()}.json"
    payload = {
        "kind": "s100_phase31_measure",
        "status": "started",
        "arm": args.arm,
        "tag": args.tag,
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "claim_boundary": (
            "fresh-process exact target-only H4 timing; Phase30E parent versus "
            "exact MoE terminal residual-sink fusion"
        ),
    }
    runtime = None
    try:
        if args.arm == "compile":
            payload.update(
                {
                    "status": "compiled",
                    "kernel_resources": compile_audit(),
                    "completed_utc": utc_now(),
                }
            )
        else:
            tokens = load_trace()["tokens"]
            required = int(args.context) + 4 * (
                int(args.blocks) + int(args.warmup)
            ) + 1
            if required > len(tokens):
                raise RuntimeError(
                    f"canonical trace too short need={required} have={len(tokens)}"
                )
            if args.arm == "parent":
                runtime, graph, keep = make_parent(int(args.context))
            elif args.arm == "staged":
                runtime, graph, keep = make_staged_candidate(int(args.context))
            elif args.arm == "group_down":
                runtime, graph, keep = make_group_down_candidate(
                    int(args.context)
                )
            elif args.arm == "attention_direct":
                runtime, graph, keep = make_attention_direct_candidate(
                    int(args.context)
                )
            elif args.arm == "dense_direct":
                runtime, graph, keep = make_dense_direct_candidate(
                    int(args.context)
                )
            elif args.arm.startswith("attention_head_"):
                runtime, graph, keep = make_attention_head_direct_candidate(
                    int(args.context), head_mode=args.arm.rsplit("_", 1)[-1]
                )
            else:
                runtime, graph, keep = make_candidate(
                    int(args.context), mode=args.arm
                )

            capture = graph.setup_graph()
            before = telemetry()
            records, summary = timed_synth_blocks(
                runtime,
                graph,
                tokens,
                int(args.context),
                int(args.blocks),
                int(args.warmup),
            )
            after = telemetry()
            payload.update(
                {
                    "status": "measured",
                    "capture_info": capture,
                    "records": records,
                    "summary": summary,
                    "correctness_green": bool(summary.get("all_token_exact")),
                    "ms_per_useful_token": float(summary["median_ms"] / 4.0),
                    "target_only_tok_s": float(4000.0 / summary["median_ms"]),
                    "telemetry": {"before": before, "after": after},
                    "completed_utc": utc_now(),
                }
            )
    except Exception as exc:
        message = str(exc).lower()
        oom = "out of memory" in message or "cuda_error_out_of_memory" in message
        payload.update(
            {
                "status": "infeasible_vram" if oom else "technical_failure",
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
