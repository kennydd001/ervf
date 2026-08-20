from __future__ import annotations

import argparse
import json
import subprocess
import traceback

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase24_common import make_synth, timed_synth_blocks
from s100_phase27_common import make_candidate as make_phase27, phase27_gate
from s100_phase30e_common import (
    FROZEN_PHASE27R,
    RESULTS,
    compile_audit,
    make_candidate,
)


def telemetry() -> dict:
    fields = (
        "timestamp", "temperature.gpu", "pstate", "clocks.sm", "clocks.mem",
        "power.draw", "utilization.gpu", "memory.used",
    )
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip().splitlines()[0]
        result = dict(zip(fields, (x.strip() for x in raw.split(","))))
        for key in fields:
            if key in ("timestamp", "pstate"):
                continue
            try:
                result[key] = float(result[key])
            except Exception:
                pass
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase30E exact H4 measure")
    parser.add_argument(
        "--arm",
        choices=("parent", "phase27", "combined", "candidate", "compile"),
        required=True,
    )
    parser.add_argument("--tag", default="SMOKE")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    out = RESULTS / f"S100_PHASE30E_{args.tag.upper()}_{args.arm.upper()}.json"
    payload = {
        "kind": "s100_phase30e_measure",
        "status": "started",
        "arm": args.arm,
        "tag": args.tag,
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "phase27_variant": FROZEN_PHASE27R.as_dict(),
        "started_utc": utc_now(),
        "claim_boundary": (
            "fresh-process exact target-only H4 timing; candidate is Phase27R + "
            "direct-L2 shared M4 + two-launch M1-2/M3-4 routed-UP dispatch"
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
            cfg, _, _ = phase27_gate()
            tokens = load_trace()["tokens"]
            required = int(args.context) + 4 * (
                int(args.blocks) + int(args.warmup)
            ) + 1
            if required > len(tokens):
                raise RuntimeError(
                    f"canonical trace too short need={required} have={len(tokens)}"
                )

            if args.arm == "parent":
                runtime, graph, keep = make_synth(int(args.context), cfg)
            elif args.arm == "phase27":
                runtime, graph, keep = make_phase27(
                    int(args.context), FROZEN_PHASE27R
                )
            elif args.arm == "combined":
                runtime, graph, keep = make_candidate(
                    int(args.context), shared_direct=True, group_dispatch=False
                )
            else:
                runtime, graph, keep = make_candidate(int(args.context))

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

    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "arm": args.arm,
                "tag": args.tag,
                "summary": payload.get("summary"),
                "correctness_green": payload.get("correctness_green"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") in {"compiled", "measured"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
