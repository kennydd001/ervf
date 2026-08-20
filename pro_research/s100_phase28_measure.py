from __future__ import annotations

import argparse
import json
import subprocess
import traceback

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase24_common import (
    make_synth,
    selected_config,
    timed_synth_blocks,
)
from s100_phase28_common import (
    ARM_NAMES,
    RESULTS,
    Arm,
    make_arm,
    phase28_gate,
    timed_arm,
)


def telemetry() -> dict:
    fields = (
        "timestamp",
        "temperature.gpu",
        "pstate",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "utilization.gpu",
        "memory.used",
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
        values = [item.strip() for item in raw.split(",")]
        row = dict(zip(fields, values))
        for key in fields:
            if key in ("timestamp", "pstate"):
                continue
            try:
                row[key] = float(row[key])
            except Exception:
                pass
        return row
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("parent",) + ARM_NAMES,
        required=True,
    )
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=6)
    args = parser.parse_args()

    out = RESULTS / f"S100_PHASE28_{args.tag.upper()}.json"
    arm = None if args.arm == "parent" else Arm(args.arm)

    payload = {
        "kind": "s100_phase28_measure",
        "status": "started",
        "arm": args.arm,
        "arm_config": None if arm is None else arm.as_dict(),
        "context": int(args.context),
        "tag": args.tag,
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "claim_boundary": (
            "fresh-process exact H4 target-only timing"
        ),
    }

    runtime = None
    try:
        cfg, _, _, _ = phase28_gate()

        preflight = json.loads(
            (
                RESULTS / "S100_PHASE28_PREFLIGHT.json"
            ).read_text(encoding="utf-8")
        )
        audit = json.loads(
            (
                RESULTS / "S100_PHASE28_AUDIT.json"
            ).read_text(encoding="utf-8")
        )
        if not preflight.get("PREFLIGHT_GREEN"):
            raise RuntimeError("Phase28 preflight is not green")
        if not audit.get("AUDIT_GREEN"):
            raise RuntimeError("Phase28 actual audit is not green")
        if (
            arm is not None
            and arm.vector_bytes == 16
            and not audit.get("V16_ARMS_ELIGIBLE")
        ):
            raise RuntimeError(
                f"{arm.name} is ineligible: real pointer/stride "
                "alignment is not 16-byte safe"
            )

        trace = load_trace()
        tokens = trace["tokens"]
        required = int(args.context) + 4 * (
            int(args.blocks) + int(args.warmup)
        ) + 1
        if required > len(tokens):
            raise RuntimeError(
                f"canonical trace too short: need={required}, "
                f"have={len(tokens)}"
            )

        if arm is None:
            runtime, graph, keep = make_synth(
                int(args.context),
                cfg,
            )
        else:
            runtime, graph, keep = make_arm(
                int(args.context),
                arm,
            )

        capture_info = graph.setup_graph()
        before = telemetry()

        if arm is None:
            records, summary = timed_synth_blocks(
                runtime,
                graph,
                tokens,
                int(args.context),
                int(args.blocks),
                int(args.warmup),
            )
        else:
            records, summary = timed_arm(
                runtime,
                graph,
                tokens,
                int(args.context),
                int(args.blocks),
                int(args.warmup),
            )

        after = telemetry()

        wrapper = getattr(graph, "gmoe", None)
        mirror_bytes_removed = int(
            getattr(wrapper, "freed_mirror_bytes", 0)
        )
        alignment = getattr(wrapper, "alignment", None)

        payload.update(
            {
                "status": "measured",
                "capture_info": capture_info,
                "records": records,
                "summary": summary,
                "correctness_green": bool(
                    summary.get("all_token_exact")
                ),
                "ms_per_useful_token": float(
                    summary["median_ms"] / 4.0
                ),
                "target_only_tok_s": float(
                    4000.0 / summary["median_ms"]
                ),
                "mirror_bytes_removed": mirror_bytes_removed,
                "alignment": alignment,
                "telemetry": {
                    "after_graph_setup": before,
                    "after_measure": after,
                },
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        message = str(exc).lower()
        is_oom = (
            "out of memory" in message
            or "cuda_error_out_of_memory" in message
            or type(exc).__name__.lower()
            in ("outofmemoryerror", "memoryerror")
        )
        payload.update(
            {
                "status": (
                    "infeasible_vram"
                    if is_oom
                    else "technical_failure"
                ),
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
                "arm_config": payload.get("arm_config"),
                "context": args.context,
                "tag": args.tag,
                "summary": payload.get("summary"),
                "mirror_bytes_removed": payload.get(
                    "mirror_bytes_removed"
                ),
                "telemetry": payload.get("telemetry"),
                "error": (
                    payload.get("error") or {}
                ).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
