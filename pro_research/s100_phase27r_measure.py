from __future__ import annotations

import argparse
import json
import subprocess
import traceback

from common import REPO, utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase24_common import make_synth, selected_config, timed_synth_blocks
from s100_phase27_common import Variant, make_candidate, phase27_gate

RESULTS = REPO / "pro_research" / "results" / "s100_phase27r"
FROZEN_VARIANT = Variant(gather_y=4, batches=3, shared_overlap=True)


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
        values = [x.strip() for x in raw.split(",")]
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
    parser.add_argument("--arm", choices=("parent", "candidate"), required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    out = RESULTS / f"S100_PHASE27R_{args.tag.upper()}.json"
    payload = {
        "kind": "s100_phase27r_measure",
        "status": "started",
        "arm": args.arm,
        "variant": (
            FROZEN_VARIANT.as_dict() if args.arm == "candidate" else None
        ),
        "context": int(args.context),
        "tag": args.tag,
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "claim_boundary": (
            "fresh-process thermally balanced exact H4 target-only timing"
        ),
    }

    runtime = None
    try:
        cfg, _, _ = phase27_gate()

        p27 = json.loads(
            (
                REPO
                / "pro_research"
                / "results"
                / "s100_phase27"
                / "S100_PHASE27_SUMMARY.json"
            ).read_text(encoding="utf-8")
        )
        if not p27.get("instrumentation_complete"):
            raise RuntimeError("Phase27 summary is incomplete")
        if not p27.get("SELECTED_STATE_GREEN"):
            raise RuntimeError("Phase27 selected state gate is not green")
        if p27.get("PHASE27_ACTIVE_PARENT_ADOPTED"):
            raise RuntimeError(
                "Phase27 already adopted according to local summary; "
                "Phase27R premise is invalid"
            )

        selected = p27.get("selected_variant") or {}
        expected = FROZEN_VARIANT.as_dict()
        for key in ("gather_y", "batches", "shared_overlap"):
            if selected.get(key) != expected[key]:
                raise RuntimeError(
                    f"Phase27 selected variant mismatch for {key}: "
                    f"expected={expected[key]!r}, got={selected.get(key)!r}"
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

        if args.arm == "parent":
            runtime, graph, keep = make_synth(int(args.context), cfg)
        else:
            runtime, graph, keep = make_candidate(
                int(args.context), FROZEN_VARIANT
            )

        capture_info = graph.setup_graph()
        telemetry_before = telemetry()

        records, summary = timed_synth_blocks(
            runtime,
            graph,
            tokens,
            int(args.context),
            int(args.blocks),
            int(args.warmup),
        )
        telemetry_after = telemetry()

        positions = [int(row["pos"]) for row in records]
        if len(positions) != int(args.blocks):
            raise RuntimeError(
                f"measured record count {len(positions)} != {args.blocks}"
            )
        if len(set(positions)) != len(positions):
            raise RuntimeError("duplicate measured canonical positions")

        payload.update(
            {
                "status": "measured",
                "capture_info": capture_info,
                "records": records,
                "positions": positions,
                "summary": summary,
                "correctness_green": bool(
                    summary.get("all_token_exact")
                ),
                "ms_per_useful_token": float(summary["median_ms"] / 4.0),
                "target_only_tok_s": float(
                    4000.0 / summary["median_ms"]
                ),
                "telemetry": {
                    "after_graph_setup": telemetry_before,
                    "after_measure": telemetry_after,
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
                    "infeasible_vram" if is_oom else "technical_failure"
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
                "tag": args.tag,
                "context": args.context,
                "variant": payload.get("variant"),
                "summary": payload.get("summary"),
                "telemetry": payload.get("telemetry"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
