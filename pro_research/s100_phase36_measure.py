from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, prefill_to, release
from s100_phase25_common import expected_for_h8, timed_parent_h8_windows
from s100_phase30e_measure import telemetry
from s100_phase32_common import make_parent
from s100_phase36_common import RESULTS, make_candidate


def summarize(rows) -> dict:
    values = np.asarray([row["ms"] for row in rows], np.float64)
    median = float(np.median(values))
    correct = sum(row["matching_ids"] for row in rows)
    total = 8 * len(rows)
    return {
        "count": len(rows),
        "median_ms": median,
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "mean_ms": float(values.mean()),
        "mad_ms": float(np.median(np.abs(values - median))),
        "target_only_tok_s": 8000.0 / median,
        "matching_ids": correct,
        "total_ids": total,
        "top1_agreement": correct / max(total, 1),
        "all_token_exact": correct == total,
    }


def timed_candidate(rt, graph, tokens, context, blocks, warmup):
    import cupy as cp

    prefill_to(rt, tokens, context)
    graph.prepare_after_prefill()
    rows = []
    saturation = 0
    represented_max = (
        448.0 * 6.0 * graph.native_head.tensor_scale_value
    )
    for index in range(warmup + blocks):
        pos = int(rt.pos)
        drafts, expected = expected_for_h8(tokens, pos)
        begin = time.perf_counter_ns()
        got = graph.launch(drafts.tolist())
        elapsed = (time.perf_counter_ns() - begin) / 1e6
        saturation += int(
            cp.asnumpy(
                cp.count_nonzero(cp.abs(graph.core.final_normed) > represented_max)
            )
        )
        if index >= warmup:
            rows.append(
                {
                    "block": index - warmup,
                    "pos": pos,
                    "ms": elapsed,
                    "got": got.tolist(),
                    "expected": expected.tolist(),
                    "matching_ids": int(np.sum(got == expected)),
                }
            )
    return rows, summarize(rows), saturation, represented_max


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase36 native FP4 head H8")
    parser.add_argument("--arm", choices=("parent", "native_head"), required=True)
    parser.add_argument("--tag", default="SCREEN")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    out = RESULTS / f"S100_PHASE36_{args.tag.upper()}_{args.arm.upper()}_CTX{args.context}.json"
    payload = {
        "kind": "s100_phase36_measure",
        "status": "started",
        "arm": args.arm,
        "tag": args.tag,
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "claim_boundary": "quality-contract target-only H8; native head is not bitexact logits",
    }
    runtime = None
    try:
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
        else:
            runtime, graph, keep = make_candidate(int(args.context))
        capture = graph.setup_graph()
        before = telemetry()
        if args.arm == "parent":
            rows, summary = timed_parent_h8_windows(
                runtime, graph, tokens, int(args.context),
                int(args.blocks), int(args.warmup),
            )
            saturation = None
            represented_max = None
        else:
            rows, summary, saturation, represented_max = timed_candidate(
                runtime, graph, tokens, int(args.context),
                int(args.blocks), int(args.warmup),
            )
        after = telemetry()
        finite_logits = True
        if args.arm != "parent":
            import cupy as cp

            finite_logits = bool(cp.asnumpy(cp.isfinite(graph.core.logits).all()))
        payload.update(
            {
                "status": "measured",
                "capture_info": capture,
                "records": rows,
                "summary": summary,
                "finite_logits": finite_logits,
                "static_represented_max": represented_max,
                "static_saturation_values": saturation,
                "screen_promotion_open": bool(
                    args.arm != "parent"
                    and summary["top1_agreement"] >= 0.99
                    and saturation == 0
                    and finite_logits
                ),
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
                "summary": payload.get("summary"),
                "finite_logits": payload.get("finite_logits"),
                "static_saturation_values": payload.get("static_saturation_values"),
                "capture_info": payload.get("capture_info"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
