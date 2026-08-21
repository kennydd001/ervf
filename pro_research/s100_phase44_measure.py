from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback

import numpy as np

from common import REPO, utc_now, write_json_atomic
from s100_phase21_common import expected_for_block, load_trace, prefill_to, release
from s100_phase24_common import timed_synth_blocks
from s100_phase30e_measure import telemetry
from s100_phase44_prefetch_oracle import (
    install_perfect_prefetch,
    install_route_capture,
    make_phase31_parent,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase44"


def run_isolated_arm(
    arm: str,
    *,
    context: int,
    blocks: int,
    warmup: int,
    output,
    routes_file,
    oracle_mode: str,
):
    command = [
        sys.executable,
        str(REPO / "pro_research" / "s100_phase44_arm.py"),
        "--arm",
        arm,
        "--context",
        str(context),
        "--blocks",
        str(blocks),
        "--warmup",
        str(warmup),
        "--output",
        str(output),
        "--oracle-mode",
        oracle_mode,
    ]
    if routes_file is not None:
        command.extend(("--routes-file", str(routes_file)))
    subprocess.run(command, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("status") != "measured":
        raise RuntimeError(
            f"isolated {arm} failed: {(payload.get('error') or {}).get('message')}"
        )
    return payload


def capture_routes(tokens, context: int, blocks: int, warmup: int):
    import cupy as cp

    runtime = None
    graph = None
    proxy = None
    table = None
    try:
        runtime, graph, keep = make_phase31_parent(context)
        del keep
        block_count = (context + 4 * (blocks + warmup) + 8) // 4 + 1
        proxy, table, kernels = install_route_capture(graph, block_count)
        del kernels
        capture_info = graph.setup_graph()
        proxy.restore()
        prefill_to(runtime, tokens, context)
        graph.prepare_after_prefill()
        positions = []
        for _ in range(blocks + warmup):
            pos = int(runtime.pos)
            drafts, expected = expected_for_block(tokens, pos)
            got = graph.launch(drafts.tolist())
            if not np.array_equal(got, expected):
                raise RuntimeError(
                    f"route capture token mismatch pos={pos} "
                    f"got={got.tolist()} expected={expected.tolist()}"
                )
            positions.append(pos)
        table_host = cp.asnumpy(table)
        for pos in positions:
            rows = table_host[pos // 4]
            if np.any(rows < 0) or np.any(rows >= int(runtime.n_experts)):
                raise RuntimeError(f"invalid captured routes at pos={pos}")
        return table_host, positions, capture_info
    finally:
        # Drop graph-owned runtime/CUDA references before release() runs its
        # collection and memory-pool cleanup.  Keeping these locals alive made
        # later A/C/B arms inherit several live graph allocations.
        proxy = None
        table = None
        graph = None
        if runtime is not None:
            release(runtime)


def run_parent(tokens, context: int, blocks: int, warmup: int):
    runtime = None
    graph = None
    try:
        runtime, graph, keep = make_phase31_parent(context)
        del keep
        capture_info = graph.setup_graph()
        before = telemetry()
        records, summary = timed_synth_blocks(
            runtime, graph, tokens, context, blocks, warmup
        )
        after = telemetry()
        return {
            "capture_info": capture_info,
            "records": records,
            "summary": summary,
            "telemetry": {"before": before, "after": after},
        }
    finally:
        graph = None
        if runtime is not None:
            release(runtime)


def run_oracle(
    tokens,
    route_table,
    context: int,
    blocks: int,
    warmup: int,
    oracle_mode: str,
):
    runtime = None
    graph = None
    oracle = None
    try:
        runtime, graph, keep = make_phase31_parent(context)
        del keep
        oracle, kernels = install_perfect_prefetch(
            graph, route_table, mode=oracle_mode
        )
        del kernels
        capture_info = graph.setup_graph()
        oracle.restore_after_capture()
        oracle.mismatches.fill(0)
        before = telemetry()
        records, summary = timed_synth_blocks(
            runtime, graph, tokens, context, blocks, warmup
        )
        after = telemetry()
        mismatch_count = oracle.mismatch_count()
        if mismatch_count:
            raise RuntimeError(
                f"perfect-route oracle mismatch_count={mismatch_count}"
            )
        return {
            "capture_info": capture_info,
            "records": records,
            "summary": summary,
            "route_mismatch_count": mismatch_count,
            "telemetry": {"before": before, "after": after},
        }
    finally:
        oracle = None
        graph = None
        if runtime is not None:
            release(runtime)


def bootstrap_saving(parent_a, candidate, parent_b):
    a = np.asarray([row["ms"] for row in parent_a["records"]], np.float64)
    c = np.asarray([row["ms"] for row in candidate["records"]], np.float64)
    b = np.asarray([row["ms"] for row in parent_b["records"]], np.float64)
    rng = np.random.default_rng(44)
    count = 200_000
    ma = np.median(rng.choice(a, (count, len(a)), replace=True), axis=1)
    mc = np.median(rng.choice(c, (count, len(c)), replace=True), axis=1)
    mb = np.median(rng.choice(b, (count, len(b)), replace=True), axis=1)
    savings = 0.5 * (ma + mb) - mc
    return {
        "mean_ms_per_h4": float(np.mean(savings)),
        "median_ms_per_h4": float(np.median(savings)),
        "lower95_ms_per_h4": float(np.percentile(savings, 2.5)),
        "upper95_ms_per_h4": float(np.percentile(savings, 97.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase44 perfect prefetch oracle")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--tag", default="SCREEN")
    parser.add_argument(
        "--oracle-mode",
        choices=("layer_now", "moe_l1", "moe_l2", "block_all"),
        default="layer_now",
    )
    args = parser.parse_args()

    out = RESULTS / f"S100_PHASE44_{args.tag.upper()}_ADJUDICATION.json"
    payload = {
        "kind": "s100_phase44_prefetch_oracle",
        "status": "started",
        "started_utc": utc_now(),
        "frozen_parent": (
            "codex/s100-phase31-critical-path@"
            "1046da1"
        ),
        "context": int(args.context),
        "blocks": int(args.blocks),
        "warmup": int(args.warmup),
        "oracle_mode": args.oracle_mode,
        "claim_boundary": (
            "perfect future-route target-only H4 oracle; real routers and "
            "all arithmetic remain authoritative; no predictor/drafter cost"
        ),
    }

    try:
        tokens = load_trace()["tokens"]
        required = int(args.context) + 4 * (
            int(args.blocks) + int(args.warmup)
        ) + 1
        if required > len(tokens):
            raise RuntimeError(
                f"canonical trace too short need={required} have={len(tokens)}"
            )

        RESULTS.mkdir(parents=True, exist_ok=True)
        prefix = f"S100_PHASE44_{args.tag.upper()}"
        routes_file = RESULTS / f"{prefix}_ROUTES.npz"
        route_payload = run_isolated_arm(
            "routes",
            context=int(args.context),
            blocks=int(args.blocks),
            warmup=int(args.warmup),
            output=RESULTS / f"{prefix}_ROUTES.json",
            routes_file=routes_file,
            oracle_mode=args.oracle_mode,
        )
        parent_a_payload = run_isolated_arm(
            "parent",
            context=int(args.context),
            blocks=int(args.blocks),
            warmup=int(args.warmup),
            output=RESULTS / f"{prefix}_PARENT_A.json",
            routes_file=None,
            oracle_mode=args.oracle_mode,
        )
        candidate_payload = run_isolated_arm(
            "oracle",
            context=int(args.context),
            blocks=int(args.blocks),
            warmup=int(args.warmup),
            output=RESULTS / f"{prefix}_CANDIDATE.json",
            routes_file=routes_file,
            oracle_mode=args.oracle_mode,
        )
        parent_b_payload = run_isolated_arm(
            "parent",
            context=int(args.context),
            blocks=int(args.blocks),
            warmup=int(args.warmup),
            output=RESULTS / f"{prefix}_PARENT_B.json",
            routes_file=None,
            oracle_mode=args.oracle_mode,
        )
        route_capture = route_payload["route_capture"]
        positions = route_capture["positions"]
        parent_a = parent_a_payload["result"]
        candidate = candidate_payload["result"]
        parent_b = parent_b_payload["result"]

        a_ms = float(parent_a["summary"]["median_ms"])
        c_ms = float(candidate["summary"]["median_ms"])
        b_ms = float(parent_b["summary"]["median_ms"])
        baseline_midpoint = 0.5 * (a_ms + b_ms)
        saving = baseline_midpoint - c_ms
        drift = abs(a_ms - b_ms) / max(baseline_midpoint, 1e-30)
        bootstrap = bootstrap_saving(parent_a, candidate, parent_b)
        route_open = bool(
            candidate["route_mismatch_count"] == 0
            and drift <= 0.05
            and bootstrap["lower95_ms_per_h4"] >= 2.0
        )

        payload.update(
            {
                "status": "measured",
                "route_capture": {
                    "positions": positions,
                    "layer_count": int(route_capture["layer_count"]),
                    "routes_per_layer": int(route_capture["routes_per_layer"]),
                    "capture_info": route_capture["capture_info"],
                    "routes_file": str(routes_file),
                },
                "parent_a": parent_a,
                "candidate": candidate,
                "parent_b": parent_b,
                "adjudication": {
                    "parent_a_median_ms_per_h4": a_ms,
                    "candidate_median_ms_per_h4": c_ms,
                    "parent_b_median_ms_per_h4": b_ms,
                    "baseline_midpoint_ms_per_h4": baseline_midpoint,
                    "baseline_drift_fraction": drift,
                    "saving_ms_per_h4": saving,
                    "candidate_target_only_tok_s": float(4000.0 / c_ms),
                    "bootstrap_saving": bootstrap,
                    "PREFETCH_RESEARCH_OPEN": route_open,
                    "STRUCTURAL_BREAKTHROUGH": bool(
                        route_open and bootstrap["mean_ms_per_h4"] >= 4.0
                    ),
                },
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

    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "adjudication": payload.get("adjudication"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
