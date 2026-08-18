
"""One fresh process for a phase-8 static-cache timing arm."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import traceback

from common import REPO, percentiles, utc_now, write_json_atomic
from diag_component_marginals_graph import (
    _prefill,
    _reset_exact_state,
    _run,
)
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from graph_e1f22 import _load_prompt_set
from s100_phase8_common import (
    BUDGETS,
    selection_for,
    selection_hash,
)
from s100_phase8_runtime import (
    build_phase8_runtime,
    public_record,
    recapture,
)

ROLES = ("base_a", "cand_a", "cand_b", "base_b", "bad")


def preheat(rt, ids, count):
    _reset_exact_state(rt)
    _prefill(rt, ids)
    for _ in range(count):
        rt.step_graph(None)
    rt._graph_stream.synchronize()


def smi():
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,"
            "clocks.sm,clocks.mem,power.draw,"
            "temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode:
        return {"error": (p.stderr or p.stdout).strip()}
    v = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
    return {
        "memory_used_mib": int(v[0]),
        "utilization_percent": int(v[1]),
        "sm_mhz": float(v[2]),
        "mem_mhz": float(v[3]),
        "power_w": float(v[4]),
        "temperature_c": float(v[5]),
        "pstate": v[6],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--budget",
        type=int,
        choices=BUDGETS,
        required=True,
    )
    ap.add_argument("--role", choices=ROLES, required=True)
    ap.add_argument(
        "--mode",
        choices=("smoke", "full"),
        required=True,
    )
    args = ap.parse_args()

    out = (
        REPO / "pro_research" / "results"
        / (
            f"S100_PHASE8_STATIC_{args.budget}_"
            f"{args.mode.upper()}_{args.role.upper()}.json"
        )
    )
    payload = {
        "kind": "s100_phase8_backend_arm",
        "status": "started",
        "budget": int(args.budget),
        "role": args.role,
        "mode": args.mode,
        "started_utc": utc_now(),
    }

    try:
        payload["gpu_idle_preflight"] = _require_gpu_idle_wddm()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(
            args.mode
        )
        n = (
            min(int(n), 32)
            if args.mode == "smoke"
            else max(int(n), 256)
        )
        warm = 48 if args.mode == "smoke" else 128

        selection = selection_for(args.budget)
        expected_hash = selection_hash(selection)
        backend = (
            "legacy"
            if args.role.startswith("base")
            else "static"
        )
        bundle = build_phase8_runtime(
            capacity=int(capacity),
            selection=selection,
            backend=backend,
        )
        if args.role == "bad":
            bundle.rt._bad_pick = 1
            recapture(bundle)

        preheat(bundle.rt, prompts[0]["prompt_ids"], warm)
        before = smi()
        raw = []
        ids = {}
        for prompt in prompts:
            generated, ms = _run(
                bundle.rt, prompt["prompt_ids"], n
            )
            ids[prompt["prompt"]] = [
                int(x) for x in generated
            ]
            raw.extend(float(x) for x in ms)
        bundle.rt._graph_stream.synchronize()
        after = smi()

        record = public_record(bundle)
        actual_hash = (
            record["static_cache"]["selection_sha256"]
            if record["static_cache"] is not None
            else None
        )
        payload.update(
            {
                "status": "measured",
                "runtime": record,
                "expected_selection_sha256": expected_hash,
                "actual_selection_sha256": actual_hash,
                "timing": percentiles(raw),
                "raw_timing_ms": raw,
                "ids": ids,
                "finite": bool(
                    cp.isfinite(bundle.rt.logits).all().item()
                ),
                "vram_mib": max(
                    int(before.get("memory_used_mib", 0)),
                    int(after.get("memory_used_mib", 0)),
                ),
                "smi_before": before,
                "smi_after": after,
                "completed_utc": utc_now(),
            }
        )
        bundle.restore_combined()
        bundle.restore_selective()
        del bundle
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
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

    write_json_atomic(out, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "budget": args.budget,
                "role": args.role,
                "mode": args.mode,
                "timing": payload.get("timing"),
                "vram_mib": payload.get("vram_mib"),
                "error": (payload.get("error") or {}).get(
                    "message"
                ),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 2 if payload.get("status") == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
