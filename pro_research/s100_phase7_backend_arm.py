
"""One fresh process for legacy, packed, or destructive-control backend."""
from __future__ import annotations
import argparse
import gc
import json
import subprocess
import traceback
from typing import Any

from common import REPO, percentiles, utc_now, write_json_atomic
from diag_component_marginals_graph import (
    _prefill,
    _reset_exact_state,
    _run,
)
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from graph_e1f22 import _load_prompt_set
from s100_phase7_runtime import (
    build_phase7_runtime,
    public_record,
    recapture,
)

ROLES = ("base_a", "cand_a", "cand_b", "base_b", "bad")


def smi() -> dict[str, Any]:
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,clocks.sm,"
            "clocks.mem,power.draw,temperature.gpu,pstate",
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


def preheat(rt, ids, count):
    _reset_exact_state(rt)
    _prefill(rt, ids)
    for _ in range(count):
        rt.step_graph(None)
    rt._graph_stream.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=ROLES, required=True)
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = ap.parse_args()

    out = (
        REPO / "pro_research" / "results"
        / f"S100_PHASE7_PACKED_{args.mode.upper()}_{args.role.upper()}.json"
    )
    payload = {
        "kind": "s100_phase7_backend_arm",
        "status": "started",
        "role": args.role,
        "mode": args.mode,
        "started_utc": utc_now(),
    }

    try:
        payload["gpu_idle_preflight"] = _require_gpu_idle_wddm()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(int(n), 32) if args.mode == "smoke" else max(int(n), 256)
        warm = 48 if args.mode == "smoke" else 128
        backend = "legacy" if args.role.startswith("base") else "packed"
        bundle = build_phase7_runtime(int(capacity), backend=backend)
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
            ids[prompt["prompt"]] = [int(x) for x in generated]
            raw.extend(float(x) for x in ms)
        bundle.rt._graph_stream.synchronize()
        after = smi()
        finite = bool(cp.isfinite(bundle.rt.logits).all().item())

        payload.update(
            {
                "status": "measured",
                "runtime": public_record(bundle),
                "timing": percentiles(raw),
                "raw_timing_ms": raw,
                "ids": ids,
                "finite": finite,
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
                "role": args.role,
                "mode": args.mode,
                "timing": payload.get("timing"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 2 if payload.get("status") == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
