
"""One completely fresh V18 timing arm for phase 4."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

from common import (
    REPO,
    environment_snapshot,
    percentiles,
    utc_now,
    write_json_atomic,
)
from diag_component_marginals_graph import _run
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from graph_e1f22 import _load_prompt_set
from s100_phase3_runtime import build_v18_runtime, public_bundle_record

PROFILES = (
    "qfast", "mamba", "fast", "k5", "k4", "fast_k5", "fast_k4"
)
ROLES = ("exact_a", "cand_a", "cand_b", "exact_b")
PREREG = (
    REPO / "pro_research"
    / "S100_PHASE4_FRESH_TIMING_PREREGISTRATION.md"
)


def _smi() -> dict[str, Any]:
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,"
            "power.draw,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode:
        return {"error": (p.stderr or p.stdout).strip()}
    v = [x.strip() for x in (p.stdout or "").splitlines()[0].split(",")]
    return {
        "memory_used_mib": int(v[0]),
        "utilization_percent": int(v[1]),
        "sm_mhz": float(v[2]),
        "mem_mhz": float(v[3]),
        "power_w": float(v[4]),
        "temperature_c": float(v[5]),
        "pstate": v[6],
    }


def _preheat(rt, prompt_ids: list[int], count: int) -> None:
    from diag_component_marginals_graph import (
        _prefill,
        _reset_exact_state,
    )
    _reset_exact_state(rt)
    _prefill(rt, prompt_ids)
    for _ in range(count):
        rt.step_graph(None)
    rt._graph_stream.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--role", choices=ROLES, required=True)
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = ap.parse_args()

    out = (
        REPO / "pro_research" / "results"
        / (
            f"S100_PHASE4_FRESH_{args.profile.upper()}_"
            f"{args.mode.upper()}_{args.role.upper()}.json"
        )
    )
    runtime_profile = None if args.role.startswith("exact") else args.profile
    payload: dict[str, Any] = {
        "kind": "s100_phase4_fresh_arm",
        "status": "started",
        "series_profile": args.profile,
        "runtime_profile": runtime_profile or "exact",
        "role": args.role,
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "one fresh-process full V18 arm; comparison and quality are "
            "separate artifacts"
        ),
    }

    try:
        payload["gpu_idle_preflight"] = _require_gpu_idle_wddm()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(int(n), 32) if args.mode == "smoke" else max(int(n), 256)
        preheat = 48 if args.mode == "smoke" else 128
        payload["config"] = {
            "tokens_per_prompt": n,
            "prompt_count": len(prompts),
            "capacity": int(capacity),
            "preheat_tokens": preheat,
        }
        payload["environment_start"] = environment_snapshot(
            (
                Path(__file__),
                PREREG,
                REPO / "pro_research" / "s100_phase3_profiles.py",
                REPO / "pro_research" / "s100_phase3_runtime.py",
                REPO / "pro_research" / "d4_mixed_runtime.py",
                REPO / "pro_research" / "moe_dev_combined.py",
            )
        )

        bundle = build_v18_runtime(int(capacity), runtime_profile)
        rt = bundle.rt
        _preheat(rt, prompts[0]["prompt_ids"], preheat)

        before = _smi()
        raw: list[float] = []
        prompt_records = []
        for p in prompts:
            ids, ms = _run(rt, p["prompt_ids"], n)
            raw.extend(float(x) for x in ms)
            prompt_records.append(
                {
                    "prompt": p["prompt"],
                    "kind": p["kind"],
                    "ids": [int(x) for x in ids],
                    "timing_ms": [float(x) for x in ms],
                }
            )
        rt._graph_stream.synchronize()
        after = _smi()
        finite = bool(cp.isfinite(rt.logits).all().item())

        payload.update(
            {
                "status": "measured",
                "runtime": public_bundle_record(bundle),
                "finite": finite,
                "timing": percentiles(raw),
                "raw_timing_ms": raw,
                "prompts": prompt_records,
                "smi_before": before,
                "smi_after": after,
                "vram_mib": max(
                    int(before.get("memory_used_mib", 0)),
                    int(after.get("memory_used_mib", 0)),
                ),
                "environment_end": environment_snapshot(),
                "completed_utc": utc_now(),
            }
        )
        bundle.restore_combined()
        bundle.restore_selective()
        del rt, bundle
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "completed_utc": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    write_json_atomic(out, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "series_profile": args.profile,
                "runtime_profile": payload.get("runtime_profile"),
                "role": args.role,
                "mode": args.mode,
                "timing": payload.get("timing"),
                "vram_mib": payload.get("vram_mib"),
                "smi_before": payload.get("smi_before"),
                "smi_after": payload.get("smi_after"),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(out),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 2 if payload.get("status") == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
