"""PRO-G0: execute the already-built, frozen E1F22 full-token CUDA graph A/B.

The runtime graph code landed in commit 96811c4 but had not been measured. This
runner does not modify that code or its frozen gates. It writes all output under
``pro_research/results`` and can therefore be run without touching the closed
Treesweep/NERVF reports.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    REPO,
    environment_snapshot,
    first_divergence,
    load_json,
    percentiles,
    require_gpu_free,
    require_model_dir,
    result_path,
    utc_now,
    write_json_atomic,
)

TS200 = REPO / "reports" / "treesweep200"
OUT = result_path("PRO_G0_E1F22_GRAPH_AB.json")
CODE_PROMPT = (
    "Write a correct Python function that computes the longest increasing "
    "subsequence length in O(n log n), then explain its invariant.\n"
)


def _load_prompt_set(mode: str) -> tuple[list[dict[str, Any]], dict[str, list[int]], int, int]:
    anchor = load_json(TS200 / "V36_DETERMINISTIC_ANCHOR.json")
    a1 = load_json(TS200 / "A1_ADOPTION_PRECONDITION.json")
    expected = a1["gates"]["G_A2_ANCHOR_informative"]["produced_ids"]
    prompts = [
        {"prompt": p["prompt"], "prompt_ids": [int(x) for x in p["prompt_ids"]], "kind": "anchor"}
        for p in anchor["prompts"]
    ]

    model = require_model_dir()
    tokenizer_error = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model, local_files_only=True, trust_remote_code=True, use_fast=True
        )
        code_ids = tok.encode(CODE_PROMPT, add_special_tokens=False)
        if not code_ids:
            raise RuntimeError("tokenizer returned no ids")
        prompts.append({"prompt": CODE_PROMPT, "prompt_ids": [int(x) for x in code_ids], "kind": "code"})
    except Exception as exc:  # recorded, not hidden
        tokenizer_error = f"{type(exc).__name__}: {exc}"
        if mode == "full":
            raise RuntimeError(
                "Full E1F22 requires the preregistered third code prompt, but local "
                f"tokenization failed: {tokenizer_error}"
            ) from exc

    n = 16 if mode == "smoke" else 256
    capacity = int(anchor.get("capacity", 72))
    for p in prompts:
        p["tokenizer_error"] = tokenizer_error
    return prompts, expected, n, capacity


def _new_runtime(capacity: int):
    # Imports after the GPU-free preflight so this process is the only context.
    sys.path.insert(0, str(REPO / "src"))
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(
        require_model_dir(), contexts_max=4096, embed_on_host=True,
        fp8_kv=True, verbose=False,
    )
    rt.enable_cache(capacity)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True
    return rt


def _run_eager_timed(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    import cupy as cp

    rt.reset()
    nxt = None
    for token in prompt_ids:
        nxt = int(rt.step(int(token)))
    if nxt is None:
        raise ValueError("prompt must contain at least one token")
    cp.cuda.Device(0).synchronize()

    ids = [nxt]
    samples: list[float] = []
    cur = nxt
    for _ in range(n - 1):
        t0 = time.perf_counter_ns()
        cur = int(rt.step(cur))
        cp.cuda.Device(0).synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(cur)
    return ids, samples


def _run_graph_timed(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    rt.reset()
    start = int(rt._ring_i)
    for token in prompt_ids:
        rt.step_graph(int(token))
    first_slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    cur = int(rt.ring_harvest(first_slot, 1)[0])
    ids = [cur]
    samples: list[float] = []
    for _ in range(n - 1):
        slot = int(rt._ring_i)
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        cur = int(rt.ring_harvest(slot, 1)[0])
        samples.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(cur)
    return ids, samples


def _run_graph_collect(rt, prompt_ids: list[int], n: int) -> list[int]:
    rt.reset()
    start = int(rt._ring_i)
    for token in prompt_ids:
        rt.step_graph(int(token))
    for _ in range(n - 1):
        rt.step_graph(None)
    first_slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    return [int(x) for x in rt.ring_harvest(first_slot, n)]


def _prompt_result(prompt: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    return {
        "prompt": prompt["prompt"],
        "kind": prompt["kind"],
        "prompt_tokens": len(prompt["prompt_ids"]),
        "ids": ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    ap.add_argument("--skip-control", action="store_true", help="technical smoke only")
    args = ap.parse_args()

    started = utc_now()
    payload: dict[str, Any] = {
        "kind": "pro_g0_e1f22_graph_ab",
        "status": "started",
        "mode": args.mode,
        "started_utc": started,
        "claim_boundary": (
            "End-to-end A/B of the already-built E1F22 graph implementation in the "
            "registered <=4096 context regime. No long-context or 50 tok/s claim."
        ),
    }

    try:
        require_gpu_free()
        prompts, expected, n, capacity = _load_prompt_set(args.mode)
        payload["config"] = {
            "tokens_per_prompt": n,
            "capacity": capacity,
            "contexts_max": 4096,
            "prompts": [{"prompt": p["prompt"], "kind": p["kind"], "prompt_tokens": len(p["prompt_ids"])} for p in prompts],
            "skip_control": bool(args.skip_control),
        }
        runtime_file = REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py"
        prereg_file = TS200 / "E1F22_GRAPH_CAPTURE_PREREGISTRATION_2026-08-15.md"
        payload["environment"] = environment_snapshot((runtime_file, prereg_file))

        rt = _new_runtime(capacity)
        payload["runtime_loaded_utc"] = utc_now()

        # EGR: same device-resident route/cache path, eager submission.
        eager_ids: dict[str, list[int]] = {}
        eager_samples: list[float] = []
        for prompt in prompts:
            ids, samples = _run_eager_timed(rt, prompt["prompt_ids"], n)
            eager_ids[prompt["prompt"]] = ids
            eager_samples.extend(samples)
        payload["arms"] = {
            "EGR": {
                "prompts": [_prompt_result(p, eager_ids[p["prompt"]]) for p in prompts],
                "timing_ms": percentiles(eager_samples),
                "raw_timing_ms": eager_samples,
            }
        }

        # Rebuild cache before capture so graph pointers/state are clean and frozen.
        rt.enable_cache(capacity)
        rt.device_cache = True
        rt.deterministic_accum = True
        import cupy as cp

        free_before = int(cp.cuda.Device(0).mem_info[0])
        rt.setup_graph()
        free_after = int(cp.cuda.Device(0).mem_info[0])
        graph_extra = int(getattr(rt, "graph_extra_vram_bytes", free_before - free_after))

        graph_ids: dict[str, list[int]] = {}
        graph_samples: list[float] = []
        for prompt in prompts:
            ids, samples = _run_graph_timed(rt, prompt["prompt_ids"], n)
            graph_ids[prompt["prompt"]] = ids
            graph_samples.extend(samples)
        payload["arms"]["GRAPH"] = {
            "prompts": [_prompt_result(p, graph_ids[p["prompt"]]) for p in prompts],
            "timing_ms": percentiles(graph_samples),
            "raw_timing_ms": graph_samples,
            "graph_extra_vram_bytes": graph_extra,
        }

        # DET: repeat each graph rollout twice. This is deliberately done after
        # timing so it cannot warm the timed arm.
        det: dict[str, Any] = {}
        det_n = n if args.mode == "full" else min(n, 16)
        for prompt in prompts:
            a = _run_graph_collect(rt, prompt["prompt_ids"], det_n)
            b = _run_graph_collect(rt, prompt["prompt_ids"], det_n)
            det[prompt["prompt"]] = {
                "identical": a == b,
                "first_divergence": first_divergence(a, b),
                "ids_a": a,
                "ids_b": b,
            }
        payload["arms"]["DET"] = det

        ctl: dict[str, Any] | None = None
        if not args.skip_control:
            # Re-capture with the deliberate 7th-expert sabotage baked into the
            # kernel argument. No gate is weakened if this fails to diverge.
            rt._bad_pick = 1
            rt.enable_cache(capacity)
            rt.device_cache = True
            rt.setup_graph()
            ctl_n = min(n, 64)
            ctl = {}
            for prompt in prompts:
                ids = _run_graph_collect(rt, prompt["prompt_ids"], ctl_n)
                ref = eager_ids[prompt["prompt"]][:ctl_n]
                ctl[prompt["prompt"]] = {
                    "ids": ids,
                    "reference_ids": ref,
                    "identical": ids == ref,
                    "first_divergence": first_divergence(ids, ref),
                }
            payload["arms"]["CTL"] = ctl
            rt._bad_pick = 0

        eager_p50 = payload["arms"]["EGR"]["timing_ms"]["p50"]
        graph_p50 = payload["arms"]["GRAPH"]["timing_ms"]["p50"]
        graph_vs_eager = {
            p["prompt"]: {
                "identical": graph_ids[p["prompt"]] == eager_ids[p["prompt"]],
                "first_divergence": first_divergence(graph_ids[p["prompt"]], eager_ids[p["prompt"]]),
            }
            for p in prompts
        }
        anchor_first64 = {
            p["prompt"]: (
                graph_ids[p["prompt"]][:64] == expected[p["prompt"]][:64]
                if p["kind"] == "anchor" and p["prompt"] in expected and len(graph_ids[p["prompt"]]) >= 64
                else None
            )
            for p in prompts
        }
        control_diverged = None if ctl is None else any(not x["identical"] for x in ctl.values())
        gates = {
            "G_E1F22_PAR": {
                "passed": all(v["identical"] for v in graph_vs_eager.values())
                and all(v is not False for v in anchor_first64.values()),
                "graph_vs_eager": graph_vs_eager,
                "anchor_first64": anchor_first64,
            },
            "G_E1F22_CTL": {
                "passed": control_diverged if control_diverged is not None else None,
                "required": "bad_pick graph diverges on at least one prompt",
            },
            "G_E1F22_DET": {
                "passed": all(x["identical"] for x in det.values()),
            },
            "G_E1F22_S1": {
                "passed": bool(graph_p50 is not None and eager_p50 is not None and graph_p50 <= eager_p50 - 2.5),
                "eager_p50_ms": eager_p50,
                "graph_p50_ms": graph_p50,
                "gain_ms": None if eager_p50 is None or graph_p50 is None else float(eager_p50 - graph_p50),
                "required_gain_ms": 2.5,
                "timed_samples": len(graph_samples),
                "sample_gate_ge_500": len(graph_samples) >= 500 if args.mode == "full" else None,
            },
            "G_E1F22_VRAM": {
                "passed": graph_extra < 64 * 1024 * 1024,
                "extra_bytes": graph_extra,
                "limit_bytes": 64 * 1024 * 1024,
            },
        }
        payload["gates"] = gates
        if args.mode == "smoke":
            # Smoke is a technical/correctness check, not a speed claim. Sixteen
            # tokens cannot adjudicate the frozen >=500-sample S1 performance gate.
            # Keep S1 in the JSON as diagnostic data, but do not let it turn a
            # working graph into a false smoke failure.
            smoke_mandatory = [
                bool(gates["G_E1F22_PAR"]["passed"]),
                bool(gates["G_E1F22_DET"]["passed"]),
                bool(gates["G_E1F22_VRAM"]["passed"]),
            ]
            if not args.skip_control:
                smoke_mandatory.append(bool(gates["G_E1F22_CTL"]["passed"]))
            payload["status"] = "smoke_pass" if all(smoke_mandatory) else "smoke_gate_failed"
        else:
            mandatory = [
                bool(gates["G_E1F22_PAR"]["passed"]),
                bool(gates["G_E1F22_DET"]["passed"]),
                bool(gates["G_E1F22_S1"]["passed"]),
                bool(gates["G_E1F22_VRAM"]["passed"]),
            ]
            if not args.skip_control:
                mandatory.append(bool(gates["G_E1F22_CTL"]["passed"]))
            payload["status"] = "pass" if all(mandatory) else "gate_failed"
        payload["completed_utc"] = utc_now()
        payload["summary"] = {
            "eager_tok_s_from_p50": None if not eager_p50 else float(1000.0 / eager_p50),
            "graph_tok_s_from_p50": None if not graph_p50 else float(1000.0 / graph_p50),
            "gain_ms": gates["G_E1F22_S1"]["gain_ms"],
        }

        del rt
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    except Exception as exc:
        payload["status"] = "technical_failure"
        payload["completed_utc"] = utc_now()
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    write_json_atomic(OUT, payload)
    console = {"status": payload["status"], "output": str(OUT)}
    if "gates" in payload:
        console["gates"] = {
            name: gate.get("passed") for name, gate in payload["gates"].items()
        }
        s1 = payload["gates"].get("G_E1F22_S1", {})
        console["timing"] = {
            "eager_p50_ms": s1.get("eager_p50_ms"),
            "graph_p50_ms": s1.get("graph_p50_ms"),
            "gain_ms": s1.get("gain_ms"),
        }
    print(json.dumps(console, indent=2))
    return 0 if payload["status"] in {"pass", "gate_failed", "smoke_pass", "smoke_gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
