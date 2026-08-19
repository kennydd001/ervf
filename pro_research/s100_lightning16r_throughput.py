from __future__ import annotations

import gc
import hashlib
import json
import time
import traceback

import numpy as np

from common import require_model_dir, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    assert_lightning,
    load_trace,
    normalize_eager_moe,
    prompt_manifest,
)
from s100_lightning16r_common import (
    RESULTS,
    canonical_cases,
    ensure_results,
    load_json,
)
from s100_lightning16r_native import PointerDispatch

OUT = RESULTS / "S100_LIGHTNING16R_THROUGHPUT.json"
WARMUP_TARGETS = 8
MIN_SPEEDUP = 1.03

def metrics(samples: list[float]) -> dict:
    values = np.asarray(samples, np.float64)
    if values.size == 0:
        raise RuntimeError("no timing samples")
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p10_ms": float(np.percentile(values, 10)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "aggregate_tok_s": float(
            1000.0 * values.size / values.sum()
        ),
        "tok_s_from_p50": float(
            1000.0 / np.percentile(values, 50)
        ),
        "raw_ms": [float(value) for value in values],
    }

def frozen_workload():
    from transformers import AutoTokenizer

    metadata, trace = load_trace("calibration")
    manifest = prompt_manifest()
    tokenizer = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    rows = []
    for index, record in enumerate(
        metadata["prompt_records"]
    ):
        source = manifest[record["id"]]
        prompt_ids = [
            int(value) for value in tokenizer.encode(
                source["prompt"],
                add_special_tokens=False,
            )
        ]
        digest = hashlib.sha256(
            np.asarray(prompt_ids, dtype="<i4").tobytes()
        ).hexdigest()
        if digest != record["prompt_ids_sha256"]:
            raise RuntimeError(
                f"tokenizer drift: {record['id']}"
            )
        targets = [
            int(value)
            for value in trace["target_ids"][index]
        ]
        if len(targets) <= WARMUP_TARGETS:
            raise RuntimeError(
                f"trace too short: {record['id']}"
            )
        rows.append({
            "id": record["id"],
            "domain": record["domain"],
            "prompt_ids": prompt_ids,
            "target_ids": targets,
        })
    return rows

def cleanup(bundle) -> None:
    import cupy as cp

    if bundle is not None:
        try:
            bundle.restore_combined()
            bundle.restore_sel()
        except Exception:
            pass
        # Break Bundle-held runtime/closure references before emptying pools.
        for attribute in (
            "rt",
            "restore_combined",
            "restore_sel",
            "state",
            "panel_cache",
        ):
            try:
                setattr(bundle, attribute, None)
            except Exception:
                pass
    bundle = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        cp.cuda.Device(0).synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()

def graph_arm(workload, label: str) -> dict:
    bundle = None
    runtime = None
    try:
        bundle = build()
        runtime = bundle.rt
        samples = []
        per_prompt = []
        for row in workload:
            runtime.reset()
            start = int(runtime._ring_i)
            for token in row["prompt_ids"]:
                runtime.step_graph(int(token))
            last_slot = (
                start + len(row["prompt_ids"]) - 1
            ) % int(runtime._ring_size)
            runtime.ring_harvest(last_slot, 1)

            prompt_samples = []
            for index, token in enumerate(
                row["target_ids"]
            ):
                slot = int(runtime._ring_i)
                if index < WARMUP_TARGETS:
                    runtime.step_graph(int(token))
                    runtime.ring_harvest(slot, 1)
                    continue
                started = time.perf_counter_ns()
                runtime.step_graph(int(token))
                runtime.ring_harvest(slot, 1)
                elapsed = (
                    time.perf_counter_ns() - started
                ) / 1e6
                samples.append(elapsed)
                prompt_samples.append(elapsed)
            per_prompt.append({
                "id": row["id"],
                "domain": row["domain"],
                "timing": metrics(prompt_samples),
            })
        return {
            "label": label,
            "kind": "production_cuda_graph",
            "timing": metrics(samples),
            "per_prompt": per_prompt,
        }
    finally:
        runtime = None
        cleanup(bundle)
        bundle = None

def eager_arm(
    workload,
    *,
    label: str,
    candidate: dict | None,
) -> dict:
    import cupy as cp

    bundle = None
    runtime = None
    dispatch = None
    try:
        bundle = build()
        runtime = bundle.rt
        runtime._graph = None
        runtime.graph_mode = False
        normalize_eager_moe(runtime)

        dispatch = None
        if candidate is not None:
            dispatch = PointerDispatch(
                runtime,
                terms=int(candidate["terms"]),
                handoff=str(candidate["handoff"]),
                enabled_cases=set(candidate["cases"]),
            ).install()

        samples = []
        per_prompt = []
        native_calls = 0
        original_calls = 0
        sync_calls = 0
        paired_elisions = 0

        for row in workload:
            runtime.reset()
            if dispatch is not None:
                dispatch.last_native_case = None
            for token in row["prompt_ids"]:
                runtime.step(int(token))
            cp.cuda.Device(0).synchronize()

            for index, token in enumerate(
                row["target_ids"][:WARMUP_TARGETS]
            ):
                runtime.step(int(token))
                cp.cuda.Device(0).synchronize()

            if dispatch is not None:
                dispatch.reset_counters()
                dispatch.last_native_case = None

            prompt_samples = []
            for token in row["target_ids"][
                WARMUP_TARGETS:
            ]:
                started = time.perf_counter_ns()
                runtime.step(int(token))
                cp.cuda.Device(0).synchronize()
                elapsed = (
                    time.perf_counter_ns() - started
                ) / 1e6
                samples.append(elapsed)
                prompt_samples.append(elapsed)

            if dispatch is not None:
                native_calls += dispatch.native_calls
                original_calls += dispatch.original_calls
                sync_calls += dispatch.sync_calls
                paired_elisions += (
                    dispatch.paired_sync_elisions
                )

            per_prompt.append({
                "id": row["id"],
                "domain": row["domain"],
                "timing": metrics(prompt_samples),
            })

        result = {
            "label": label,
            "kind": (
                "eager_parent"
                if candidate is None
                else "selective_native_eager"
            ),
            "timing": metrics(samples),
            "per_prompt": per_prompt,
        }
        if candidate is not None:
            measured_tokens = len(samples)
            result.update({
                "candidate": candidate,
                "dispatch": {
                    "native_calls": native_calls,
                    "original_calls": original_calls,
                    "sync_calls": sync_calls,
                    "paired_sync_elisions": paired_elisions,
                    "native_calls_per_measured_token": (
                        native_calls
                        / max(measured_tokens, 1)
                    ),
                    "sync_calls_per_measured_token": (
                        sync_calls
                        / max(measured_tokens, 1)
                    ),
                },
            })
        return result
    finally:
        dispatch = None
        runtime = None
        cleanup(bundle)
        bundle = None

def heldout_candidates() -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(
        RESULTS.glob(
            "S100_LIGHTNING16R_QUALITY_*_HELDOUT.json"
        )
    ):
        payload = load_json(path)
        if not payload or payload.get("status") != "measured":
            continue
        deterministic = bool(
            (payload.get("summary") or {}).get(
                "deterministic_anchor_repeat"
            )
        )
        if not payload.get("official_pass") or not deterministic:
            continue
        signature = str(payload["candidate_signature"])
        if signature in seen:
            continue
        seen.add(signature)
        rows.append({
            "name": str(payload["name"]),
            "terms": int(payload["terms"]),
            "handoff": str(payload["handoff"]),
            "cases": canonical_cases(payload["cases"]),
            "candidate_signature": signature,
            "heldout_quality_path": str(path),
            "heldout_summary": payload.get("summary"),
            "heldout_per_domain": payload.get(
                "per_domain"
            ),
        })
    return rows

def main() -> int:
    ensure_results()
    payload = {
        "kind": "s100_lightning16r_throughput",
        "status": "started",
        "warmup_targets_per_prompt": WARMUP_TARGETS,
        "minimum_speedup": MIN_SPEEDUP,
        "started_utc": utc_now(),
        "claim_boundary": (
            "fixed-token teacher-forced end-to-end decode "
            "comparison; includes Python, synchronization, "
            "routing, cache, state and LM-head costs"
        ),
    }
    try:
        identity = assert_lightning()
        workload = frozen_workload()
        candidates = heldout_candidates()

        graph_a = graph_arm(workload, "graph_A")
        eager_parent = eager_arm(
            workload,
            label="eager_parent",
            candidate=None,
        )

        candidate_arms = []
        for candidate in candidates:
            arm = eager_arm(
                workload,
                label=candidate["name"],
                candidate=candidate,
            )
            candidate_arms.append(arm)

        graph_b = graph_arm(workload, "graph_B")
        graph_samples = (
            graph_a["timing"]["raw_ms"]
            + graph_b["timing"]["raw_ms"]
        )
        graph_reference = metrics(graph_samples)
        graph_reference["construction"] = (
            "combined A/B bracket samples"
        )

        comparisons = []
        for arm in candidate_arms:
            candidate_timing = arm["timing"]
            aggregate_speedup = (
                candidate_timing["aggregate_tok_s"]
                / graph_reference["aggregate_tok_s"]
            )
            p50_speedup = (
                graph_reference["p50_ms"]
                / candidate_timing["p50_ms"]
            )
            speed_open = bool(
                aggregate_speedup >= MIN_SPEEDUP
                and p50_speedup >= MIN_SPEEDUP
                and candidate_timing["samples"] >= 500
            )
            s100 = bool(
                candidate_timing["aggregate_tok_s"]
                >= 100.0
            )
            comparisons.append({
                "name": arm["candidate"]["name"],
                "candidate_signature": arm[
                    "candidate"
                ]["candidate_signature"],
                "aggregate_speedup_vs_graph": float(
                    aggregate_speedup
                ),
                "p50_speedup_vs_graph": float(
                    p50_speedup
                ),
                "SELECTIVE_NATIVE_NET_SPEEDUP_OPEN": (
                    speed_open
                ),
                "S100_SINGLE_CANDIDATE_OPEN": s100,
            })

        payload.update({
            "status": "measured",
            "identity": identity,
            "workload": {
                "prompts": len(workload),
                "target_tokens_per_prompt": len(
                    workload[0]["target_ids"]
                ),
                "warmup_targets_per_prompt": (
                    WARMUP_TARGETS
                ),
                "measured_tokens_per_arm": sum(
                    len(row["target_ids"])
                    - WARMUP_TARGETS
                    for row in workload
                ),
                "prompt_ids": [
                    row["id"] for row in workload
                ],
            },
            "heldout_green_candidates": candidates,
            "graph_A": graph_a,
            "eager_parent": eager_parent,
            "candidate_arms": candidate_arms,
            "graph_B": graph_b,
            "graph_reference": graph_reference,
            "comparisons": comparisons,
            "ANY_SELECTIVE_NATIVE_NET_SPEEDUP_OPEN": any(
                row[
                    "SELECTIVE_NATIVE_NET_SPEEDUP_OPEN"
                ]
                for row in comparisons
            ),
            "S100_SINGLE_ACHIEVED": any(
                row["S100_SINGLE_CANDIDATE_OPEN"]
                for row in comparisons
            ),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "graph_reference": (
            payload.get("graph_reference") or {}
        ).get("aggregate_tok_s"),
        "comparisons": payload.get("comparisons"),
        "ANY_SELECTIVE_NATIVE_NET_SPEEDUP_OPEN": (
            payload.get(
                "ANY_SELECTIVE_NATIVE_NET_SPEEDUP_OPEN"
            )
        ),
        "S100_SINGLE_ACHIEVED": payload.get(
            "S100_SINGLE_ACHIEVED"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
