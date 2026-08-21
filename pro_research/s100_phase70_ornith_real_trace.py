"""Phase70 analysis of real Ornith routes and final normalized activations."""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase35_c3c_quantizer import FusedStaticNVFP4Quantizer
from s100_phase48_ornith_swiglu_h8 import _load_projection
from s100_phase64_ornith_shortlist_kernel import ExactERVFShortlist


RESULTS = REPO / "pro_research" / "results" / "s100_phase70"
PREREG = REPO / "pro_research" / "S100_PHASE70_ORNITH_REAL_TRACE_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase70_ornith_real_trace.py"
RUNNER = REPO / "pro_research" / "llama_ornith_trace.cpp"
TRACE_MODULE = REPO / "src" / "moe_lab" / "ornith" / "trace_analysis.py"
PHASE64 = REPO / "pro_research" / "results" / "s100_phase64" / "S100_PHASE64_ORNITH_NATIVE_SHORTLIST_HEAD.json"
SHORTLIST = 64
REFERENCE_TOP = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trace(path: Path):
    import sys

    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from moe_lab.ornith.trace_analysis import parse_llama_trace

    payload = json.loads(path.read_text("utf-8"))
    return payload, parse_llama_trace(payload)


def _trace_quality(trace, repeat) -> dict[str, Any]:
    route_equal = trace.routes == repeat.routes
    token_equal = trace.tokens == repeat.tokens
    weights_a = np.asarray([
        value for layer in range(40) for row in trace.weights[layer] for value in row
    ], dtype=np.float32)
    weights_b = np.asarray([
        value for layer in range(40) for row in repeat.weights[layer] for value in row
    ], dtype=np.float32)
    norm_a = np.asarray(trace.result_norm, dtype=np.float32)
    norm_b = np.asarray(repeat.result_norm, dtype=np.float32)
    sums = np.asarray([
        sum(row) for layer in range(40) for row in trace.weights[layer]
    ], dtype=np.float64)
    return {
        "tokens_equal": token_equal,
        "routes_exact": route_equal,
        "route_layers": len(trace.routes),
        "route_shape": [8, len(trace.tokens), 1, 1],
        "weight_layers": len(trace.weights),
        "weights_finite": bool(np.isfinite(weights_a).all()),
        "weight_sum_max_abs_from_one": float(np.max(np.abs(sums - 1.0))),
        "weights_repeat_max_abs": float(np.max(np.abs(weights_a - weights_b))),
        "result_norm_shape": list(norm_a.T.shape),
        "result_norm_finite": bool(np.isfinite(norm_a).all()),
        "result_norm_repeat_max_abs": float(np.max(np.abs(norm_a - norm_b))),
    }


def _head_replay(trace, snapshot: Path) -> dict[str, Any]:
    import cupy as cp
    import sys
    import torch
    import torch.nn.functional as F
    import native_nvfp4_c3a_layout_v2 as layout_v2
    import native_nvfp4_c3a_lib as c3lib
    from diag_native_nvfp4_c3b_realact import native_call

    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

    if len(trace.result_norm) != 4:
        raise ValueError(f"expected four result_norm rows, got {len(trace.result_norm)}")
    layout_v2.install(c3lib)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
    head = _load_projection(snapshot, index["weight_map"], "lm_head")
    rows, cols = 248320, 2048
    x_host = np.asarray(trace.result_norm, dtype=np.float32)
    if x_host.shape != (4, cols):
        raise ValueError(f"expected activation shape (4, {cols}), got {x_host.shape}")

    fused = FusedNVFP4()
    exact = ExactERVFShortlist()
    quantizer = FusedStaticNVFP4Quantizer(cols, 4)
    codes = cp.asarray(head["codes"])
    scales = cp.asarray(head["scales"])
    x = cp.asarray(x_host)
    control = cp.empty((4, rows), dtype=cp.float32)
    for token in range(4):
        fused.gemv_into(control[token], codes, scales, x[token], head["global_scale"], rows, cols)

    b = c3lib.make_b(
        torch, head["codes"].tobytes(), head["scales"].tobytes(),
        head["global_scale"], rows, cols,
    )
    tensor_scale = float(np.max(np.abs(x_host)) * 1.10 / (448.0 * 6.0))
    quantizer.quantize(x, tensor_scale)
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    external = torch.cuda.ExternalStream(stream.ptr)
    packed_t = torch.utils.dlpack.from_dlpack(quantizer.packed)
    blocked_t = torch.utils.dlpack.from_dlpack(quantizer.blocked_scales)
    a = {
        "fp4": packed_t.view(torch.float4_e2m1fn_x2),
        "block": blocked_t.view(torch.float8_e4m3fn),
        "global": torch.tensor([tensor_scale], dtype=torch.float32, device="cuda"),
    }
    with torch.cuda.stream(external):
        native = native_call(torch, F, F.ScalingType, F.SwizzleType, a, b)
        _, ids_t = torch.topk(native, SHORTLIST, dim=1)
    external.synchronize()
    ids = cp.from_dlpack(ids_t)
    rerank = cp.empty((4, SHORTLIST), dtype=cp.float32)
    exact(
        codes, scales, fused.e2m1, fused.e4m3, x,
        ids.data.ptr, rerank.data.ptr, head["global_scale"], SHORTLIST, cols,
    )
    stream.synchronize()

    control_host = cp.asnumpy(control)
    ids_host = cp.asnumpy(ids).astype(np.int64)
    rerank_host = cp.asnumpy(rerank)
    gathered = np.take_along_axis(control_host, ids_host, axis=1)
    reference_ids = np.argsort(-control_host, axis=1, kind="stable")[:, :REFERENCE_TOP]
    candidate_order = np.argsort(-rerank_host, axis=1, kind="stable")[:, :REFERENCE_TOP]
    candidate_ids = np.take_along_axis(ids_host, candidate_order, axis=1)
    recalls = [
        len(set(reference_ids[token]) & set(ids_host[token])) / REFERENCE_TOP
        for token in range(4)
    ]
    exact_sets = [
        set(reference_ids[token]) == set(candidate_ids[token]) for token in range(4)
    ]
    exact_order = [
        bool(np.array_equal(reference_ids[token], candidate_ids[token])) for token in range(4)
    ]
    error = rerank_host.astype(np.float64) - gathered.astype(np.float64)
    denominator = max(float(np.sqrt(np.mean(gathered.astype(np.float64) ** 2))), 1.0e-12)
    return {
        "activation_tensor_scale": tensor_scale,
        "activation_max_abs": float(np.max(np.abs(x_host))),
        "reference_top": REFERENCE_TOP,
        "shortlist": SHORTLIST,
        "recall_by_token": recalls,
        "top32_recall": float(np.mean(recalls)),
        "candidate_exact_set_by_token": exact_sets,
        "candidate_exact_order_by_token": exact_order,
        "reference_top1": reference_ids[:, 0].tolist(),
        "native_top1": torch.argmax(native, dim=1).cpu().numpy().astype(np.int64).tolist(),
        "reranked_top1": candidate_ids[:, 0].tolist(),
        "score_bit_exact": bool(np.array_equal(gathered.view(np.uint32), rerank_host.view(np.uint32))),
        "score_nrmse": float(np.sqrt(np.mean(error ** 2)) / denominator),
    }


def _cache_analysis(trace, long_trace=None) -> dict[str, Any]:
    import sys

    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from moe_lab.ornith.trace_analysis import replay_trace, summarize_h4_miss_groups

    def compact(replay):
        return {
            "policy": replay["policy"],
            "slots_per_layer": replay["slots_per_layer"],
            "tokens": replay["tokens"],
            "summary": replay["summary"],
            "layers": {
                layer: {
                    key: row[key]
                    for key in (
                        "assignments", "hits", "misses", "hit_rate", "miss_rate",
                        "unique_miss_experts", "evictions",
                    )
                }
                for layer, row in replay["layers"].items()
            },
        }

    result = {"h4": {}}
    for label, policy in (("lru52", "lru"), ("belady52", "belady")):
        result["h4"][label] = compact(replay_trace(trace, slots=52, policy=policy))
    if long_trace is not None:
        result["long"] = {}
        for label, policy in (("lru52", "lru"), ("belady52", "belady")):
            replay = replay_trace(long_trace, slots=52, policy=policy)
            h4_miss_groups = summarize_h4_miss_groups(replay)
            result["long"][label] = compact(replay)
            result["long"][label]["h4_miss_groups"] = h4_miss_groups
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--repeat-trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--long-trace", type=Path)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE70_ORNITH_REAL_TRACE.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase70_ornith_real_trace",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        trace_path = args.trace.resolve()
        repeat_path = args.repeat_trace.resolve()
        _, trace = _load_trace(trace_path)
        _, repeat = _load_trace(repeat_path)
        long_trace = None
        if args.long_trace:
            _, long_trace = _load_trace(args.long_trace.resolve())
        quality = _trace_quality(trace, repeat)
        cache = _cache_analysis(trace, long_trace)
        head = _head_replay(trace, args.snapshot.resolve())
        phase64 = json.loads(PHASE64.read_text("utf-8"))
        previous_latency = phase64["summary"]["candidate_h4_ms"]
        gates = {
            "P70_G1_40_top8_h4_route_tensors": (
                quality["route_layers"] == 40 and quality["route_shape"] == [8, 4, 1, 1]
            ),
            "P70_G2_route_weights_normalized": (
                quality["weight_layers"] == 40 and quality["weights_finite"]
                and quality["weight_sum_max_abs_from_one"] <= 5e-4
            ),
            "P70_G3_result_norm_h4_repeat": (
                quality["result_norm_shape"] == [2048, 4]
                and quality["result_norm_finite"]
                and quality["result_norm_repeat_max_abs"] <= 1e-5
                and quality["routes_exact"]
            ),
            "P70_G4_real_activation_top32_recall": (
                all(value == 1.0 for value in head["recall_by_token"])
                and all(head["candidate_exact_set_by_token"])
                and head["reference_top1"] == head["reranked_top1"]
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "inputs": {
                "trace": str(trace_path),
                "trace_sha256": _sha256(trace_path),
                "repeat_trace": str(repeat_path),
                "repeat_trace_sha256": _sha256(repeat_path),
                "long_trace": str(args.long_trace.resolve()) if args.long_trace else None,
                "snapshot": str(args.snapshot.resolve()),
            },
            "trace_quality": quality,
            "cache": cache,
            "head": head,
            "phase64_measured_candidate_h4_ms": previous_latency,
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as error:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, RUNNER, TRACE_MODULE, PHASE64))
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "trace_quality": payload.get("trace_quality"),
        "head": payload.get("head"),
        "h4_cache_summary": {
            key: value["summary"]
            for key, value in (((payload.get("cache") or {}).get("h4") or {}).items())
        },
        "long_cache_summary": {
            key: {
                **value["summary"],
                "warm_h4_miss_groups": value["h4_miss_groups"]["warm"],
            }
            for key, value in (((payload.get("cache") or {}).get("long") or {}).items())
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
