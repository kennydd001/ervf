"""Phase58 benchmark of direct-L2 FP8 H4 on real Ornith projections."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase58_ornith_fp8_h4_kernels import (
    OrnithFP8H4Kernels,
    decode_e4m3_host,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase58"
PREREG = REPO / "pro_research" / "S100_PHASE58_ORNITH_FP8_H4_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase58_ornith_fp8_h4.py"
KERNELS = REPO / "pro_research" / "s100_phase58_ornith_fp8_h4_kernels.py"


MATRICES = (
    ("official", "linear_qkv", "model.language_model.layers.20.linear_attn.in_proj_qkv.weight"),
    ("official", "linear_z", "model.language_model.layers.20.linear_attn.in_proj_z.weight"),
    ("official", "linear_out", "model.language_model.layers.20.linear_attn.out_proj.weight"),
    ("pottokao", "linear_qkv", "model.layers.20.linear_attn.in_proj_qkv.weight"),
    ("pottokao", "linear_z", "model.layers.20.linear_attn.in_proj_z.weight"),
    ("pottokao", "linear_out", "model.layers.20.linear_attn.out_proj.weight"),
    ("pottokao", "full_q", "model.layers.23.self_attn.q_proj.weight"),
    ("pottokao", "full_k", "model.layers.23.self_attn.k_proj.weight"),
    ("pottokao", "full_v", "model.layers.23.self_attn.v_proj.weight"),
    ("pottokao", "full_out", "model.layers.23.self_attn.o_proj.weight"),
)


def _weight_map(repo: Path) -> dict[str, str]:
    index = json.loads((repo / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return index["weight_map"]


def _load_fp8(repo: Path, weight_map: dict[str, str], name: str):
    import torch
    from safetensors import safe_open

    shard = repo / weight_map[name]
    scale_name = name.removesuffix(".weight") + ".weight_scale"
    input_scale_name = name.removesuffix(".weight") + ".input_scale"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name).contiguous()
        if tensor.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name}: expected float8_e4m3fn, got {tensor.dtype}")
        raw = tensor.view(torch.uint8).numpy().copy()
        weight_scale = float(handle.get_tensor(scale_name).item())
        input_scale = float(handle.get_tensor(input_scale_name).item())
    return raw, weight_scale, input_scale, shard.name


def _time_gpu(cp, function, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        function()
    cp.cuda.get_current_stream().synchronize()
    samples = []
    for _ in range(repeats):
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        begin.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(begin, end)))
    return statistics.median(samples), samples


def _decode_gate() -> dict[str, Any]:
    import torch

    raw = np.arange(256, dtype=np.uint8)
    ours = decode_e4m3_host(raw)
    reference = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).float().numpy()
    finite = np.isfinite(reference)
    exact = bool(np.array_equal(ours[finite].view(np.uint32), reference[finite].view(np.uint32)))
    nan_exact = bool(np.array_equal(np.isnan(ours), np.isnan(reference)))
    return {
        "finite_count": int(finite.sum()),
        "nan_count": int((~finite).sum()),
        "finite_bit_exact": exact,
        "nan_mask_exact": nan_exact,
        "max_abs": float(np.max(np.abs(ours[finite] - reference[finite]))),
    }


def _bench_matrix(
    kernels: OrnithFP8H4Kernels,
    repo: Path,
    weight_map: dict[str, str],
    repository: str,
    label: str,
    name: str,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import cupy as cp
    import torch

    raw, weight_scale, input_scale, shard = _load_fp8(repo, weight_map, name)
    rows, cols = map(int, raw.shape)
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(0.0, 0.5, size=(4, cols)).astype(np.float32))
    x_raw = x.to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
    weight_gpu = cp.asarray(raw)
    x_gpu = cp.asarray(x_raw)
    out_m1 = cp.empty((4, rows), dtype=cp.float32)
    out_m4 = cp.empty((4, rows), dtype=cp.float32)
    scale = float(weight_scale * input_scale)

    def m1_x4() -> None:
        for index in range(4):
            kernels.m1(weight_gpu, x_gpu[index], out_m1[index], rows, cols, scale)

    def m4() -> None:
        kernels.m4(weight_gpu, x_gpu, out_m4, rows, cols, scale)

    m1_x4()
    m4()
    cp.cuda.get_current_stream().synchronize()
    m1_host = cp.asnumpy(out_m1)
    m4_host = cp.asnumpy(out_m4)
    repeat_before = m4_host.copy()
    m4()
    cp.cuda.get_current_stream().synchronize()
    repeat_after = cp.asnumpy(out_m4)
    m1_ms, m1_samples = _time_gpu(cp, m1_x4, warmup, repeats)
    m4_ms, m4_samples = _time_gpu(cp, m4, warmup, repeats)
    result = {
        "repository": repository,
        "label": label,
        "tensor": name,
        "shard": shard,
        "shape": [rows, cols],
        "weight_bytes": int(raw.nbytes),
        "weight_scale": weight_scale,
        "input_scale": input_scale,
        "combined_scale": scale,
        "m1_x4_median_ms": m1_ms,
        "m4_median_ms": m4_ms,
        "speedup": m1_ms / m4_ms,
        "m1_x4_samples_ms": m1_samples,
        "m4_samples_ms": m4_samples,
        "m4_vs_m1_bit_exact": bool(np.array_equal(m1_host.view(np.uint32), m4_host.view(np.uint32))),
        "m4_repeat_bit_exact": bool(
            np.array_equal(repeat_before.view(np.uint32), repeat_after.view(np.uint32))
        ),
        "all_finite": bool(np.isfinite(m4_host).all()),
        "max_abs_m4_vs_m1": float(np.max(np.abs(m4_host - m1_host))),
    }
    del weight_gpu, x_gpu, out_m1, out_m4
    cp.get_default_memory_pool().free_all_blocks()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--pottokao", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=41)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE58_ORNITH_FP8_H4.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase58_ornith_fp8_h4",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
    }
    try:
        for path in (args.official, args.pottokao):
            if not path.is_dir():
                raise FileNotFoundError(path)
        repositories = {
            "official": args.official.resolve(),
            "pottokao": args.pottokao.resolve(),
        }
        maps = {name: _weight_map(path) for name, path in repositories.items()}
        decode = _decode_gate()
        kernels = OrnithFP8H4Kernels()
        resources = kernels.resource_audit()
        records = []
        for index, (repository, label, tensor) in enumerate(MATRICES):
            records.append(_bench_matrix(
                kernels,
                repositories[repository],
                maps[repository],
                repository,
                label,
                tensor,
                seed=5800 + index,
                warmup=args.warmup,
                repeats=args.repeats,
            ))
        speedups = [record["speedup"] for record in records]
        by_key = {(record["repository"], record["label"]): record for record in records}
        parity = {
            label: (
                by_key[("official", label)]["m4_median_ms"]
                / by_key[("pottokao", label)]["m4_median_ms"]
            )
            for label in ("linear_qkv", "linear_z", "linear_out")
        }
        m4_resource = resources["fp8_e4m3_m4_direct_l2"]
        gates = {
            "P58_G1_e4m3_decoder_matches_torch": (
                decode["finite_bit_exact"] and decode["nan_mask_exact"]
            ),
            "P58_G2_all_m4_vs_m1_bit_exact": all(
                record["m4_vs_m1_bit_exact"] for record in records
            ),
            "P58_G3_all_finite_and_repeat_exact": all(
                record["all_finite"] and record["m4_repeat_bit_exact"]
                for record in records
            ),
            "P58_G4_resource_budget": (
                all((resource.get("local_size_bytes") or 0) == 0 for resource in resources.values())
                and (m4_resource.get("num_regs") or 10_000) <= 64
            ),
            "P58_G5_all_positive_and_median_speedup_ge_2": (
                all(speedup > 1.0 for speedup in speedups)
                and statistics.median(speedups) >= 2.0
            ),
            "P58_G6_official_pottokao_latency_parity": all(
                0.8 <= ratio <= 1.25 for ratio in parity.values()
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "repositories": {name: str(path) for name, path in repositories.items()},
            "decode_gate": decode,
            "resources": resources,
            "records": records,
            "summary": {
                "median_speedup": statistics.median(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups),
                "official_pottokao_m4_latency_ratio": parity,
            },
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNELS))
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "decode": payload.get("decode_gate"),
        "resources": payload.get("resources"),
        "summary": payload.get("summary"),
        "records": [
            {
                "repository": record["repository"],
                "label": record["label"],
                "shape": record["shape"],
                "m1_x4_ms": record["m1_x4_median_ms"],
                "m4_ms": record["m4_median_ms"],
                "speedup": record["speedup"],
                "exact": record["m4_vs_m1_bit_exact"],
            }
            for record in payload.get("records", [])
        ],
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
