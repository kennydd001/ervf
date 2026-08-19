from __future__ import annotations

from dataclasses import dataclass
import json
import statistics
import sys
import traceback

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_phase14r_common import RESULTS, ensure_results

OUT = RESULTS / "S100_PHASE14R_NATIVE_ZERO_COPY.json"
BLOCKS = (2, 4, 8)

@dataclass
class Case:
    name: str
    family: str
    weight: object
    rows: int
    cols: int
    weight_bytes: int

def collect(rt):
    cases = []
    for layer in rt.mamba_layers:
        d = rt.layer[int(layer)]
        if d.get("in_k") == "bf16":
            cases.append(Case(
                f"mamba_{layer}_in", "mamba", d["in_w"],
                int(rt.proj.size), int(rt.hidden), int(d["in_w"].nbytes),
            ))
        if d.get("out_k") == "bf16":
            cases.append(Case(
                f"mamba_{layer}_out", "mamba", d["out_w"],
                int(rt.hidden), int(rt.d_inner), int(d["out_w"].nbytes),
            ))

    for layer in rt.attn_layers:
        d = rt.layer[int(layer)]
        hq = int(rt.n_heads * rt.head_dim)
        for side, key, rows, cols in (
            ("q", "q_proj", hq, int(rt.hidden)),
            ("k", "k_proj", int(rt.kv_dim), int(rt.hidden)),
            ("v", "v_proj", int(rt.kv_dim), int(rt.hidden)),
            ("o", "o_proj", int(rt.hidden), hq),
        ):
            if key in d:
                W = d[key]
                cases.append(Case(
                    f"attention_{layer}_{side}", "attention", W,
                    rows, cols, int(W.nbytes),
                ))

    if getattr(rt, "lm_head_kind", None) != "nvfp4" and hasattr(rt, "lm_head"):
        cases.append(Case(
            "lm_head", "lm_head", rt.lm_head,
            int(rt.vocab), int(rt.hidden), int(rt.lm_head.nbytes),
        ))
    return cases

def cuda_measure(cp, fn, reps=18):
    for _ in range(4):
        fn()
    cp.cuda.get_current_stream().synchronize()
    values = []
    for _ in range(reps):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(cp.cuda.get_elapsed_time(start, end)))
    return {
        "median_ms": statistics.median(values),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "raw_ms": values,
    }

def main():
    ensure_results()
    payload = {
        "kind": "s100_phase14r_native_zero_copy",
        "status": "started",
        "blocks": list(BLOCKS),
        "started_utc": utc_now(),
    }
    try:
        # Import order is intentional on this CUDA stack.
        import torch
        import cupy as cp
        sys.path.insert(0, str(REPO / "src"))
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA unavailable")

        rt = LightningRuntime(
            require_model_dir(),
            contexts_max=4096,
            embed_on_host=True,
            fp8_kv=True,
            verbose=False,
        )
        cases = collect(rt)
        if not cases:
            raise RuntimeError("no live BF16 matrices found")

        props = cp.cuda.runtime.getDeviceProperties(0)
        l2 = int(props.get("l2CacheSize", 32 * 1024**2))
        rotation = sum(case.weight_bytes for case in cases)
        if rotation <= 4 * l2:
            raise RuntimeError(
                f"weight rotation {rotation} is not >4x L2 {l2}"
            )

        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        torch_stream = torch.cuda.ExternalStream(stream_ptr)
        per_block = {}

        for block in BLOCKS:
            rows = []
            for index, case in enumerate(cases):
                # DLPack alias only. No clone and no contiguous transpose.
                weight_t = (
                    torch.utils.dlpack.from_dlpack(case.weight)
                    .view(torch.bfloat16)
                    .reshape(case.rows, case.cols)
                )
                generator = torch.Generator(device="cuda")
                generator.manual_seed(20260819 + block * 1000 + index)
                x_t = torch.randn(
                    block, case.cols, device="cuda", dtype=torch.float32,
                    generator=generator,
                )
                x_bf16 = x_t.to(torch.bfloat16)
                x_cp = cp.from_dlpack(x_t)
                ref_cp = cp.empty((block, case.rows), dtype=cp.float32)
                native_out = torch.empty(
                    (block, case.rows), device="cuda", dtype=torch.bfloat16
                )

                free_before = int(cp.cuda.runtime.memGetInfo()[0])

                def baseline():
                    for row in range(block):
                        rt.k.mv_bf16(
                            ref_cp[row], case.weight, x_cp[row],
                            case.rows, case.cols,
                        )

                def native():
                    with torch.cuda.stream(torch_stream):
                        torch.mm(
                            x_bf16, weight_t.t(), out=native_out
                        )

                baseline_t = cuda_measure(cp, baseline)
                native_t = cuda_measure(cp, native)

                baseline()
                native()
                cp.cuda.get_current_stream().synchronize()
                reference = cp.asnumpy(ref_cp).astype(np.float64)
                candidate = (
                    native_out.detach().float().cpu().numpy()
                    .astype(np.float64)
                )
                diff = candidate - reference
                reference_norm = np.linalg.norm(reference)
                row_reference = np.linalg.norm(reference, axis=1)
                row_diff = np.linalg.norm(diff, axis=1)
                free_after = int(cp.cuda.runtime.memGetInfo()[0])

                native_gbs = (
                    case.weight_bytes
                    / (native_t["median_ms"] * 1e-3)
                    / 1e9
                )
                rows.append({
                    "case": case.name,
                    "family": case.family,
                    "shape": [case.rows, case.cols],
                    "weight_bytes": case.weight_bytes,
                    "zero_copy_weight_alias": True,
                    "contiguous_weight_copy": False,
                    "baseline": baseline_t,
                    "native": native_t,
                    "speedup": (
                        baseline_t["median_ms"] / native_t["median_ms"]
                    ),
                    "native_effective_weight_gbs": native_gbs,
                    "free_vram_before_bytes": free_before,
                    "free_vram_after_bytes": free_after,
                    "free_vram_delta_bytes": free_after - free_before,
                    "error_vs_current_ervf": {
                        "nrmse": float(
                            np.linalg.norm(diff)
                            / max(reference_norm, 1e-30)
                        ),
                        "max_abs": float(np.max(np.abs(diff))),
                        "p95_relative_row_error": float(np.percentile(
                            row_diff / np.maximum(row_reference, 1e-30), 95
                        )),
                        "row_argmax_agreement": float(np.mean(
                            np.argmax(reference, axis=1)
                            == np.argmax(candidate, axis=1)
                        )),
                        "finite": bool(np.isfinite(candidate).all()),
                    },
                })

                del weight_t, x_t, x_bf16, x_cp, ref_cp, native_out
                torch.cuda.empty_cache()

            base_sum = sum(row["baseline"]["median_ms"] for row in rows)
            native_sum = sum(row["native"]["median_ms"] for row in rows)
            speedup = base_sum / native_sum
            large = [
                row for row in rows
                if row["weight_bytes"] >= 16 * 1024**2
            ]
            aggregate_gbs = rotation / (native_sum * 1e-3) / 1e9
            paging_safe = bool(
                aggregate_gbs >= 40.0
                and all(
                    row["native_effective_weight_gbs"] >= 20.0
                    for row in large
                )
            )
            max_nrmse = max(
                row["error_vs_current_ervf"]["nrmse"] for row in rows
            )
            mean_argmax = float(np.mean([
                row["error_vs_current_ervf"]["row_argmax_agreement"]
                for row in rows
            ]))
            per_block[str(block)] = {
                "case_count": len(rows),
                "cases": rows,
                "aggregate": {
                    "sum_independent_ervf_ms": base_sum,
                    "sum_native_ms": native_sum,
                    "useful_row_speedup": speedup,
                    "native_effective_weight_gbs": aggregate_gbs,
                    "max_case_nrmse": max_nrmse,
                    "mean_row_argmax_agreement": mean_argmax,
                    "paging_safe": paging_safe,
                },
                "component_gate_pass": bool(
                    block == 4
                    and speedup >= 2.5
                    and max_nrmse <= 0.005
                    and mean_argmax >= 0.97
                    and paging_safe
                ),
            }

        payload.update({
            "status": "measured",
            "runtime": "lean LightningRuntime",
            "matrix_count": len(cases),
            "rotation_bytes": rotation,
            "l2_bytes": l2,
            "rotation_over_l2": rotation / l2,
            "per_block": per_block,
            "b4_component_gate_pass": bool(
                per_block["4"]["component_gate_pass"]
            ),
            "b4_measurement_valid": bool(
                per_block["4"]["aggregate"]["paging_safe"]
            ),
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

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "rotation_over_l2": payload.get("rotation_over_l2"),
        "per_block": {
            key: {
                "speedup": value.get("aggregate", {}).get(
                    "useful_row_speedup"
                ),
                "native_gbs": value.get("aggregate", {}).get(
                    "native_effective_weight_gbs"
                ),
                "paging_safe": value.get("aggregate", {}).get("paging_safe"),
                "gate": value.get("component_gate_pass"),
            }
            for key, value in payload.get("per_block", {}).items()
        },
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
