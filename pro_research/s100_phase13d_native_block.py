from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BLOCKS = (2, 4, 8)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/nemotron_3_5_lightning")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13d/S100_PHASE13D_NATIVE_BLOCK.json"))
    args = ap.parse_args()
    os.environ["LS_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    sys.path.insert(0, str(REPO / "src"))
    import torch
    # Import torch before CuPy. On this CUDA stack importing CuPy first can
    # leave cuBLAS in a state where BF16 GEMM returns CUBLAS_STATUS_INVALID_VALUE.
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = Path(args.model_dir).resolve()
    rt = LightningRuntime(model, contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False)
    cases = []
    # All six BF16 Mamba blocks provide a real-weight rotation comfortably over 4x L2.
    for layer in rt.mamba_layers:
        if rt.layer[int(layer)].get("in_k") == "bf16":
            cases.append((f"mamba_{layer}_in", rt.layer[int(layer)]["in_w"], int(rt.proj.size), int(rt.hidden)))
            cases.append((f"mamba_{layer}_out", rt.layer[int(layer)]["out_w"], int(rt.hidden), int(rt.d_inner)))
    if not cases:
        raise RuntimeError("no BF16 Mamba cases found")
    props = cp.cuda.runtime.getDeviceProperties(0)
    l2_bytes = int(props.get("l2CacheSize", 32 * 1024**2))
    rotation_bytes = sum(int(w.nbytes) for _, w, _, _ in cases)
    if rotation_bytes <= 4 * l2_bytes:
        raise RuntimeError(f"rotation {rotation_bytes} is not >4x L2 {l2_bytes}")

    per_block = {}
    for B in BLOCKS:
        # Keep only one large BF16 matrix on the torch side at a time. The
        # checkpoint's resident CuPy weights already occupy most of the useful
        # VRAM; retaining twelve transposed copies can make cuBLAS fail before
        # the benchmark starts.
        case_results = []
        for name, w_cp, rows, cols in cases:
            w_t = torch.utils.dlpack.from_dlpack(w_cp).view(torch.bfloat16).reshape(rows, cols).clone()
            w_tt = w_t.t().contiguous()
            x = torch.randn(B, cols, device="cuda", dtype=torch.float32)
            xb = x.to(torch.bfloat16)
            # The legacy kernel receives a resident device vector; conversion
            # from the test tensor is deliberately outside the timed functions.
            x_cp = cp.asarray(x.detach().float().cpu().numpy())
            out = cp.empty(rows, dtype=cp.float32)

            def baseline():
                for row in range(B):
                    rt.k.mv_bf16(out, w_cp, x_cp[row], rows, cols)

            def native():
                torch.mm(xb, w_tt)

            for fn in (baseline, native):
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
            base_times, native_times = [], []
            for _ in range(args.reps):
                t0 = time.perf_counter(); baseline(); torch.cuda.synchronize(); base_times.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter(); native(); torch.cuda.synchronize(); native_times.append((time.perf_counter() - t0) * 1000)

            with torch.no_grad():
                candidate = torch.mm(xb, w_tt).float()
                reference = torch.mm(xb.float(), w_t.float().t().contiguous())
                diff = candidate - reference
                error = {
                    "nrmse": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(1e-12)),
                    "max_abs": float(diff.abs().max()),
                    "row_argmax_agreement": float((candidate.argmax(dim=1) == reference.argmax(dim=1)).float().mean()),
                }
            base_med, native_med = statistics.median(base_times), statistics.median(native_times)
            case_results.append({
                "case": name,
                "baseline_custom_rowwise_ms": {"median": base_med, "p10": percentile(base_times, 10), "p90": percentile(base_times, 90), "raw": base_times},
                "native_torch_bf16_mm_ms": {"median": native_med, "p10": percentile(native_times, 10), "p90": percentile(native_times, 90), "raw": native_times},
                "speedup_vs_custom_rowwise": base_med / native_med,
                "error_vs_fp32_same_bf16_input": error,
            })
            del w_t, w_tt, x, xb, x_cp, out
            torch.cuda.empty_cache()

        base_times = [r["baseline_custom_rowwise_ms"]["median"] for r in case_results]
        native_times = [r["native_torch_bf16_mm_ms"]["median"] for r in case_results]
        base_med, native_med = statistics.median(base_times), statistics.median(native_times)
        route_agreement = [r["error_vs_fp32_same_bf16_input"]["row_argmax_agreement"] for r in case_results]
        per_block[str(B)] = {
            "aggregation": "median across resident BF16 Mamba matrices; each case timed independently to cap VRAM",
            "baseline_custom_rowwise_ms": {"median": base_med, "p10": percentile(base_times, 10), "p90": percentile(base_times, 90), "raw_case_medians": base_times},
            "native_torch_bf16_mm_ms": {"median": native_med, "p10": percentile(native_times, 10), "p90": percentile(native_times, 90), "raw_case_medians": native_times},
            "speedup_vs_custom_rowwise": base_med / native_med,
            "case_results": case_results,
            "mean_row_argmax_agreement": float(np.mean(route_agreement)),
            "dense_gate_2_5x": bool(base_med / native_med >= 2.5),
        }
    result = {
        "kind": "s100_phase13d_native_tensor_core_block",
        "status": "measured",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model),
        "claim_boundary": "native BF16 component ceiling; no full validation, route agreement, or end-to-end causal claim",
        "blocks": list(BLOCKS),
        "matrix_count": len(cases),
        "rotation_bytes": rotation_bytes,
        "l2_bytes": l2_bytes,
        "rotation_over_l2": rotation_bytes / l2_bytes,
        "cases": [{"name": n, "shape": [r, c], "bytes": int(w.nbytes)} for n, w, r, c in cases],
        "per_block": per_block,
        "gates": {"b4_dense_speedup_ge_2_5": bool(per_block["4"]["dense_gate_2_5x"]), "official_quality_green": False, "promotion_open": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
