from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BLOCKS = (2, 4, 8)


def timed(fn, reps, torch):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); values.append((time.perf_counter() - t0) * 1000)
    return float(np.median(values))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/nemotron_3_5_lightning")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13h/S100_PHASE13H_NATIVE_DTYPE_BLOCKS.json"))
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src")); os.environ["LS_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    import torch
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(Path(args.model_dir).resolve(), contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False)
    cases = []
    # One representative per native path; D already covered the full BF16
    # Mamba population, so this screen adds BF16 attention and quantized paths.
    for layer in rt.attn_layers[:1]:
        d = rt.layer[int(layer)]
        for key, rows, cols in (("q_proj", rt.n_heads * rt.head_dim, rt.hidden), ("o_proj", rt.hidden, rt.n_heads * rt.head_dim)):
            cases.append({"name": f"attention_{layer}_{key}", "kind": "bf16", "weight": d[key], "rows": int(rows), "cols": int(cols), "scale": 1.0})
    for layer in rt.mamba_layers:
        d = rt.layer[int(layer)]
        if d.get("in_k") == "fp8_tensor":
            cases.append({"name": f"mamba_{layer}_in", "kind": "fp8_tensor", "weight": d["in_w8"], "rows": int(rt.proj.size), "cols": int(rt.hidden), "scale": float(d["in_s"])})
            break
    for layer in rt.mamba_layers:
        d = rt.layer[int(layer)]
        if d.get("in_k") == "nvfp4":
            cases.append({"name": f"mamba_{layer}_in", "kind": "nvfp4", "weight": d["in_codes"], "scales": d["in_scales"], "rows": int(rt.proj.size), "cols": int(rt.hidden), "scale": float(d["in_g"])})
            break

    results = []
    for case in cases:
        rows, cols = case["rows"], case["cols"]
        w = case["weight"]
        x = torch.randn(8, cols, device="cuda", dtype=torch.float32)
        x_cp = cp.asarray(x.detach().cpu().numpy())
        out = cp.empty(rows, dtype=cp.float32)

        if case["kind"] == "bf16":
            wt = torch.utils.dlpack.from_dlpack(w).view(torch.bfloat16).reshape(rows, cols).clone()
            wtt = wt.t().contiguous()
            xb = x.to(torch.bfloat16)
            baseline = lambda: [rt.k.mv_bf16(out, w, x_cp[row], rows, cols) for row in range(B)]
            native = lambda: torch.mm(xb, wtt)
            native_kind = "torch_bf16_mm"
            native_setup = "supported"
        elif case["kind"] == "fp8_tensor":
            wt8 = torch.utils.dlpack.from_dlpack(w).view(torch.uint8).reshape(rows, cols)
            try:
                wt8 = wt8.view(torch.float8_e4m3fn).clone()
                wtt = wt8.t().contiguous()
                xb = x.to(torch.float8_e4m3fn)
                torch.mm(xb[:1], wtt); torch.cuda.synchronize()
                native = lambda: torch.mm(xb, wtt)
                native_kind = "torch_float8_mm"
                native_setup = "supported"
            except Exception as exc:
                native = None; native_kind = "torch_float8_mm"; native_setup = f"unsupported: {type(exc).__name__}: {exc}"
            baseline = lambda: [rt.k.mv_fp8_tensor(out, w, x_cp[row], case["scale"], rows, cols) for row in range(B)]
        else:
            try:
                packed = torch.utils.dlpack.from_dlpack(w).view(torch.uint8).reshape(rows, cols // 2)
                wt4 = packed.view(torch.float4_e2m1fn_x2)
                torch.mm(x[:1].to(torch.float16), wt4.t())
                native_setup = "supported"
            except Exception as exc:
                native_setup = f"unsupported: {type(exc).__name__}: {exc}"
            native = None; native_kind = "torch_float4_mm"
            baseline = lambda: [rt.fused.gemv_into(out, w, case["scales"], x_cp[row], case["scale"], rows, cols) for row in range(B)]

        B = 4
        base_ms = timed(baseline, args.reps, torch)
        native_ms = timed(native, args.reps, torch) if native is not None else None
        results.append({"name": case["name"], "kind": case["kind"], "shape": [rows, cols], "native_kernel": native_kind, "native_setup": native_setup, "B": B, "baseline_custom_ms": base_ms, "native_ms": native_ms, "speedup": base_ms / native_ms if native_ms else None})
        print(f"tested {case['name']} kind={case['kind']} native={native_setup}", flush=True)
        torch.cuda.empty_cache()

    result = {"kind": "s100_phase13h_native_dtype_block_screen", "status": "measured", "created_utc": datetime.now(timezone.utc).isoformat(), "model_dir": str(Path(args.model_dir).resolve()), "claim_boundary": "native datatype capability/component screen; no end-to-end quality or full B=4/B=8 speculative runtime", "cases": results, "gates": {"native_bf16_attention_measured": any(r["kind"] == "bf16" and r["native_ms"] is not None for r in results), "native_fp8_measured": any(r["kind"] == "fp8_tensor" and r["native_ms"] is not None for r in results), "native_nvfp4_measured": any(r["kind"] == "nvfp4" and r["native_ms"] is not None for r in results), "official_quality_green": False, "promotion_open": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(results), "gates": result["gates"]}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
