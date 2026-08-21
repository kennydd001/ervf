"""Phase83 long-context sweep for the Phase69 Ornith component floor."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase68_ornith_full_attn_h4 import _bench_case
from s100_phase68_ornith_full_attn_h4_kernels import OrnithFullAttentionH4Kernels


RESULTS = REPO / "pro_research" / "results" / "s100_phase83"
PREREG = REPO / "pro_research" / "S100_PHASE83_ORNITH_LONG_CONTEXT_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase83_ornith_long_context.py"
KERNELS = REPO / "pro_research" / "s100_phase68_ornith_full_attn_h4_kernels.py"
PHASE68 = REPO / "pro_research" / "results" / "s100_phase68" / "S100_PHASE68_ORNITH_FULL_ATTN_H4.json"
PHASE69 = REPO / "pro_research" / "results" / "s100_phase69" / "S100_PHASE69_ORNITH_SUPPORT_H4.json"
CONTEXTS = (1024, 4096, 16384, 50000, 100000)
LAYERS = 10
KV_HEADS = 2
HEAD_WIDTH = 256
KV_PLANES = 2
FP32_BYTES = 4
RUNTIME_RESERVE_BYTES = 1 << 29
TARGET_MS = 4000.0 / 65.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pottokao", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE83_ORNITH_LONG_CONTEXT.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase83_ornith_long_context",
        "status": "started",
        "started_utc": utc_now(),
        "contexts": list(CONTEXTS),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    cp = None
    try:
        import cupy as cp_module

        cp = cp_module
        snapshot = args.pottokao.resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        kernels = OrnithFullAttentionH4Kernels()
        records = []
        for index, context in enumerate(CONTEXTS):
            row = _bench_case(
                kernels,
                "pottokao",
                snapshot,
                "model.layers.23.self_attn",
                context,
                83000 + index,
                args.warmup,
                args.repeats,
            )
            kv_bytes = (
                LAYERS * KV_HEADS * (context + 4) * HEAD_WIDTH
                * KV_PLANES * FP32_BYTES
            )
            row["ten_layer_fp32_kv_bytes"] = kv_bytes
            row["ten_layer_fp32_kv_GiB"] = kv_bytes / (1 << 30)
            row["fits_phase46_runtime_reserve"] = kv_bytes <= RUNTIME_RESERVE_BYTES
            records.append(row)

        phase68 = json.loads(PHASE68.read_text("utf-8"))
        phase69 = json.loads(PHASE69.read_text("utf-8"))
        frozen_full_1024 = float(phase68["budget"]["worse_10_full_layers_ms_h4"])
        frozen_total_1024 = float(phase69["budget"]["combined_known_floor_ms_h4"])
        non_full = frozen_total_1024 - frozen_full_1024
        projections = []
        for row in records:
            full_ms = float(row["projected_10_full_layers_ms_h4"])
            total_ms = non_full + full_ms
            projections.append({
                "context": int(row["base_context"]),
                "selected_arm": row["selected_arm"],
                "one_full_layer_ms_h4": float(row["selected_complete_ms"]),
                "ten_full_layers_ms_h4": full_ms,
                "complete_component_floor_ms_h4": total_ms,
                "complete_component_floor_tok_s": 4000.0 / total_ms,
                "ten_layer_fp32_kv_bytes": int(row["ten_layer_fp32_kv_bytes"]),
                "ten_layer_fp32_kv_GiB": float(row["ten_layer_fp32_kv_GiB"]),
                "fits_phase46_runtime_reserve": bool(row["fits_phase46_runtime_reserve"]),
            })
        by_context = {row["context"]: row for row in projections}
        quality_ok = all(
            row["arms"][row["selected_arm"]]["quality"]["nrmse"] <= 5.0e-5
            and row["arms"][row["selected_arm"]]["quality"]["candidate_finite"]
            and row["arms"][row["selected_arm"]]["fresh_input_bit_deterministic"]
            for row in records
        )
        ctx1024_delta = abs(
            by_context[1024]["ten_full_layers_ms_h4"] - frozen_full_1024
        ) / frozen_full_1024
        gates = {
            "P83_G1_selected_quality": quality_ok,
            "P83_G2_ctx1024_reproduction_within_10pct": ctx1024_delta <= 0.10,
            "P83_G3_ctx50k_component_floor_ge_65tps": (
                by_context[50000]["complete_component_floor_ms_h4"] <= TARGET_MS
            ),
            "P83_G4_ctx100k_component_floor_ge_65tps": (
                by_context[100000]["complete_component_floor_ms_h4"] <= TARGET_MS
            ),
            "P83_G5_long_context_fp32_kv_fits_reserve": (
                by_context[50000]["fits_phase46_runtime_reserve"]
                and by_context[100000]["fits_phase46_runtime_reserve"]
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "inputs": {
                "snapshot": str(snapshot),
                "frozen_phase68_full_1024_ms_h4": frozen_full_1024,
                "frozen_phase69_total_1024_ms_h4": frozen_total_1024,
                "frozen_non_full_attention_ms_h4": non_full,
                "runtime_reserve_bytes": RUNTIME_RESERVE_BYTES,
            },
            "records": records,
            "projections": projections,
            "ctx1024_relative_delta": ctx1024_delta,
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
    finally:
        if cp is not None:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNELS, PHASE68, PHASE69))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "projections": payload.get("projections"),
        "gates": payload.get("gates"),
        "error": payload.get("error"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
