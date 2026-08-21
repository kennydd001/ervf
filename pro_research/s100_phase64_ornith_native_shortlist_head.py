"""Phase64 native Ornith LM-head shortlist followed by exact ERVF rerank."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase35_c3c_quantizer import FusedStaticNVFP4Quantizer
from s100_phase48_ornith_swiglu_h8 import _load_projection, _measure
from s100_phase64_ornith_shortlist_kernel import ExactERVFShortlist


RESULTS = REPO / "pro_research" / "results" / "s100_phase64"
PREREG = REPO / "pro_research" / "S100_PHASE64_ORNITH_NATIVE_SHORTLIST_HEAD_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase64_ornith_native_shortlist_head.py"
KERNELS = REPO / "pro_research" / "s100_phase64_ornith_shortlist_kernel.py"
SHORTLIST = 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=25)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE64_ORNITH_NATIVE_SHORTLIST_HEAD.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase64_ornith_native_shortlist_head",
        "status": "started",
        "started_utc": utc_now(),
        "snapshot": str(args.snapshot.resolve()),
        "shortlist": SHORTLIST,
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    cp = None
    try:
        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        head = _load_projection(snapshot, index["weight_map"], "lm_head")
        rows, cols = 248320, 2048

        import cupy as cp_module
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

        cp = cp_module
        layout_v2.install(c3lib)
        fused = FusedNVFP4()
        exact = ExactERVFShortlist()
        quantizer = FusedStaticNVFP4Quantizer(cols, 4)
        codes = cp.asarray(head["codes"])
        scales = cp.asarray(head["scales"])
        b = c3lib.make_b(
            torch, head["codes"].tobytes(), head["scales"].tobytes(),
            head["global_scale"], rows, cols,
        )
        stream = cp.cuda.get_current_stream()
        external = torch.cuda.ExternalStream(stream.ptr)
        packed_t = torch.utils.dlpack.from_dlpack(quantizer.packed)
        blocked_t = torch.utils.dlpack.from_dlpack(quantizer.blocked_scales)

        rng = np.random.default_rng(64000000)
        x_host = rng.standard_normal((32, cols), dtype=np.float32)
        tensor_scale = float(np.max(np.abs(x_host)) * 1.10 / (448.0 * 6.0))
        a_global = torch.tensor([tensor_scale], dtype=torch.float32, device="cuda")
        a = {
            "fp4": packed_t.view(torch.float4_e2m1fn_x2),
            "block": blocked_t.view(torch.float8_e4m3fn),
            "global": a_global,
        }

        quality = []
        timing_state = {}
        for batch in range(8):
            x = cp.asarray(x_host[batch * 4:(batch + 1) * 4])
            control = cp.empty((4, rows), dtype=cp.float32)
            for token in range(4):
                fused.gemv_into(
                    control[token], codes, scales, x[token], head["global_scale"], rows, cols
                )
            quantizer.quantize(x, tensor_scale)
            stream.synchronize()
            with torch.cuda.stream(external):
                native = native_call(torch, F, F.ScalingType, F.SwizzleType, a, b)
                _native_values, ids_t = torch.topk(native, SHORTLIST, dim=1)
            external.synchronize()
            ids = cp.from_dlpack(ids_t)
            rerank = cp.empty((4, SHORTLIST), dtype=cp.float32)
            exact(
                codes, scales, fused.e2m1, fused.e4m3, x,
                ids.data.ptr, rerank.data.ptr, head["global_scale"], SHORTLIST, cols,
            )
            selected = cp.take_along_axis(
                ids, cp.argmax(rerank, axis=1).reshape(-1, 1), axis=1
            ).reshape(-1)
            stream.synchronize()
            control_host = cp.asnumpy(control)
            ids_host = cp.asnumpy(ids)
            rerank_host = cp.asnumpy(rerank)
            selected_host = cp.asnumpy(selected)
            control_ids = np.argmax(control_host, axis=1)
            gathered = np.take_along_axis(control_host, ids_host, axis=1)
            quality.append({
                "batch": batch,
                "control_ids": control_ids.astype(np.int64).tolist(),
                "native_ids": torch.argmax(native, dim=1).cpu().numpy().astype(np.int64).tolist(),
                "selected_ids": selected_host.astype(np.int64).tolist(),
                "top64_contains_control": [
                    bool(control_ids[token] in ids_host[token]) for token in range(4)
                ],
                "rerank_score_bit_exact": bool(np.array_equal(
                    gathered.view(np.uint32), rerank_host.view(np.uint32)
                )),
                "selected_exact": bool(np.array_equal(selected_host, control_ids)),
            })
            if batch == 0:
                timing_state = {
                    "x": x,
                    "control": control,
                    "ids": ids,
                    "rerank": rerank,
                    "selected": selected,
                    "native": native,
                }

        x = timing_state["x"]
        control = timing_state["control"]

        def run_control() -> None:
            for token in range(4):
                fused.gemv_into(
                    control[token], codes, scales, x[token], head["global_scale"], rows, cols
                )

        def run_candidate() -> None:
            quantizer.quantize(x, tensor_scale)
            stream.synchronize()
            with torch.cuda.stream(external):
                native = native_call(torch, F, F.ScalingType, F.SwizzleType, a, b)
                _values, ids_t = torch.topk(native, SHORTLIST, dim=1)
            external.synchronize()
            ids = cp.from_dlpack(ids_t)
            rerank = cp.empty((4, SHORTLIST), dtype=cp.float32)
            exact(
                codes, scales, fused.e2m1, fused.e4m3, x,
                ids.data.ptr, rerank.data.ptr, head["global_scale"], SHORTLIST, cols,
            )
            selected = cp.take_along_axis(
                ids, cp.argmax(rerank, axis=1).reshape(-1, 1), axis=1
            )
            timing_state.update({"native": native, "ids_t": ids_t, "ids": ids,
                                 "rerank": rerank, "selected": selected})

        control_timing = _measure(cp, run_control, args.warmup, args.reps)
        candidate_timing = _measure(cp, run_candidate, args.warmup, args.reps)
        run_candidate()
        stream.synchronize()
        repeat_before = cp.asnumpy(timing_state["selected"])
        run_candidate()
        stream.synchronize()
        repeat_after = cp.asnumpy(timing_state["selected"])
        all_contains = all(all(row["top64_contains_control"]) for row in quality)
        all_selected = all(row["selected_exact"] for row in quality)
        all_scores = all(row["rerank_score_bit_exact"] for row in quality)
        exact_resource = exact.resource_audit()
        quant_resource = quantizer.resource_audit()
        speedup = float(control_timing["p50"] / candidate_timing["p50"])
        gates = {
            "P64_G1_top64_recall_32_of_32": all_contains,
            "P64_G2_rerank_ids_exact_and_repeat": (
                all_selected and bool(np.array_equal(repeat_before, repeat_after))
            ),
            "P64_G3_shortlist_scores_bit_exact": all_scores,
            "P64_G4_below_2ms_and_speedup_ge_2_5": (
                candidate_timing["p50"] < 2.0 and speedup >= 2.5
            ),
            "P64_G5_resource_budget": all(
                (row.get("local_size_bytes") or 0) == 0
                and (row.get("num_regs") or 10_000) <= 64
                for row in (exact_resource, quant_resource)
            ),
        }
        native_top1_exact = sum(
            int(native_id == control_id)
            for row in quality
            for native_id, control_id in zip(row["native_ids"], row["control_ids"])
        )
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "activation_tensor_scale": tensor_scale,
            "quality": quality,
            "summary": {
                "rows": 32,
                "native_top1_exact": native_top1_exact,
                "native_top1_rate": native_top1_exact / 32,
                "top64_recall": sum(
                    int(value) for row in quality for value in row["top64_contains_control"]
                ) / 32,
                "rerank_exact_rate": sum(
                    4 if row["selected_exact"] else 0 for row in quality
                ) / 32,
                "control_h1_x4_ms": control_timing["p50"],
                "candidate_h4_ms": candidate_timing["p50"],
                "speedup": speedup,
            },
            "timings_ms": {"control_h1_x4": control_timing, "candidate": candidate_timing},
            "resources": {"exact_shortlist": exact_resource, "quantizer": quant_resource},
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
    finally:
        if cp is not None:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNELS))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "resources": payload.get("resources"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
