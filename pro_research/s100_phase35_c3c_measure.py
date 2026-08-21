from __future__ import annotations

import json
import struct
import traceback

import numpy as np

from common import REPO, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
from diag_native_nvfp4_c3b_realact import (
    load_raw_tensor,
    native_call,
    quantize_a_reference,
)
import native_nvfp4_c3a_layout_v2 as c3v2
import native_nvfp4_c3a_lib as c3lib
from s100_phase35_c3c_quantizer import FusedStaticNVFP4Quantizer


RESULTS = REPO / "pro_research" / "results" / "native_nvfp4"
C3B = RESULTS / "C3B_REAL_ACTIVATION.json"
OUT = RESULTS / "C3C_FUSED_STATIC_QUANTIZER.json"
M_VALUES = (1, 2, 4, 8)


def event_p50(cp, fn, reps: int = 100, rounds: int = 5) -> dict:
    for _ in range(8):
        fn()
    cp.cuda.get_current_stream().synchronize()
    values = []
    for _ in range(rounds):
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        for _ in range(reps):
            fn()
        end.record()
        end.synchronize()
        values.append(float(cp.cuda.get_elapsed_time(begin, end)) / reps)
    return {
        "samples_ms": values,
        "p50_ms": float(np.median(values)),
        "min_ms": min(values),
        "max_ms": max(values),
        "reps": reps,
    }


def torch_a_from_fused(torch, packed, blocked, global_scale):
    return {
        "fp4": torch.as_tensor(packed, device="cuda").view(
            torch.float4_e2m1fn_x2
        ),
        "block": torch.as_tensor(blocked, device="cuda").view(
            torch.float8_e4m3fn
        ),
        "global": global_scale,
    }


def graph_replay_gate(cp, torch, x_cp, quantizer, scale, a_fused, b, reference, F, ST, SW):
    stream = cp.cuda.Stream(non_blocking=True)
    external = torch.cuda.ExternalStream(stream.ptr)
    with stream:
        with torch.cuda.stream(external):
            for _ in range(3):
                quantizer.quantize(x_cp, scale)
                native_call(torch, F, ST, SW, a_fused, b)
    stream.synchronize()
    holder = {}
    with stream:
        stream.begin_capture()
        with torch.cuda.stream(external):
            quantizer.quantize(x_cp, scale)
            holder["out"] = native_call(torch, F, ST, SW, a_fused, b)
        graph = stream.end_capture()
    out = holder["out"]
    out.zero_()
    torch.cuda.synchronize()
    graph.launch(stream)
    stream.synchronize()
    first = bool(torch.equal(out, reference))
    out.zero_()
    torch.cuda.synchronize()
    graph.launch(stream)
    stream.synchronize()
    second = bool(torch.equal(out, reference))
    return {"capture_succeeded": True, "replay1_exact": first, "replay2_exact": second}


def main() -> int:
    payload = {
        "kind": "s100_phase35_c3c_fused_static_quantizer",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": "fused quantizer and native component timing; no full-verifier claim",
    }
    try:
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        import cupy as cp
        import torch
        import torch.nn.functional as F

        ST = F.ScalingType
        SW = F.SwizzleType
        c3v2.install(c3lib)
        c3b = json.loads(C3B.read_text(encoding="utf-8"))
        if c3b.get("status") != "real_activation_native_candidate":
            raise RuntimeError("C3B parent is not green")
        if (c3b.get("summary") or {}).get("selected_candidate_arm") != "static_1p10":
            raise RuntimeError("C3C requires the frozen static_1p10 parent")

        arrays = c3b["capture_manifest"]["arrays"]
        x_sources = {
            "lm_head": load_raw_tensor(torch, arrays["lm_head_in"]),
            "shared_up": load_raw_tensor(torch, arrays["moe_normed"]),
            "shared_down": load_raw_tensor(torch, arrays["shared_act"]),
            "routed_up": load_raw_tensor(torch, arrays["moe_normed"]),
        }
        entries, headers = c3lib.load_index_headers()
        families = []
        graph_gate = None
        all_bytes_exact = True
        all_native_exact = True
        all_quant_fast = True
        all_combined_better = True

        for family in c3b["families"]:
            label = family["label"]
            selected = family["selected"]
            n, k = int(selected["N"]), int(selected["K"])
            scale = float(family["activation"]["static_tensor_scale"])
            weight_raw = c3lib.tensor_raw(selected["weight"], entries, headers)
            scale_raw = c3lib.tensor_raw(selected["scale"], entries, headers)
            global_raw = c3lib.tensor_raw(selected["global"], entries, headers)
            weight_global = float(struct.unpack("<f", global_raw)[0])
            b = c3lib.make_b(
                torch, weight_raw, scale_raw, weight_global, n, k
            )
            family_rows = []
            x_all = x_sources[label][32:64].contiguous()

            for m in M_VALUES:
                x = x_all[:m].contiguous().to("cuda")
                x_cp = cp.from_dlpack(x)
                static_g = torch.tensor(
                    [scale], dtype=torch.float32, device="cuda"
                )
                reference_a = quantize_a_reference(torch, x, static_g)
                quantizer = FusedStaticNVFP4Quantizer(k, m)
                packed, blocked = quantizer.quantize(x_cp, scale)
                cp.cuda.get_current_stream().synchronize()
                codes_exact = bool(
                    np.array_equal(
                        cp.asnumpy(packed),
                        reference_a["u8"].detach().cpu().numpy(),
                    )
                )
                scales_exact = bool(
                    np.array_equal(
                        cp.asnumpy(blocked),
                        reference_a["block"].view(torch.uint8).detach().cpu().numpy(),
                    )
                )
                fused_a = torch_a_from_fused(torch, packed, blocked, static_g)
                reference_out = native_call(
                    torch, F, ST, SW, reference_a, b
                )
                fused_out = native_call(torch, F, ST, SW, fused_a, b)
                torch.cuda.synchronize()
                native_exact = bool(torch.equal(reference_out, fused_out))
                all_bytes_exact &= codes_exact and scales_exact
                all_native_exact &= native_exact

                row = {
                    "M": m,
                    "codes_byte_exact": codes_exact,
                    "blocked_scales_byte_exact": scales_exact,
                    "native_output_exact": native_exact,
                }
                if m == 8:
                    quant_timing = event_p50(
                        cp, lambda: quantizer.quantize(x_cp, scale)
                    )
                    external = torch.cuda.ExternalStream(
                        cp.cuda.get_current_stream().ptr
                    )

                    def combined():
                        quantizer.quantize(x_cp, scale)
                        with torch.cuda.stream(external):
                            return native_call(torch, F, ST, SW, fused_a, b)

                    combined_timing = event_p50(cp, combined, reps=60)
                    reference_combined = float(
                        family["reference_quantizer_plus_native_upper_bound"]
                        ["static_1p10"]["M8"]["p50_ms"]
                    )
                    row.update(
                        {
                            "quantizer_timing": quant_timing,
                            "quantizer_plus_native_timing": combined_timing,
                            "c3b_reference_combined_ms": reference_combined,
                            "combined_gain_vs_reference": (
                                reference_combined
                                - combined_timing["p50_ms"]
                            ) / reference_combined,
                            "resource": quantizer.resource_audit(),
                        }
                    )
                    all_quant_fast &= quant_timing["p50_ms"] <= 0.10
                    all_combined_better &= combined_timing["p50_ms"] < reference_combined
                    if label == "lm_head":
                        graph_gate = graph_replay_gate(
                            cp, torch, x_cp, quantizer, scale,
                            fused_a, b, reference_out, F, ST, SW,
                        )
                family_rows.append(row)
                del reference_out, fused_out, reference_a, fused_a, static_g, x
                torch.cuda.empty_cache()
            families.append(
                {
                    "label": label,
                    "N": n,
                    "K": k,
                    "static_tensor_scale": scale,
                    "rows": family_rows,
                }
            )
            del b
            torch.cuda.empty_cache()

        graph_green = bool(
            graph_gate
            and graph_gate.get("capture_succeeded")
            and graph_gate.get("replay1_exact")
            and graph_gate.get("replay2_exact")
        )
        gates = {
            "C3C_G1_all_codes_and_scales_byte_exact": all_bytes_exact,
            "C3C_G2_all_native_outputs_exact_to_reference_A": all_native_exact,
            "C3C_G3_all_M8_quantizers_le_0p10_ms": all_quant_fast,
            "C3C_G4_all_combined_paths_beat_reference": all_combined_better,
            "C3C_G5_graph_capture_replays_exact": graph_green,
        }
        payload.update(
            {
                "status": "fused_static_quantizer_candidate" if all(gates.values()) else "fused_quantizer_gate_failed",
                "selected_scale_policy": "static_1p10",
                "families": families,
                "graph_gate": graph_gate,
                "gates": gates,
                "C4_NATIVE_HEAD_INTEGRATION_OPEN": bool(all(gates.values())),
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )
    write_json_atomic(OUT, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "gates": payload.get("gates"),
                "graph_gate": payload.get("graph_gate"),
                "families": [
                    {
                        "label": family["label"],
                        "M8": next(
                            row for row in family["rows"] if row["M"] == 8
                        ),
                    }
                    for family in payload.get("families", [])
                ],
                "error": (payload.get("error") or {}).get("message"),
                "output": str(OUT),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") != "technical_failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
