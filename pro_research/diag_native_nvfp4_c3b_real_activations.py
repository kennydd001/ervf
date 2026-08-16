"""C3B phase 2: real V18 activation -> dynamic NVFP4 -> real checkpoint native SM120."""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from native_nvfp4_c3b_quant import (
    CAPTURE, COLD_L2_MULTIPLE, M_VALUES, RESULT, dequantize_activation, event_p50,
    gpu_idle_snapshot, lm_distribution_metrics, load_capture, parse_f32_scalar,
    quantize_activation, read_f32, sha256_file, tensor_metrics,
)

C3A = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_REAL_WEIGHT.json"
C3A_PRE = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_V2_LAYOUT_PREFLIGHT.json"
W4A8 = REPO / "pro_research" / "results" / "native_nvfp4" / "FP4_W4A8_RECIPES.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_C3B_REAL_ACT_PREREGISTRATION.md"

ACT_COS_MIN = 0.995
ACT_NRMSE_MAX = 0.120
OUT_COS_MIN = 0.995
OUT_NRMSE_MAX = 0.100
OUT_NMAX_MAX = 0.250
LM_TOP1_MIN = 0.95
LM_TOP5_MIN = 0.80
LM_KL_MEAN_MAX = 0.020
LM_KL_MAX_MAX = 0.100
M8_TOTAL_OVER_M1_MAX = 2.0

FAMILY_IO = {
    "lm_head": ("lm_head_input", "lm_head_ref"),
    "shared_up": ("moe_normed", "shared_up_ref"),
    "shared_down": ("shared_down_input", "shared_down_ref"),
    "routed_up": ("moe_normed", "routed_up_ref"),
}


def _group_indices(rows: list[dict[str, Any]], m: int) -> list[list[int]]:
    by_prompt: dict[int, list[int]] = {}
    for r in rows:
        by_prompt.setdefault(int(r["prompt_index"]), []).append(int(r["row"]))
    out = []
    for pi in sorted(by_prompt):
        idx = by_prompt[pi]
        if len(idx) != 8 or len(idx) % m:
            raise RuntimeError(f"prompt {pi}: expected 8 rows divisible by M={m}; got {len(idx)}")
        out.extend([idx[i:i+m] for i in range(0, len(idx), m)])
    return out


def _hash_tensor_bytes(torch, t) -> str:
    import hashlib
    raw = bytes(t.detach().contiguous().cpu().view(torch.uint8).flatten().tolist())
    return hashlib.sha256(raw).hexdigest()


def _aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "group_count": len(groups),
        "activation_min_cosine": min(g["activation_metrics"]["cosine"] for g in groups),
        "activation_max_normalized_rmse": max(g["activation_metrics"]["normalized_rmse"] for g in groups),
        "output_min_cosine": min(g["output_metrics"]["cosine"] for g in groups),
        "output_max_normalized_rmse": max(g["output_metrics"]["normalized_rmse"] for g in groups),
        "output_max_normalized_max_abs_error": max(g["output_metrics"]["normalized_max_abs_error"] for g in groups),
    }


def _clone_b(torch, base):
    u = base["u8"].clone(); block = base["block"].clone()
    return {"u8": u, "fp4": u.view(torch.float4_e2m1fn_x2).t(),
            "block": block, "global": base["global"]}


def _performance(torch, F, ST, SW, c3lib, x8, base_b, l2_bytes: int) -> dict[str, Any]:
    one_b = int(base_b["u8"].numel() + base_b["block"].numel())
    cycle = max(1, math.ceil((COLD_L2_MULTIPLE * l2_bytes) / one_b))
    working = cycle * one_b
    free, _ = torch.cuda.mem_get_info()
    estimated = working + one_b + 192 * 1024 * 1024
    rec: dict[str, Any] = {"one_b_bytes": one_b, "cycle": cycle,
        "rotation_working_set_bytes": working, "working_set_over_l2": working / l2_bytes,
        "free_bytes_before": int(free), "estimated_peak_bytes": int(estimated)}
    if estimated > int(free * 0.70):
        rec["status"] = "not_run_memory_gate"
        return rec
    bs = [base_b] + [_clone_b(torch, base_b) for _ in range(1, cycle)]
    by_m = {}
    for m in M_VALUES:
        xm = x8[:m].contiguous()
        qa = quantize_activation(torch, xm)
        q_counter = {"i": 0}
        def gemm_only():
            i = q_counter["i"]; q_counter["i"] = i + 1
            return c3lib.native_call(torch, F, ST, SW, qa, bs[i % cycle])
        def quant_only():
            return quantize_activation(torch, xm)
        c_counter = {"i": 0}
        def combined():
            q = quantize_activation(torch, xm)
            i = c_counter["i"]; c_counter["i"] = i + 1
            return c3lib.native_call(torch, F, ST, SW, q, bs[i % cycle])
        reps = max(cycle * 2, 20)
        by_m[f"M{m}"] = {
            "quantizer": event_p50(torch, quant_only, max(20, min(reps, 80))),
            "prequantized_gemm_cold": event_p50(torch, gemm_only, reps),
            "combined_quant_plus_gemm_cold": event_p50(torch, combined, reps),
        }
        del qa
    rec["by_M"] = by_m
    m1 = float(by_m["M1"]["combined_quant_plus_gemm_cold"]["p50_ms"])
    m8 = float(by_m["M8"]["combined_quant_plus_gemm_cold"]["p50_ms"])
    rec["M8_total_over_M1"] = m8 / m1 if m1 else None
    rec["M8_combined_per_token_ms"] = m8 / 8.0
    rec["status"] = "measured"
    del bs
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c3b_real_activation",
        "status": "started", "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "24 frozen V18 states; W4A4 activation PTQ projection/logit quality + prototype quantizer/GEMM timing; no rollout/tok-s claim",
        "thresholds": {"activation_cosine_min": ACT_COS_MIN, "activation_nrmse_max": ACT_NRMSE_MAX,
            "output_cosine_min": OUT_COS_MIN, "output_nrmse_max": OUT_NRMSE_MAX,
            "output_normalized_max_abs_max": OUT_NMAX_MAX, "lm_top1_min": LM_TOP1_MIN,
            "lm_top5_overlap_min": LM_TOP5_MIN, "lm_mean_kl_max": LM_KL_MEAN_MAX,
            "lm_max_kl_max": LM_KL_MAX_MAX, "cold_l2_multiple_min": COLD_L2_MULTIPLE,
            "M8_total_over_M1_max_engineering_signal": M8_TOTAL_OVER_M1_MAX},
    }
    try:
        payload["gpu_idle_preflight"] = gpu_idle_snapshot()
        import torch
        import torch.nn.functional as F
        import native_nvfp4_c3a_lib as c3lib
        import native_nvfp4_c3a_layout_v2 as layout_v2
        layout_v2.install(c3lib)
        ST, SW = F.ScalingType, F.SwizzleType
        cap = tuple(torch.cuda.get_device_capability(0))
        if not (str(torch.__version__).startswith("2.12.1") and str(torch.version.cuda).startswith("13.2") and cap >= (12, 0)):
            raise RuntimeError(f"wrong isolated environment torch={torch.__version__} cuda={torch.version.cuda} cap={cap}")

        capture = load_capture()
        if capture.get("status") != "capture_pass" or len(capture.get("rows") or []) != 24:
            raise RuntimeError(f"capture parent not green: {capture.get('status')}")
        arrays = {k: read_f32(torch, v) for k, v in capture["arrays"].items()}
        rows = list(capture["rows"])
        expected_shapes = {
            "moe_normed": [24, 2688], "shared_up_ref": [24, 3712],
            "shared_down_input": [24, 3712], "shared_down_ref": [24, 2688],
            "routed_up_ref": [24, 1856], "lm_head_input": [24, 2688],
            "lm_head_ref": [24, 131072],
        }
        prompt_rows: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            prompt_rows.setdefault(int(r["prompt_index"]), []).append(r)
        prompt_partition_ok = set(prompt_rows) == {0, 1, 2} and all(
            len(v) == 8 and sorted(int(x["position"]) for x in v) == list(range(8))
            for v in prompt_rows.values()
        )
        shape_ok = set(arrays) == set(expected_shapes) and all(
            list(arrays[k].shape) == expected_shapes[k] for k in expected_shapes
        )
        capture_integrity = bool(
            capture.get("finite") and prompt_partition_ok and shape_ok
            and all(torch.isfinite(v).all().item() for v in arrays.values())
        )

        c3a = json.loads(C3A.read_text(encoding="utf-8"))
        c3pre = json.loads(C3A_PRE.read_text(encoding="utf-8"))
        w4a8 = json.loads(W4A8.read_text(encoding="utf-8")) if W4A8.exists() else {}
        c3a_gates = c3a.get("gates") or {}
        c3pre_gates = c3pre.get("gates") or {}
        parents_green = bool(
            c3a.get("status") in {"real_weight_representation_and_geometry_candidate", "real_weight_representation_green_perf_miss"}
            and all(c3a_gates.get(f"C3A_G{i}") is True for i in range(1, 9))
            and c3pre.get("status") == "layout_v2_preflight_pass"
            and all(v is True for k, v in c3pre_gates.items() if k.startswith("V2_G"))
            and w4a8.get("verdict") == "w4a4_only"
        )

        entries, headers = c3lib.load_index_headers()
        reps = {x["label"]: x for x in c3lib.choose_representatives(c3lib.all_nvfp4_triples(entries, headers))}
        props = torch.cuda.get_device_properties(0)
        l2 = int(getattr(props, "L2_cache_size", 0) or getattr(props, "l2_cache_size", 0) or 0)
        if l2 <= 0:
            raise RuntimeError("unable to read L2 size")

        family_records = {}
        all_native_finite = True
        act_quality = True; out_quality = True; out_max_quality = True
        lm_gates = {"top1": True, "top5": True, "kl": True}
        perf_ratio_pass = 0; perf_measured = 0; cold_honest = True

        for label in ("lm_head", "shared_up", "shared_down", "routed_up"):
            sel = reps[label]
            wraw = c3lib.tensor_raw(sel["weight"], entries, headers)
            sraw = c3lib.tensor_raw(sel["scale"], entries, headers)
            graw = c3lib.tensor_raw(sel["global"], entries, headers)
            b = c3lib.make_b(torch, wraw, sraw, parse_f32_scalar(graw), int(sel["N"]), int(sel["K"]))
            in_name, ref_name = FAMILY_IO[label]
            x_all = arrays[in_name]
            ref_all = arrays[ref_name]
            if int(x_all.shape[1]) != int(sel["K"]) or int(ref_all.shape[1]) != int(sel["N"]):
                raise RuntimeError(f"{label}: capture/checkpoint shape mismatch")
            by_m = {}
            for m in M_VALUES:
                groups = []
                lm_native_ids, lm_ref_ids = [], []
                lm_top5_sum = 0.0; lm_kl_sum = 0.0; lm_kl_max = 0.0; lm_rows = 0
                for gi, idx in enumerate(_group_indices(rows, m)):
                    x = x_all[idx].to("cuda").contiguous()
                    ref = ref_all[idx].to("cuda").contiguous()
                    q = quantize_activation(torch, x)
                    deq = dequantize_activation(torch, q, int(sel["K"]))
                    out = c3lib.native_call(torch, F, ST, SW, q, b)
                    torch.cuda.synchronize()
                    am = tensor_metrics(torch, deq, x)
                    om = tensor_metrics(torch, out.float(), ref)
                    finite = bool(torch.isfinite(out).all().item())
                    all_native_finite = all_native_finite and finite
                    rec = {"prompt_index": int(rows[idx[0]]["prompt_index"]), "rows": idx,
                           "finite": finite, "activation_metrics": am, "output_metrics": om,
                           "packed_sha256": _hash_tensor_bytes(torch, q["u8"]),
                           "natural_scale_sha256": _hash_tensor_bytes(torch, q["natural_scale"]),
                           "global_scale_f32": float(q["global"].item())}
                    if label == "lm_head":
                        lm = lm_distribution_metrics(torch, out.float(), ref)
                        rec["lm_distribution"] = lm
                        lm_native_ids.extend(lm["native_top1_ids"]); lm_ref_ids.extend(lm["reference_top1_ids"])
                        lm_top5_sum += lm["mean_top5_overlap"] * lm["rows"]
                        lm_kl_sum += lm["mean_kl_ref_to_native"] * lm["rows"]
                        lm_kl_max = max(lm_kl_max, lm["max_kl_ref_to_native"]); lm_rows += lm["rows"]
                    groups.append(rec)
                    del x, ref, q, deq, out
                agg = _aggregate(groups)
                if label == "lm_head":
                    agree = sum(a == b0 for a, b0 in zip(lm_native_ids, lm_ref_ids))
                    agg["lm_distribution"] = {"rows": lm_rows, "top1_agree": agree,
                        "top1_agreement_fraction": agree / lm_rows if lm_rows else 0.0,
                        "mean_top5_overlap": lm_top5_sum / lm_rows if lm_rows else 0.0,
                        "mean_kl_ref_to_native": lm_kl_sum / lm_rows if lm_rows else math.inf,
                        "max_kl_ref_to_native": lm_kl_max,
                        "native_top1_ids": lm_native_ids, "reference_top1_ids": lm_ref_ids}
                    lm_gates["top1"] &= agg["lm_distribution"]["top1_agreement_fraction"] >= LM_TOP1_MIN
                    lm_gates["top5"] &= agg["lm_distribution"]["mean_top5_overlap"] >= LM_TOP5_MIN
                    lm_gates["kl"] &= (agg["lm_distribution"]["mean_kl_ref_to_native"] <= LM_KL_MEAN_MAX
                                        and agg["lm_distribution"]["max_kl_ref_to_native"] <= LM_KL_MAX_MAX)
                act_quality &= agg["activation_min_cosine"] >= ACT_COS_MIN and agg["activation_max_normalized_rmse"] <= ACT_NRMSE_MAX
                out_quality &= agg["output_min_cosine"] >= OUT_COS_MIN and agg["output_max_normalized_rmse"] <= OUT_NRMSE_MAX
                out_max_quality &= agg["output_max_normalized_max_abs_error"] <= OUT_NMAX_MAX
                by_m[f"M{m}"] = {"aggregate": agg, "groups": groups}

            perf = _performance(torch, F, ST, SW, c3lib, x_all[:8].to("cuda").contiguous(), b, l2)
            if perf.get("status") == "measured":
                perf_measured += 1
                cold_honest &= float(perf["working_set_over_l2"]) >= COLD_L2_MULTIPLE
                if float(perf["M8_total_over_M1"]) <= M8_TOTAL_OVER_M1_MAX:
                    perf_ratio_pass += 1
            family_records[label] = {"selected": sel, "capture_input": in_name, "capture_reference": ref_name,
                                     "quality_by_M": by_m, "performance": perf}
            del b
            torch.cuda.empty_cache()

        reuse_ok = bool(capture.get("reuse_contract", {}).get("identical_by_construction")
                        and FAMILY_IO["shared_up"][0] == FAMILY_IO["routed_up"][0] == "moe_normed")
        gates = {
            "C3B_G1_capture_integrity": capture_integrity,
            "C3B_G2_parents_green": parents_green,
            "C3B_G3_activation_reuse_identity": reuse_ok,
            "C3B_G4_native_executes": all_native_finite,
            "C3B_G5_activation_quant_quality": act_quality,
            "C3B_G6_projection_quality": out_quality,
            "C3B_G7_projection_max_error": out_max_quality,
            "C3B_G8_lm_top1": lm_gates["top1"],
            "C3B_G9_lm_top5_overlap": lm_gates["top5"],
            "C3B_G10_lm_distribution": lm_gates["kl"],
            "C3B_P1_cold_honest": bool(perf_measured and cold_honest),
            "C3B_P2_M8_total_over_M1_le_2_at_least_3_of_4": perf_ratio_pass >= 3,
            "C3B_P3_reuse_cost_model": reuse_ok,
        }
        correctness = all(gates[k] for k in gates if k.startswith("C3B_G"))
        perf_signal = gates["C3B_P1_cold_honest"] and gates["C3B_P2_M8_total_over_M1_le_2_at_least_3_of_4"] and gates["C3B_P3_reuse_cost_model"]
        payload.update({
            "environment": environment_snapshot((Path(__file__), PREREG, CAPTURE, C3A, C3A_PRE, W4A8)),
            "capture_manifest_sha256": sha256_file(CAPTURE),
            "parents": {"c3a_status": c3a.get("status"), "c3a_v2_preflight": c3pre.get("status"), "w4a8_verdict": w4a8.get("verdict")},
            "reuse_cost_model": {
                "moe_layers": int(capture["v18"]["moe_layer_count"]),
                "top_k": int(capture["v18"]["top_k"]),
                "shared_up_calls": int(capture["v18"]["moe_layer_count"]),
                "routed_up_weight_calls": int(capture["v18"]["moe_layer_count"]) * int(capture["v18"]["top_k"]),
                "moe_normed_quantizations": int(capture["v18"]["moe_layer_count"]),
                "explanation": "one normed A quantization per MoE layer is reused by shared_up + all routed-up weights; routed-up is not charged one quantizer per expert",
            },
            "families": family_records,
            "gates": gates,
            "summary": {"quality_green": correctness, "performance_engineering_signal_green": perf_signal,
                        "performance_measured_families": perf_measured, "M8_ratio_pass_count": perf_ratio_pass,
                        "M8_total_over_M1": {k: v["performance"].get("M8_total_over_M1") for k, v in family_records.items()},
                        "lm_head_by_M": {m: family_records["lm_head"]["quality_by_M"][m]["aggregate"].get("lm_distribution") for m in ("M1","M2","M4","M8")}},
            "status": ("w4a4_real_activation_quality_candidate" if correctness
                       else "w4a4_real_activation_quality_failed"),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})
    write_json_atomic(RESULT, payload, archive=True)
    print(json.dumps({"status": payload.get("status"), "summary": payload.get("summary"),
                      "gates": payload.get("gates"), "error": (payload.get("error") or {}).get("message"),
                      "output": str(RESULT)}, indent=2))
    return 0 if payload.get("status") != "technical_failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
