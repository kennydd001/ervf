"""C3A: real Lightning NVFP4 checkpoint bytes -> native SM120 scaled_mm."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, require_gpu_free, utc_now, write_json_atomic
from native_nvfp4_c3a_lib import (
    COLD_L2_MULTIPLE, INDEX, M_VALUES, all_nvfp4_triples, choose_representatives,
    load_index_headers, run_family, two_level_smoke,
)

C1 = REPO / "pro_research" / "results" / "native_nvfp4" / "C1_REPACK_AUDIT.json"
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_REAL_WEIGHT.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_C3A_REAL_WEIGHT_PREREGISTRATION.md"
NRMSE_MAX, COSINE_MIN, NMAX_MAX = 0.020, 0.9990, 0.050
M8_ROW_NMAX_MAX, M8_OVER_M1_MAX = 0.005, 1.15


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c3a_real_weight", "status": "started", "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": ("real checkpoint B representation + exact +1 A and cold M geometry only; "
                           "no real-activation quantization, grouped-MoE, full-model quality or end-to-end tok/s claim"),
        "thresholds": {"normalized_rmse_max": NRMSE_MAX, "cosine_min": COSINE_MIN,
            "normalized_max_abs_error_max": NMAX_MAX,
            "M8_identical_rows_normalized_max_diff_max": M8_ROW_NMAX_MAX,
            "cold_working_set_over_l2_min": COLD_L2_MULTIPLE, "M8_over_M1_p50_max": M8_OVER_M1_MAX},
    }
    try:
        require_gpu_free()
        import torch
        import torch.nn.functional as F
        ST, SW = F.ScalingType, F.SwizzleType
        cap = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (-1, -1)
        api_ok = bool(str(torch.__version__).startswith("2.12.1") and str(torch.version.cuda).startswith("13.2")
            and torch.cuda.is_available() and cap >= (12, 0) and hasattr(F, "scaled_mm")
            and hasattr(torch, "float4_e2m1fn_x2") and hasattr(ST, "BlockWise1x16")
            and hasattr(ST, "TensorWise") and hasattr(SW, "SWIZZLE_32_4_4") and hasattr(SW, "NO_SWIZZLE"))
        c1 = json.loads(C1.read_text(encoding="utf-8"))
        c1_ok = c1.get("status") in {"repack_lossless", "repack_lossless_high_padding"} and all(
            bool(v) for k, v in (c1.get("gates") or {}).items() if k.startswith("C1_G"))
        entries, headers = load_index_headers(); selected = choose_representatives(all_nvfp4_triples(entries, headers))
        metadata_ok = len(selected) == 4 and all(
            x["weight_shape"] == [x["N"], x["K"] // 2] and x["scale_shape"] == [x["N"], x["K"] // 16]
            and x["weight_dtype"] == "U8" and x["scale_dtype"] == "F8_E4M3" and x["global_dtype"] == "F32"
            for x in selected)
        smoke = two_level_smoke(torch, F, ST, SW) if api_ok else {"all_equal_expected": False}
        smoke_ok = bool(smoke.get("finite") and smoke.get("all_equal_expected"))
        l2 = 0
        gpu_name = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            l2 = int(getattr(props, "L2_cache_size", 0) or getattr(props, "l2_cache_size", 0) or 0)
            gpu_name = torch.cuda.get_device_name(0)
        l2_ok = l2 > 0
        payload["environment"] = environment_snapshot((Path(__file__), PREREG, INDEX, C1))
        payload["api"] = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
            "gpu": gpu_name, "capability": list(cap), "l2_bytes": l2}
        payload["parents"] = {"C1_status": c1.get("status")}; payload["selection"] = selected; payload["two_level_smoke"] = smoke
        pre = {"C3A_G1_environment_and_api": api_ok and l2_ok, "C3A_G2_C1_parent_green": c1_ok,
               "C3A_G3_four_real_triples_match_contract": metadata_ok, "C3A_G4_two_level_known_value_smoke": smoke_ok}
        payload["gates"] = dict(pre)
        if not all(pre.values()):
            payload.update({"status": "precondition_failed", "completed_utc": utc_now()})
            write_json_atomic(OUT, payload, archive=True)
            print(json.dumps({"status": payload["status"], "gates": payload["gates"], "output": str(OUT)}, indent=2)); return 0

        fams = [run_family(torch, F, ST, SW, x, entries, headers, l2) for x in selected]; payload["families"] = fams
        native_exec = all(all(f["native"][f"M{m}"]["finite"] for m in M_VALUES) for f in fams)
        ref_ok = all(f["native"]["M1"]["reference_metrics_first_row"]["normalized_rmse"] <= NRMSE_MAX
                     and f["native"]["M1"]["reference_metrics_first_row"]["cosine"] >= COSINE_MIN for f in fams)
        max_ok = all(f["native"]["M1"]["reference_metrics_first_row"]["normalized_max_abs_error"] <= NMAX_MAX for f in fams)
        m8_rows = all(f["M8_identical_rows_normalized_max_diff"] <= M8_ROW_NMAX_MAX for f in fams)
        measured = [f for f in fams if f["cold_timing"].get("status") == "measured"]
        cold_ok = bool(measured) and all(f["cold_timing"]["working_set_over_l2"] >= COLD_L2_MULTIPLE for f in measured)
        ratio_pass = sum(1 for f in measured if float(f["cold_timing"]["M8_over_M1"]) <= M8_OVER_M1_MAX)
        lm = next(f for f in fams if f["label"] == "lm_head"); lmt = lm["cold_timing"]
        lm_ok = lmt.get("status") != "measured" or float(lmt["M8_over_M1"]) <= M8_OVER_M1_MAX
        gates = dict(pre); gates.update({
            "C3A_G5_real_M1_M8_execute_finite": native_exec, "C3A_G6_reference_nrmse_and_cosine": ref_ok,
            "C3A_G7_reference_normalized_max_abs": max_ok, "C3A_G8_M8_identical_rows_agree": m8_rows,
            "C3A_G9_checkpoint_hashes_reverified": None,
            "C3A_P1_cold_rotation_ge_4x_L2": cold_ok,
            "C3A_P2_M8_over_M1_le_1_15_at_least_3_of_4": ratio_pass >= 3,
            "C3A_P3_lm_head_M8_over_M1_le_1_15_if_measured": lm_ok})
        payload["gates"] = gates
        correctness = all(bool(gates[k]) for k in (
            "C3A_G1_environment_and_api", "C3A_G2_C1_parent_green", "C3A_G3_four_real_triples_match_contract",
            "C3A_G4_two_level_known_value_smoke", "C3A_G5_real_M1_M8_execute_finite",
            "C3A_G6_reference_nrmse_and_cosine", "C3A_G7_reference_normalized_max_abs", "C3A_G8_M8_identical_rows_agree"))
        perf = bool(cold_ok and ratio_pass >= 3 and lm_ok)
        payload["summary"] = {"correctness_green_before_independent_hash_verifier": correctness,
            "performance_green": perf, "cold_measured_families": len(measured), "M8_ratio_pass_count": ratio_pass,
            "M8_over_M1": {f["label"]: f["cold_timing"].get("M8_over_M1") for f in fams},
            "normalized_rmse_M1": {f["label"]: f["native"]["M1"]["reference_metrics_first_row"]["normalized_rmse"] for f in fams},
            "cosine_M1": {f["label"]: f["native"]["M1"]["reference_metrics_first_row"]["cosine"] for f in fams}}
        payload["status"] = ("real_weight_representation_and_geometry_candidate" if correctness and perf
            else "real_weight_representation_green_perf_miss" if correctness else "real_weight_representation_failed")
        payload["completed_utc"] = utc_now()
    except Exception as exc:
        payload.update({"status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now()})
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({"status": payload.get("status"), "summary": payload.get("summary"), "gates": payload.get("gates"),
                      "output": str(OUT), "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") != "technical_failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
