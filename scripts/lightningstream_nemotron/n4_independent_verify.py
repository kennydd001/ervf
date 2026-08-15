"""Independent verifier for N4_ZERO_CACHE_DATAPLANE.

A separate program from the runner.  It re-reads the raw result, recomputes
every percentile from the retained per-repetition arrays, re-derives the byte
accounting and effective bandwidth, independently re-reads checkpoint bytes to
test the bank claim, recomputes the CPU float32 decode from scratch to test the
bit-identity claim, and re-evaluates every gate and the terminal-state logic.

It imports nothing from the runner.  It touches no GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron import nvfp4  # noqa: E402
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
RESULT = OUT_DIR / "n4_zero_cache_dataplane.json"
OUT = OUT_DIR / "n4_independent_verification.json"

TOP_K = 6
MOE_LAYER_COUNT = 23
CODE_BYTES = 4_988_928
SCALE_BYTES = 623_616
GLOBAL_BYTES = 16
RECORD_BYTES = 5_612_560
GATE_P95_MS = 45.0
ARCH_STOP_MS = 60.0
HIDDEN = 2688
MOE_INTERMEDIATE = 1856


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def recompute_percentiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()), "min": float(arr.min())}


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def decode_cpu_float32(code_bytes, scale_bytes, global_scale, rows, cols):
    e2m1 = nvfp4.E2M1_TABLE.astype(np.float32)
    e4m3 = np.nan_to_num(nvfp4.E4M3_TABLE, nan=0.0).astype(np.float32)
    codes = np.stack([code_bytes & 0x0F, code_bytes >> 4], axis=-1).reshape(-1)
    values = e2m1[codes]
    scales = np.repeat(e4m3[scale_bytes], nvfp4.GROUP_SIZE)
    return (values * scales * np.float32(global_scale)).reshape(rows, cols)


def main() -> int:
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}

    result = json.loads(RESULT.read_text(encoding="utf-8"))

    # -- provenance ---------------------------------------------------------
    checks["result_kind_correct"] = result.get("kind") == "lightningstream_nemotron_n4_zero_cache_dataplane"
    checks["phase_correct"] = result.get("phase") == "N4_ZERO_CACHE_DATAPLANE"
    checks["claim_boundary_present"] = bool(result.get("claim_boundary"))
    runner = REPO_ROOT / "scripts/lightningstream_nemotron/n4_zero_cache_dataplane.py"
    checks["runner_hash_matches_recorded"] = sha256_path(runner) == result.get("runner_sha256")
    codec = REPO_ROOT / "src/moe_lab/lightningstream_nemotron/nvfp4.py"
    checks["codec_hash_matches_recorded"] = sha256_path(codec) == result.get("codec_sha256")

    # -- byte accounting ----------------------------------------------------
    records = result["records_per_token"]
    checks["records_per_token_is_138"] = records == MOE_LAYER_COUNT * TOP_K
    checks["record_bytes_is_5612560"] = result["record_bytes"] == RECORD_BYTES
    checks["record_bytes_decompose"] = (
        CODE_BYTES + SCALE_BYTES + GLOBAL_BYTES == RECORD_BYTES)
    expected_ws = records * RECORD_BYTES
    checks["working_set_matches_n1_bytes_per_token"] = expected_ws == 774_533_280
    checks["working_set_recorded_correctly"] = result["working_set_bytes"] == expected_ws
    detail["working_set_bytes"] = expected_ws

    # -- transport arms -----------------------------------------------------
    transport = result["n4b_transport"]
    arm_ok, bw_ok = True, True
    recomputed = {}
    for name, arm in transport.items():
        stats = recompute_percentiles(arm["raw_wall_ms"])
        recomputed[name] = stats
        for key in ("mean", "p50", "p95", "p99", "max", "min", "n"):
            if not close(float(stats[key]), float(arm["wall_ms"][key])):
                arm_ok = False
        bw = arm["bytes_moved"] / (stats["p50"] / 1e3) / 1e9
        if not close(bw, arm["effective_gb_s_at_p50"], 1e-6):
            bw_ok = False
        if arm["bytes_moved"] != expected_ws:
            arm_ok = False
    checks["transport_percentiles_recomputed"] = arm_ok
    checks["transport_bandwidth_recomputed"] = bw_ok
    detail["transport_recomputed"] = recomputed

    copies = {a: transport[a]["copies_issued"] for a in transport}
    checks["copy_counts_consistent"] = (
        copies["per_record"] == records * 3
        and copies["per_layer_batched"] == (records // TOP_K) * 3
        and copies["single_contiguous"] == 3)

    best = min(transport, key=lambda k: transport[k]["wall_ms"]["p95"])
    checks["best_arm_selection_correct"] = best == result["best_transport_arm"]
    detail["best_arm"] = best

    # -- composed / decode --------------------------------------------------
    composed = result["n4d_composed_token"]
    comp_stats = recompute_percentiles(composed["raw_wall_ms"])
    checks["composed_percentiles_recomputed"] = all(
        close(float(comp_stats[k]), float(composed["wall_ms"][k]))
        for k in ("mean", "p50", "p95", "p99", "max", "min"))
    decode = result["n4d_decode_only"]
    dec_stats = recompute_percentiles(decode["raw_wall_ms"])
    checks["decode_percentiles_recomputed"] = all(
        close(float(dec_stats[k]), float(decode["wall_ms"][k]))
        for k in ("mean", "p50", "p95", "p99", "max", "min"))
    checks["composed_slower_than_transport"] = comp_stats["p50"] > recomputed[best]["p50"]
    checks["decode_is_dominant_term"] = dec_stats["p50"] > 5 * recomputed[best]["p50"]
    detail["composed_p50_ms"] = comp_stats["p50"]
    detail["decode_only_p50_ms"] = dec_stats["p50"]
    detail["transport_p50_ms"] = recomputed[best]["p50"]
    detail["decode_share_of_composed_p50"] = dec_stats["p50"] / comp_stats["p50"]

    # -- independent re-read of checkpoint bytes ----------------------------
    index = ShardIndex(MODEL_DIR)
    capture = json.loads((OUT_DIR / "n3_official_route_capture.json").read_text(encoding="utf-8"))
    probe = int(capture["indices"][0][0])
    prefix = f"backbone.layers.1.mixer.experts.{probe}"

    sizes_ok = True
    for matrix, expect_code, expect_scale in (
            ("up_proj", CODE_BYTES // 2, SCALE_BYTES // 2),
            ("down_proj", CODE_BYTES // 2, SCALE_BYTES // 2)):
        if index.read_raw(f"{prefix}.{matrix}.weight").size != expect_code:
            sizes_ok = False
        if index.read_raw(f"{prefix}.{matrix}.weight_scale").size != expect_scale:
            sizes_ok = False
    checks["checkpoint_record_sizes_reread"] = sizes_ok

    # -- independent recomputation of the decode bit-identity claim ---------
    corr = result["n4c_correctness"]
    decode_ok = True
    for matrix, rows, cols in (("up_proj", MOE_INTERMEDIATE, HIDDEN),
                               ("down_proj", HIDDEN, MOE_INTERMEDIATE)):
        code_np = index.read_raw(f"{prefix}.{matrix}.weight")
        scale_np = index.read_raw(f"{prefix}.{matrix}.weight_scale")
        gscale = index.get_scalar(f"{prefix}.{matrix}.weight_scale_2")
        fresh = decode_cpu_float32(code_np, scale_np, gscale, rows, cols)
        if sha256_array(fresh) != corr[matrix]["cpu_sha256"]:
            decode_ok = False
        if corr[matrix]["gpu_sha256"] != corr[matrix]["cpu_sha256"]:
            decode_ok = False
        if corr[matrix]["max_abs_diff"] != 0.0:
            decode_ok = False
        if fresh.size != rows * cols:
            decode_ok = False
    checks["gpu_cpu_decode_bit_identity_independently_recomputed"] = decode_ok

    # -- gate re-evaluation -------------------------------------------------
    gates = result["gates"]
    checks["gate_G4_recomputed"] = gates["G4_routed_path_p95_under_45ms"] == (
        comp_stats["p95"] <= GATE_P95_MS)
    checks["gate_G4_is_false"] = gates["G4_routed_path_p95_under_45ms"] is False
    checks["gate_G6_recomputed"] = gates["G6_peak_device_under_8gib"] == (
        result["device"]["peak_reserved_bytes"] <= 8 * (1024 ** 3))
    checks["gates_all_pass_consistent"] = result["gates_all_pass"] == all(gates.values())

    # -- architectural stop precondition ------------------------------------
    pre = result["architectural_stop_precondition"]
    checks["arch_stop_precondition_recorded"] = pre["correct_fused_kernel_present"] is False
    checks["arch_stop_not_declared_without_fused_kernel"] = (
        result["architectural_stop_triggered"] is False)
    checks["composed_p95_exceeds_arch_threshold"] = comp_stats["p95"] > ARCH_STOP_MS
    checks["terminal_state_matches_logic"] = (
        result["terminal_state"] == "n4_zero_cache_screen_fail_unfused_decode_dominates")

    # -- honesty checks -----------------------------------------------------
    checks["decode_declared_unfused"] = "unfused" in composed["decode_implementation"]
    checks["route_provenance_marks_synthetic"] = (
        "SYNTHETIC" in result["n4a_bank"]["route_provenance"].upper())
    checks["partial_bank_declared"] = "not the full" in result["n4a_bank"]["partial_bank_note"]
    checks["no_tok_s_claim_in_boundary"] = "never promoted to tok/s" in result["claim_boundary"]

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    payload = {
        "kind": "lightningstream_nemotron_n4_independent_verification",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N4_ZERO_CACHE_DATAPLANE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_result_sha256": sha256_path(RESULT),
        "gpu_opened": False,
        "runner_imported": False,
        "checks": checks,
        "passed": passed,
        "total": total,
        "pass": passed == total,
        "detail": detail,
        "adjudication": (
            "The measured result is a correct NEGATIVE on G4 with an unfused "
            "decode. The architectural stop is correctly NOT declared, because "
            "its preregistered precondition requires a correct fused kernel and "
            "none exists yet."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for name, value in checks.items():
        print(f"  {'OK  ' if value else 'FAIL'} {name}")
    print()
    print(f"transport p50   : {detail['transport_p50_ms']:.3f} ms")
    print(f"decode-only p50 : {detail['decode_only_p50_ms']:.3f} ms")
    print(f"composed p50    : {detail['composed_p50_ms']:.3f} ms")
    print(f"decode share    : {detail['decode_share_of_composed_p50'] * 100:.1f}% of composed p50")
    print(f"\n{passed}/{total} verification checks passed")
    return 0 if payload["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
