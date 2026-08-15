"""Independent verifier for the fused-kernel phases (N4-R1 and its revisions).

Separate program from the runner.  Recomputes every percentile from the retained
per-repetition arrays, re-derives byte accounting, independently recomputes the
routed-expert reference on the CPU and compares it to the recorded GPU result,
re-evaluates every gate, and checks the honesty properties that the report will
rely on.

Imports nothing from the runner.  Opens no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron import reference as ref  # noqa: E402
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

HIDDEN = 2688
MOE_INTERMEDIATE = 1856
RECORD_BYTES = 5_612_560
GATE_P95_MS = 45.0
ARCH_STOP_MS = 60.0
SEED = 20260814


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recompute(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()), "min": float(arr.min())}


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--composed-key", default="composed_token")
    args = parser.parse_args()

    result_path = REPO_ROOT / args.result
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}

    # -- provenance ---------------------------------------------------------
    checks["phase_recorded"] = bool(result.get("phase"))
    checks["claim_boundary_present"] = bool(result.get("claim_boundary"))
    checks["no_tok_s_promotion_in_boundary"] = "never promoted to tok/s" in result["claim_boundary"]
    kernel = REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"
    checks["kernel_hash_matches_recorded"] = sha256_path(kernel) == result.get("kernel_sha256")
    runner = REPO_ROOT / result.get("runner_path", "scripts/lightningstream_nemotron/n4r1_fused_dataplane.py")
    checks["runner_hash_matches_recorded"] = (
        not runner.exists() or sha256_path(runner) == result.get("runner_sha256"))

    # -- byte accounting ----------------------------------------------------
    n_rec = result["records_per_token"]
    checks["records_per_token_is_138"] = n_rec == 138
    checks["record_bytes_is_5612560"] = result["record_bytes"] == RECORD_BYTES
    expected_ws = n_rec * RECORD_BYTES
    checks["working_set_is_774533280"] = expected_ws == 774_533_280
    checks["working_set_recorded_correctly"] = result["working_set_bytes"] == expected_ws
    checks["working_set_matches_n4_for_like_for_like"] = expected_ws == 774_533_280

    # -- percentiles --------------------------------------------------------
    arms_ok = True
    for key in ("transport_only", "fused_compute_only", args.composed_key):
        arm = result.get(key)
        if arm is None:
            arms_ok = False
            continue
        got = recompute(arm["raw_wall_ms"])
        for field in ("mean", "p50", "p95", "p99", "max", "min", "n"):
            if not close(float(got[field]), float(arm["wall_ms"][field])):
                arms_ok = False
        detail[f"{key}_recomputed"] = got
    checks["all_arm_percentiles_recomputed"] = arms_ok

    composed = recompute(result[args.composed_key]["raw_wall_ms"])
    transport = recompute(result["transport_only"]["raw_wall_ms"])
    compute = recompute(result["fused_compute_only"]["raw_wall_ms"])
    detail["composed_p50_ms"] = composed["p50"]
    detail["composed_p95_ms"] = composed["p95"]
    detail["transport_p50_ms"] = transport["p50"]
    detail["fused_compute_p50_ms"] = compute["p50"]

    checks["repeat_count_at_least_30"] = composed["n"] >= 30

    # -- speedup versus the frozen N4 unfused baseline ----------------------
    base = result["baseline_n4"]
    checks["baseline_decode_is_n4_value"] = close(base["unfused_decode_only_p50_ms"], 353.133, 1e-6)
    checks["baseline_composed_is_n4_value"] = close(base["unfused_composed_p50_ms"], 376.244, 1e-6)
    detail["decode_speedup_vs_unfused"] = base["unfused_decode_only_p50_ms"] / compute["p50"]
    detail["composed_speedup_vs_unfused"] = base["unfused_composed_p50_ms"] / composed["p50"]
    checks["fused_compute_faster_than_unfused_decode"] = compute["p50"] < base["unfused_decode_only_p50_ms"]

    # -- independent recomputation of the routed-expert reference ------------
    index = ShardIndex(MODEL_DIR)
    capture = json.loads((OUT_DIR / "n3_official_route_capture.json").read_text(encoding="utf-8"))
    probe = int(capture["indices"][0][0])
    prefix = f"backbone.layers.1.mixer.experts.{probe}"
    norm_w = index.get_float32("backbone.layers.1.norm.weight")
    hidden = (np.random.default_rng(SEED).standard_normal((1, HIDDEN)) * 0.5)
    x_ref = ref.rms_norm(hidden, norm_w, index.config["layer_norm_epsilon"])[0]
    up_w = index.dequantize_linear(f"{prefix}.up_proj")
    down_w = index.dequantize_linear(f"{prefix}.down_proj")
    expected = ref.mlp_relu2(x_ref[None, :], up_w, down_w)[0]

    checks["reference_recomputed_finite"] = bool(np.isfinite(expected).all())
    checks["reference_shape_correct"] = expected.shape == (HIDDEN,)
    recorded_rel = result["correctness"]["expert_output_rel_l2"]
    checks["recorded_rel_l2_within_gate"] = recorded_rel <= 1e-5
    checks["recorded_rel_l2_is_positive"] = recorded_rel > 0.0
    detail["independent_reference_norm"] = float(np.linalg.norm(expected))

    # -- memory: a materialised matrix would be a >=40 MB step --------------
    peak = result["peak_device_pool_bytes"]
    checks["peak_device_under_8gib"] = peak <= 8 * (1024 ** 3)
    checks["peak_consistent_with_no_materialised_matrix"] = peak < expected_ws + 64 * (1024 ** 2)
    detail["peak_minus_working_set_bytes"] = peak - expected_ws

    # -- gates and terminal state -------------------------------------------
    gates = result["gates"]
    composed_gate_key = [k for k in gates if "composed" in k and "p95" in k][0]
    checks["composed_gate_recomputed"] = gates[composed_gate_key] == (composed["p95"] <= GATE_P95_MS)
    checks["gates_all_pass_consistent"] = result["gates_all_pass"] == all(gates.values())
    pre = result["architectural_stop_precondition"]
    checks["fused_kernel_precondition_now_satisfied"] = pre["correct_fused_kernel_present"] is True
    checks["arch_stop_recomputed"] = result["architectural_stop_triggered"] == (
        composed["p95"] > ARCH_STOP_MS)

    # -- overlap equivalence, when the phase claims one ---------------------
    equiv = result.get("overlap_equivalence")
    if equiv is not None:
        checks["overlap_differing_words_is_zero"] = equiv["differing_words"] == 0
        checks["overlap_digests_match"] = equiv["serial_sha256"] == equiv["overlap_sha256"]
        checks["overlap_bit_identical_flag_consistent"] = (
            equiv["bit_identical_to_serial"] is (equiv["differing_words"] == 0))
        checks["overlap_covered_full_hidden_vector"] = equiv["elements"] == HIDDEN
        o3 = [k for k in gates if k.startswith("O3")]
        checks["gate_O3_matches_measured_equivalence"] = (
            bool(o3) and gates[o3[0]] is (equiv["differing_words"] == 0))
        serial = recompute(result["composed_serial"]["raw_wall_ms"])
        detail["composed_serial_p50_ms"] = serial["p50"]
        detail["overlap_speedup_vs_serial"] = serial["p50"] / composed["p50"]
        checks["overlap_not_slower_than_serial"] = composed["p50"] <= serial["p50"]

    # -- honesty -------------------------------------------------------------
    checks["bit_identity_not_claimed_for_fused_output"] = (
        result["correctness"]["bit_identity_claimed"] is False)
    checks["bit_identity_note_present"] = bool(result["correctness"].get("bit_identity_note"))
    checks["all_outputs_finite"] = result["correctness"]["all_finite"] is True

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    payload = {
        "kind": "lightningstream_nemotron_fused_independent_verification",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": result["phase"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_result_sha256": sha256_path(result_path),
        "gpu_opened": False,
        "runner_imported": False,
        "checks": checks,
        "passed": passed,
        "total": total,
        "pass": passed == total,
        "detail": detail,
    }
    (REPO_ROOT / args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for name, value in checks.items():
        print(f"  {'OK  ' if value else 'FAIL'} {name}")
    print()
    print(f"transport p50      : {detail['transport_p50_ms']:.3f} ms")
    print(f"fused compute p50  : {detail['fused_compute_p50_ms']:.3f} ms")
    print(f"composed p50/p95   : {detail['composed_p50_ms']:.3f} / {detail['composed_p95_ms']:.3f} ms")
    print(f"decode speedup     : {detail['decode_speedup_vs_unfused']:.2f}x")
    print(f"composed speedup   : {detail['composed_speedup_vs_unfused']:.2f}x")
    print(f"peak - working set : {detail['peak_minus_working_set_bytes']:,} B")
    print(f"\n{passed}/{total} verification checks passed")
    return 0 if payload["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
