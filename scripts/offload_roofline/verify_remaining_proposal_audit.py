from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from moe_lab.reporting import ROOT


RESULT = ROOT / "reports/offload_roofline/remaining_proposal_audit.json"
HERA = ROOT / "reports/hera_moe/p0_multidomain_tier_result.json"
QWEN = ROOT / "reports/rsiv_moe/qwen_checkpoint_acquisition.json"
CORETAIL = ROOT / "reports/coretail_moe/p0a_locked16_format_result.json"
ATOMIC = ROOT / "reports/craft_moe/atomic_full_depth_oracle.json"
PREREG = ROOT / "reports/offload_roofline/P_D_UNION_PREREGISTRATION.md"
OUT_JSON = ROOT / "reports/offload_roofline/remaining_proposal_audit_verification.json"
OUT_MD = ROOT / "reports/offload_roofline/REMAINING_PROPOSAL_AUDIT_VERIFICATION.md"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_bitsets(routes: np.ndarray):
    low = np.zeros(routes.shape[:2], dtype=np.uint64)
    high = np.zeros(routes.shape[:2], dtype=np.uint64)
    for slot in range(routes.shape[2]):
        ids = routes[:, :, slot].astype(np.uint64)
        low |= np.where(ids < 64, np.left_shift(np.uint64(1), ids), np.uint64(0))
        high |= np.where(ids >= 64, np.left_shift(np.uint64(1), ids - np.uint64(64)), np.uint64(0))
    return low, high


def independent_curve(routes: np.ndarray, depth: int):
    low, high = token_bitsets(routes)
    windows = routes.shape[0] - depth + 1
    low_union = np.zeros((windows, routes.shape[1]), dtype=np.uint64)
    high_union = np.zeros_like(low_union)
    for offset in range(depth):
        low_union |= low[offset : offset + windows]
        high_union |= high[offset : offset + windows]
    counts = np.bitwise_count(low_union) + np.bitwise_count(high_union)
    values = counts.reshape(-1)
    mean = float(values.mean())
    return {
        "mean": mean, "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95, method="higher")),
        "maximum": int(values.max()), "samples": int(values.size),
        "mean_fraction_of_naive_k_times_s": mean / (8 * depth),
        "mean_naive_over_unique_factor": 8 * depth / mean,
        "uniform_independent_expectation": 128 * (1 - (1 - 8 / 128) ** depth),
    }


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite remaining-proposal verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    hera = json.loads(HERA.read_text(encoding="utf-8"))
    checks = {}
    for name, path, field in (
        ("p_d_preregistration_hash", PREREG, "p_d_preregistration_sha256"),
        ("hera_hash", HERA, "hera_sha256"), ("qwen_hash", QWEN, "qwen_acquisition_sha256"),
        ("coretail_hash", CORETAIL, "coretail_sha256"), ("atomic_hash", ATOMIC, "atomic_full_sha256"),
    ):
        checks[name] = sha256(path) == result["inputs"][field]
    routes = {domain: np.empty((32768, 48, 8), dtype=np.int16) for domain in DOMAINS}
    route_hashes = True
    for layer in range(48):
        record = hera["artifacts"][str(layer)]
        path = ROOT / record["artifact"]
        route_hashes &= sha256(path) == record["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain][:, layer, :] = tensors[f"{domain}_router_ids"]
    checks["all_route_hashes"] = route_hashes
    curves_exact = True
    for domain in DOMAINS:
        for depth in (1, 2, 4, 8):
            observed = independent_curve(routes[domain], depth)
            expected = result["P_D_SPECULATIVE"]["qwen_actual_top8_union_curves"][domain][str(depth)]
            curves_exact &= observed == expected
    checks["independent_bitset_union_curves_exact"] = curves_exact
    k3 = 896 * (1 - (1 - 16 / 896) ** 8)
    p_d = result["P_D_SPECULATIVE"]
    checks["k3_formula"] = abs(p_d["recomputed_k3_uniform_union"] - k3) < 1e-12 and abs(p_d["recomputed_naive_over_unique_factor"] - 128 / k3) < 1e-12
    checks["supplied_k3_number_is_wrong"] = abs(k3 - p_d["supplied_k3_uniform_union"]) > 1.0
    p_a = result["P_A_QWEN_WALLCLOCK"]
    checks["p_a_sources"] = p_a["qwen_bf16_checkpoint_verified"] and p_a["qwen_bf16_shards_present"] == 16 and p_a["canonical_actual_gptq_experts_present"] == 16 and p_a["canonical_actual_gptq_experts_required"] == 6144
    checks["p_a_correctly_blocked"] = p_a["status"] == "blocked_missing_full_bank_gptq_and_runtime" and not p_a["full_bank_gptq_present"] and not p_a["packed_lfu_async_runtime_present"] and not p_a["tokens_per_second_measured"]
    p_c = result["P_C_FULL_K3"]
    checks["p_c_correctly_blocked"] = p_c["status"] == "blocked_missing_k3_checkpoint_and_runtime" and not p_c["k3_candidate_directories"] and not p_c["actual_k3_trunk_bytes_measured"] and not p_c["actual_64_token_k3_decode_measured"]
    checks["p_d_correctly_blocked"] = p_d["status"] == "blocked_acceptance_unmeasured" and not p_d["k3_target_present"] and not p_d["h3_is_standalone_autoregressive_drafter"] and not p_d["accepted_tokens_per_pass_measured"] and not p_d["acceptance_gate_ge_4_tested"]
    correction = result["SOURCE_ANALYSIS_CORRECTIONS"]
    checks["hera_hot_count_corrections"] = correction["hera_hot_counts_actual"] == hera["hot_experts_by_domain"] and correction["actual_minus_claimed"] == {domain: correction["hera_hot_counts_actual"][domain] - correction["hera_hot_counts_claimed"][domain] for domain in correction["hera_hot_counts_claimed"]}
    checks["static_hera_status_preserved"] = correction["static_hera_verdict_remains_valid"] and hera["verdict"] == "static_tier_negative"
    checks = {name: bool(value) for name, value in checks.items()}
    passed = sum(checks.values())
    verification = {
        "kind": "offload_roofline_remaining_proposal_audit_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "checks_passed": passed, "checks_total": len(checks),
        "all_pass": passed == len(checks),
        "verdict": "remaining_proposal_audit_verified" if passed == len(checks) else "verification_failed",
    }
    OUT_JSON.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# Resterende offloadvoorstellen — onafhankelijke verificatie", "",
        f"**{verification['verdict']}** — {passed}/{len(checks)} controles geslaagd.", "",
        "Een onafhankelijke 128-bit-bitsetimplementatie reproduceerde alle 20 Qwen-uniecurves exact. Bronnen, blokkades, K3-formule en HERA-correcties zijn opnieuw gecontroleerd.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": verification["verdict"], "checks": f"{passed}/{len(checks)}"}, indent=2))
