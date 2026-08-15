from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
OUTPUT = R / "final_closure_verification_2026-08-12.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    checks: dict[str, bool] = {}
    p9av = load("p9a_mixed_q4_q5_validation.json")
    p9at = load("p9a_mixed_q4_q5_test.json")
    checks["p9a_split_and_status"] = p9av["split"] == "validation" and p9at["split"] == "test" and p9at["status"] == "quality_pass" and p9at["overall_pass"]
    checks["p9a_fixed_layers_and_ratio"] = p9at["q4_layers"] == [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17] and close(p9at["expert_code_byte_ratio_vs_uniform_q5"], 0.95)
    checks["p9a_gates"] = all(p9av["gates"].values()) and all(p9at["gates"].values()) and p9av["relative_cross_entropy_increase"] <= 0.025 and p9at["relative_cross_entropy_increase"] <= 0.02

    p9bv = load("p9b_structured_wanda_validation.json")
    p9bt = load("p9b_structured_wanda_test.json")
    masks_path = ROOT / "reports/runs/streamq5_moe/p9b_structured_wanda_keep.safetensors"
    masks = load_file(masks_path)
    mask_ok = set(masks) == {f"layer_{i:02d}" for i in range(48)}
    if mask_ok:
        for value in masks.values():
            work = value.long()
            mask_ok &= work.shape == (128, 384) and int(work.min()) >= 0 and int(work.max()) < 768
            mask_ok &= bool((work.sort(dim=1).values[:, 1:] > work.sort(dim=1).values[:, :-1]).all())
    checks["p9b_masks_48x128x384_unique"] = mask_ok
    checks["p9b_provenance"] = p9bv["inputs"]["masks_sha256"] == sha256(masks_path) == p9bt["inputs"]["masks_sha256"]
    checks["p9b_quality_pass"] = p9bv["status"] == "validation_pass_test_authorized" and p9bt["status"] == "quality_pass" and p9bt["overall_pass"] and all(p9bv["gates"].values()) and all(p9bt["gates"].values())

    p9c = load("p9c_compact_regroup_q5_validation.json")
    checks["p9c_preregistered_negative"] = p9c["status"] == "validation_closed" and not p9c["overall_pass"] and not p9c["gates"]["relative_ce_gate"] and not p9c["gates"]["top1_ge_90pct"]
    checks["p9c_no_test_opened"] = not (R / "p9c_compact_regroup_q5_test.json").exists()
    checks["p9c_mask_provenance"] = p9c["inputs"]["masks_sha256"] == sha256(masks_path)

    p10d = load("p10d_gpu_router.json")
    checks["p10d_correctness"] = p10d["correctness"]["exact_id_vectors"] == 480 and p10d["correctness"]["exact_bf16_weight_vectors"] == 480 and close(p10d["correctness"]["maximum_weight_abs_error"], 0.0)
    checks["p10d_negative_timing"] = not p10d["overall_pass"] and not p10d["gates"]["host_p50_le_90pct"] and not p10d["gates"]["host_p95_le_90pct"] and p10d["timing_ms"]["host_p50_ratio"] > 1.0 and p10d["timing_ms"]["host_p95_ratio"] > 1.0

    p14v = load("p14a_deepseek_v2_lite_q5_validation.json")
    p14t = load("p14a_deepseek_v2_lite_q5_test.json")
    checks["p14a_quality_pass"] = p14v["status"] == "validation_pass_test_authorized" and p14t["status"] == "quality_pass" and p14t["overall_pass"] and len(p14v["layers"]) == len(p14t["layers"]) == 26 and all(p14v["gates"].values()) and all(p14t["gates"].values())

    p15 = load("p15a_llama_cpp_cpu_baseline.json")
    gguf = ROOT / p15["inputs"]["model_gguf"]
    checks["p15_artifact_hash_size"] = gguf.stat().st_size == p15["inputs"]["model_gguf_bytes"] and sha256(gguf).lower() == p15["inputs"]["model_gguf_sha256"].lower()
    observed_cpu_tps = sum(p15["decode_128"]["samples_tokens_per_second"]) / 3
    checks["p15_decode_arithmetic"] = close(observed_cpu_tps, p15["decode_128"]["average_tokens_per_second"], 1e-5) and close(p15["comparison"]["p13c_to_llama_cpp_cpu_ratio"], p15["comparison"]["p13c_tokens_per_second"] / p15["decode_128"]["average_tokens_per_second"])
    checks["p15_cpu_only_boundary"] = p15["environment"]["gpu_layers"] == 0 and "CPU-only" in p15["limitations"][0]

    p16 = load("p16a_10x_quality.json")
    bootstrap = p16["bootstrap_relative_ce"]
    checks["p16_workload_and_gates"] = p16["data"]["contexts"] == 100 and p16["data"]["labels"] == 12700 and len(p16["layers"]) == 48 and p16["overall_pass"] and all(p16["gates"].values())
    checks["p16_bootstrap_order"] = bootstrap["resamples"] == 10000 and bootstrap["p2_5"] <= bootstrap["p50"] <= bootstrap["p97_5"] <= 0.025

    p13 = load("p13c_evt_pm_32g_endurance.json")
    power = [row["gpu"]["power_w"] for row in p13["telemetry"]]
    gpu_j = sum(power) / len(power) / p13["tokens_per_second"]
    checks["p13_energy_projection"] = len(power) == 40 and close(gpu_j, 3.3951930054855328, 1e-10)
    checks["p13_endurance_still_passes"] = p13["overall_pass"] and len(p13["wall_ms"]) == 10000 and all(p13["exactness_vs_p12r2"].values())

    registry = (R / "ALL_IDEAS_CLOSURE_REGISTRY_2026-08-12.yaml").read_text(encoding="utf-8")
    ids = re.findall(r"^  - id: (I\d{3})$", registry, flags=re.MULTILINE)
    statuses = re.findall(r"^    status: ([a-z_]+)$", registry, flags=re.MULTILINE)
    checks["registry_48_unique_ideas"] = len(ids) == 48 and len(set(ids)) == 48 and ids == [f"I{i:03d}" for i in range(1, 49)]
    checks["registry_no_queued"] = "status: queued" not in registry and len(statuses) == 48
    checks["registry_local_closed_world_claim_false"] = "all_local_testable_closed: true" in registry and "breakthrough_claim_allowed: false" in registry

    evidence_names = re.findall(r"^    evidence: (.+)$", registry, flags=re.MULTILINE)
    checks["registry_evidence_exists"] = all((ROOT / name.strip()).exists() for name in evidence_names)

    result = {
        "kind": "streamq5_moe_final_closure_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256(Path(__file__)),
        "checks": checks,
        "summary": {"passed": sum(checks.values()), "total": len(checks), "all_pass": all(checks.values())},
        "claim_boundary": "Independent arithmetic, provenance, gate and registry audit; this does not create external reproduction or world-SOTA evidence.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if not all(checks.values()):
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
