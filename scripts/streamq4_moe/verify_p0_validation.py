from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import load_file

from moe_lab.reporting import ROOT


REPORT = ROOT / "reports/streamq4_moe"
PREREG = REPORT / "P0_MODEL_QUALITY_PREREGISTRATION.md"
LOCK_PATH = REPORT / "p0_input_lock.json"
EVALUATOR_LOCK_PATH = REPORT / "p0_evaluator_lock.json"
RESULT_PATH = REPORT / "p0_validation_model_quality.json"
EVALUATOR = ROOT / "scripts/streamq4_moe/run_p0_model_quality.py"
OLD_INPUT = ROOT / "reports/runs/coretail_moe/p2_heldout_input_ids.safetensors"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
VARIANTS = ("bf16_teacher", "q4_experts_bf16_trunk", "bf16_experts_int8_trunk", "q4_experts_int8_trunk", "q4_experts_int4_trunk")
PRIMARY = "q4_experts_int8_trunk"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-11) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    checks = []
    add = lambda name, passed, detail="": checks.append({"name": name, "pass": bool(passed), "detail": detail})
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    evaluator_lock = json.loads(EVALUATOR_LOCK_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    input_path = ROOT / lock["artifact"]
    current = load_file(input_path)
    old = load_file(OLD_INPUT)

    add("preregistration hash", sha256(PREREG) == lock["preregistration_sha256"] == evaluator_lock["preregistration_sha256"])
    add("input lock hash", sha256(LOCK_PATH) == evaluator_lock["input_lock_sha256"])
    add("evaluator hash", sha256(EVALUATOR) == evaluator_lock["evaluator_sha256"])
    add("input artifact hash", sha256(input_path) == lock["artifact_sha256"])
    add("result provenance", result["inputs"]["evaluator_sha256"] == sha256(EVALUATOR) and result["inputs"]["evaluator_lock_sha256"] == sha256(EVALUATOR_LOCK_PATH))
    add("kind and split", result.get("kind") == "streamq4_moe_p0_full_depth_model_quality" and result.get("split") == "validation")
    add("variant set", set(result.get("variants", {})) == set(VARIANTS))
    add("domain contract", tuple(result.get("data", {}).get("domains", ())) == DOMAINS and result["data"]["labels"] == 1270)
    add("input tensor set", set(current) == {f"{split}_{domain}" for split in ("validation", "test") for domain in DOMAINS})
    add("input tensor shapes", all(tuple(value.shape) == (2, 128) for value in current.values()))
    add("input tensor hashes", all(tensor_sha(value) == lock["input_ids_sha256"][name] for name, value in current.items()))
    exact_disjoint = True
    for domain in DOMAINS:
        new_rows = list(current[f"validation_{domain}"]) + list(current[f"test_{domain}"])
        old_rows = list(old[f"validation_{domain}"]) + list(old[f"test_{domain}"])
        exact_disjoint &= not any((left == right).all().item() for index, left in enumerate(new_rows) for right in new_rows[index + 1 :])
        exact_disjoint &= not any((left == right).all().item() for left in new_rows for right in old_rows)
    add("fresh exact-context disjoint", exact_disjoint and lock["exact_context_disjoint_from_coretail_p2"] is True)

    source_rows = {}
    for value in lock["source_manifest"].values():
        if "path" in value and "sha256" in value:
            source_rows[value["path"]] = value["sha256"]
        for nested in value.values():
            if isinstance(nested, dict) and "path" in nested and "sha256" in nested:
                source_rows[nested["path"]] = nested["sha256"]
    add("source hashes", all((ROOT / path).is_file() and sha256(ROOT / path) == expected for path, expected in source_rows.items()), f"{len(source_rows)} files")

    layers = result.get("layer_reports", [])
    add("48 ordered layers", [row.get("layer") for row in layers] == list(range(48)))
    layer_ok = len(layers) == 48
    for row in layers:
        layer_ok &= set(row["errors_vs_teacher"]) == set(VARIANTS[1:]) and math.isfinite(row["seconds"]) and row["seconds"] > 0
        for error in row["errors_vs_teacher"].values():
            layer_ok &= error["finite"] is True and math.isfinite(error["relative_l2"]) and math.isfinite(error["max_abs"])
    add("finite layer diagnostics", layer_ok)

    variants = result["variants"]
    teacher = variants["bf16_teacher"]
    teacher_ce = teacher["next_token_cross_entropy"]
    aggregate_ok = domain_ok = top1_ok = final_ok = True
    for name in VARIANTS:
        metrics = variants[name]
        ce = metrics["next_token_cross_entropy"]
        aggregate_ok &= metrics["labels"] == 1270 and math.isfinite(ce) and close(metrics["relative_cross_entropy_increase"], (ce - teacher_ce) / teacher_ce)
        ce_sum = top1_sum = 0.0
        for domain in DOMAINS:
            row = metrics["domains"][domain]
            trow = teacher["domains"][domain]
            domain_ok &= row["labels"] == 254 and close(row["relative_cross_entropy_increase"], (row["next_token_cross_entropy"] - trow["next_token_cross_entropy"]) / trow["next_token_cross_entropy"])
            ce_sum += row["next_token_cross_entropy"] * 254
            if name != "bf16_teacher":
                top1_sum += row["top1_agreement_vs_teacher"] * 254
        domain_ok &= close(ce, ce_sum / 1270)
        if name == "bf16_teacher":
            top1_ok &= metrics["top1_agreement_vs_teacher"] is None
        else:
            top1_ok &= close(metrics["top1_agreement_vs_teacher"], top1_sum / 1270, 1e-7)
            final = metrics["final_hidden_error_vs_teacher"]
            last = layers[-1]["errors_vs_teacher"][name]
            final_ok &= final["finite"] is True and close(final["relative_l2"], last["relative_l2"]) and close(final["max_abs"], last["max_abs"])
    add("aggregate CE arithmetic", aggregate_ok)
    add("domain CE arithmetic", domain_ok)
    add("top1 arithmetic", top1_ok)
    add("final hidden metrics", final_ok)

    primary = variants[PRIMARY]["relative_cross_entropy_increase"]
    add("primary fixed", lock["primary_gate"]["variant"] == PRIMARY)
    add("primary metric", close(result["primary_relative_ce"], primary))
    add("progression decision", result["status"] == ("p0_validation_pass_test_authorized" if primary <= 0.03 else "p0_validation_closed"))
    add("closed flags", result["p0_pass"] is False and result["next_phases_authorized"] is False and result["repair_authorized"] is False)
    add("test remains unopened", not (REPORT / "p0_test_model_quality.json").exists() and not (REPORT / "P0_TEST_MODEL_QUALITY.md").exists())
    add("runtime telemetry", all(result["runtime"][name] > 0 for name in ("seconds", "peak_cuda_allocated_bytes", "peak_rss_bytes")))
    add("controls", all(result["controls"].values()))

    passed = sum(row["pass"] for row in checks)
    status = "p0_validation_verification_pass" if passed == len(checks) else "p0_validation_verification_fail"
    payload = {
        "kind": "streamq4_moe_p0_validation_independent_verification", "status": status,
        "checks_passed": passed, "checks_total": len(checks), "checks": checks,
        "primary_validation_relative_ce": primary,
        "distance_above_progression_gate": primary - 0.03,
        "result_sha256": sha256(RESULT_PATH),
        "conclusion": "The locked RTN-Q4/INT8 validation candidate is closed without opening test.",
    }
    output = REPORT / "p0_validation_verification.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT / "P0_VALIDATION_VERIFICATION.md").write_text(
        f"# STREAMQ4-MoE P0 - onafhankelijke validationaudit\n\n"
        f"Uitkomst: **{status}** ({passed}/{len(checks)} controles).\n\n"
        f"De primaire CE-toename is {primary:.4%}, {primary - 0.03:.4%} boven de gelockte 3%-progression gate. Test blijft ongeopend.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "checks": f"{passed}/{len(checks)}", "primary": primary}, indent=2))
    if status.endswith("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
