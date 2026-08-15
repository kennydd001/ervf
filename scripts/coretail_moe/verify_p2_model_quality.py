from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from safetensors.torch import load_file

from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
VARIANTS = (
    "bf16_teacher",
    "gptq_experts_bf16_trunk",
    "bf16_experts_int4_trunk",
    "gptq_experts_int4_trunk",
    "gptq_experts_int8_trunk",
)
STUDENTS = VARIANTS[1:]
PRIMARY = "gptq_experts_int4_trunk"
REPORT_DIR = ROOT / "reports/coretail_moe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "pass": bool(passed), "detail": detail})

    @property
    def passed(self) -> int:
        return sum(row["pass"] for row in self.checks)


def expected_test_status(validation_relative: float, test_relative: float, finite: bool) -> tuple[str, bool, bool]:
    if finite and validation_relative <= 0.02 and test_relative <= 0.02:
        return "p2_pass", True, False
    if finite and 0.02 < test_relative <= 0.10:
        return "p2_repair_authorized", False, True
    if finite and test_relative > 0.10:
        return "p2_quality_closed", False, False
    return "p2_validation_gate_fail", False, False


def audit_result(audit: Audit, split: str, result: dict, lock: dict, validation: dict) -> None:
    prefix = f"{split}:"
    audit.check(prefix + " kind", result.get("kind") == "coretail_moe_p2_full_depth_model_quality")
    audit.check(prefix + " split", result.get("split") == split)
    audit.check(prefix + " variants", set(result.get("variants", {})) == set(VARIANTS))
    audit.check(prefix + " domains", tuple(result.get("data", {}).get("domains", ())) == DOMAINS)
    audit.check(prefix + " labels", result.get("data", {}).get("labels") == 1270)
    audit.check(prefix + " contexts", result.get("data", {}).get("contexts_per_domain") == 2)
    audit.check(prefix + " context tokens", result.get("data", {}).get("context_tokens") == 128)

    layers = result.get("layer_reports", [])
    audit.check(prefix + " 48 ordered layers", [row.get("layer") for row in layers] == list(range(48)))
    layer_values_ok = len(layers) == 48
    for row in layers:
        errors = row.get("errors_vs_teacher", {})
        layer_values_ok &= set(errors) == set(STUDENTS)
        layer_values_ok &= math.isfinite(row.get("seconds", math.nan)) and row.get("seconds", 0) > 0
        for error in errors.values():
            layer_values_ok &= error.get("finite") is True
            layer_values_ok &= math.isfinite(error.get("relative_l2", math.nan))
            layer_values_ok &= math.isfinite(error.get("max_abs", math.nan))
            layer_values_ok &= error.get("relative_l2", -1) >= 0 and error.get("max_abs", -1) >= 0
    audit.check(prefix + " finite layer diagnostics", layer_values_ok)

    variants = result.get("variants", {})
    teacher = variants.get("bf16_teacher", {})
    teacher_ce = teacher.get("next_token_cross_entropy", math.nan)
    metrics_ok = math.isfinite(teacher_ce) and teacher_ce > 0
    relative_ok = True
    domain_ok = True
    top1_ok = teacher.get("top1_agreement_vs_teacher") is None
    final_error_ok = True
    for name in VARIANTS:
        metrics = variants.get(name, {})
        ce = metrics.get("next_token_cross_entropy", math.nan)
        rel = metrics.get("relative_cross_entropy_increase", math.nan)
        labels = metrics.get("labels")
        metrics_ok &= math.isfinite(ce) and ce > 0 and labels == 1270
        relative_ok &= close(rel, (ce - teacher_ce) / teacher_ce)
        domains = metrics.get("domains", {})
        domain_ok &= set(domains) == set(DOMAINS)
        weighted_ce = 0.0
        weighted_top1 = 0.0
        for domain in DOMAINS:
            row = domains.get(domain, {})
            domain_teacher = teacher.get("domains", {}).get(domain, {})
            d_ce = row.get("next_token_cross_entropy", math.nan)
            d_teacher_ce = domain_teacher.get("next_token_cross_entropy", math.nan)
            d_rel = row.get("relative_cross_entropy_increase", math.nan)
            d_labels = row.get("labels")
            domain_ok &= d_labels == 254 and math.isfinite(d_ce) and d_ce > 0
            domain_ok &= close(d_rel, (d_ce - d_teacher_ce) / d_teacher_ce)
            weighted_ce += d_ce * d_labels
            agreement = row.get("top1_agreement_vs_teacher")
            if name == "bf16_teacher":
                top1_ok &= agreement is None
            else:
                top1_ok &= isinstance(agreement, (int, float)) and 0 <= agreement <= 1
                weighted_top1 += agreement * d_labels
        domain_ok &= close(ce, weighted_ce / 1270)
        agreement = metrics.get("top1_agreement_vs_teacher")
        if name != "bf16_teacher":
            top1_ok &= isinstance(agreement, (int, float)) and 0 <= agreement <= 1
            top1_ok &= close(agreement, weighted_top1 / 1270, tolerance=1e-7)
            final_error = metrics.get("final_hidden_error_vs_teacher", {})
            last_layer = layers[-1].get("errors_vs_teacher", {}).get(name, {}) if layers else {}
            final_error_ok &= final_error.get("finite") is True
            final_error_ok &= close(final_error.get("relative_l2", math.nan), last_layer.get("relative_l2", math.nan))
            final_error_ok &= close(final_error.get("max_abs", math.nan), last_layer.get("max_abs", math.nan))
    audit.check(prefix + " aggregate metrics", metrics_ok)
    audit.check(prefix + " recomputed relative CE", relative_ok)
    audit.check(prefix + " domain arithmetic", domain_ok)
    audit.check(prefix + " top-1 arithmetic", top1_ok)
    audit.check(prefix + " final hidden diagnostics", final_error_ok)

    primary_relative = variants.get(PRIMARY, {}).get("relative_cross_entropy_increase", math.nan)
    audit.check(prefix + " primary fixed", lock.get("primary_gate", {}).get("variant") == PRIMARY)
    audit.check(prefix + " primary relative CE", close(result.get("primary_relative_ce", math.nan), primary_relative))
    controls = result.get("controls", {})
    audit.check(prefix + " controls", all(controls.get(key) is True for key in ("all_finite", "all_48_layers", "test_opened_after_validation", "primary_variant_fixed")))
    runtime = result.get("runtime", {})
    audit.check(prefix + " runtime telemetry", all(isinstance(runtime.get(key), (int, float)) and runtime.get(key) > 0 for key in ("seconds", "peak_cuda_allocated_bytes", "peak_rss_bytes")))

    if split == "validation":
        audit.check(prefix + " validation status", result.get("status") == "validation_complete_test_authorized")
        audit.check(prefix + " no premature pass", result.get("p2_pass") is False and result.get("wall_clock_authorized") is False)
    else:
        validation_relative = validation["variants"][PRIMARY]["relative_cross_entropy_increase"]
        expected_status, expected_pass, expected_repair = expected_test_status(validation_relative, primary_relative, controls.get("all_finite") is True)
        audit.check(prefix + " preregistered decision", result.get("status") == expected_status, expected_status)
        audit.check(prefix + " p2 pass flag", result.get("p2_pass") is expected_pass)
        audit.check(prefix + " repair flag", result.get("p3_repair_authorized") is expected_repair)
        audit.check(prefix + " wall-clock flag", result.get("wall_clock_authorized") is expected_pass)
        validation_completed = datetime.fromisoformat(validation["completed_utc"])
        test_started = datetime.fromisoformat(result["started_utc"])
        audit.check(prefix + " test after validation", test_started > validation_completed)


def main() -> None:
    audit = Audit()
    prereg = REPORT_DIR / "P2_MODEL_QUALITY_PREREGISTRATION.md"
    lock_path = REPORT_DIR / "p2_input_lock.json"
    input_path = ROOT / "reports/runs/coretail_moe/p2_heldout_input_ids.safetensors"
    p1_path = REPORT_DIR / "p1_full_benchmark_verification.json"
    bank_path = ROOT / "reports/qwen_gptq_bank/p0_full_bank_verification.json"
    model_index = ROOT / "models/qwen3-30b-a3b-base/model.safetensors.index.json"
    evaluator = ROOT / "scripts/coretail_moe/run_p2_model_quality.py"
    validation_path = REPORT_DIR / "p2_validation_model_quality.json"
    test_path = REPORT_DIR / "p2_test_model_quality.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))

    audit.check("preregistration hash", sha256(prereg) == lock["preregistration_sha256"])
    audit.check("held-out artifact hash", sha256(input_path) == lock["artifact_sha256"])
    audit.check("P1 verification hash", sha256(p1_path) == lock["p1_verification_sha256"])
    audit.check("model index hash", sha256(model_index) == lock["model_index_sha256"])
    audit.check("bank verification pass", json.loads(bank_path.read_text(encoding="utf-8")).get("status") == "full_bank_pass")

    tensors = load_file(input_path)
    expected_names = {f"{split}_{domain}" for split in ("validation", "test") for domain in DOMAINS}
    audit.check("input tensor names", set(tensors) == expected_names)
    shape_dtype_ok = all(tuple(tensors[name].shape) == (2, 128) and str(tensors[name].dtype) == "torch.int64" for name in expected_names)
    audit.check("input tensor shapes and dtype", shape_dtype_ok)
    input_hashes_ok = all(tensor_sha256(tensors[name]) == lock["input_ids_sha256"][name] for name in expected_names)
    audit.check("input tensor hashes", input_hashes_ok)
    split_disjoint = all(not any((left == right).all().item() for left in tensors[f"validation_{domain}"] for right in tensors[f"test_{domain}"]) for domain in DOMAINS)
    audit.check("validation/test exact-context disjoint", split_disjoint)

    source_rows = {}
    for item in lock["source_manifest"].values():
        if "path" in item and "sha256" in item:
            source_rows[item["path"]] = item["sha256"]
        for nested in item.get("java", {}), item.get("python", {}):
            if isinstance(nested, dict) and "path" in nested:
                source_rows[nested["path"]] = nested["sha256"]
    source_hashes_ok = all((ROOT / path).is_file() and sha256(ROOT / path) == expected for path, expected in source_rows.items())
    audit.check("locked source hashes", source_hashes_ok, f"{len(source_rows)} unique files")

    for split, result in (("validation", validation), ("test", test)):
        expected_inputs = {
            "preregistration_sha256": sha256(prereg),
            "lock_sha256": sha256(lock_path),
            "input_artifact_sha256": sha256(input_path),
            "p1_verification_sha256": sha256(p1_path),
            "bank_verification_sha256": sha256(bank_path),
            "model_index_sha256": sha256(model_index),
        }
        audit.check(f"{split}: result provenance", result.get("inputs") == expected_inputs)
        audit_result(audit, split, result, lock, validation)

    passed = audit.passed
    total = len(audit.checks)
    status = "p2_verification_pass" if passed == total else "p2_verification_fail"
    payload = {
        "kind": "coretail_moe_p2_independent_verification",
        "status": status,
        "checks_passed": passed,
        "checks_total": total,
        "checks": audit.checks,
        "hashes": {
            "evaluator_sha256": sha256(evaluator),
            "validation_result_sha256": sha256(validation_path),
            "test_result_sha256": sha256(test_path),
        },
        "independent_recomputation": [
            "artifact and source hashes",
            "input tensor hashes, shapes, and exact-context split separation",
            "all aggregate/domain relative cross-entropies",
            "weighted aggregate CE and top-1 agreement",
            "48-layer diagnostic completeness and finiteness",
            "preregistered P2/repair/wall-clock decision",
        ],
        "primary_test_relative_ce": test["variants"][PRIMARY]["relative_cross_entropy_increase"],
        "conclusion": "P2 quality is closed; the integrated wall-clock phase is not authorized.",
    }
    output = REPORT_DIR / "p2_model_quality_verification.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = REPORT_DIR / "P2_MODEL_QUALITY_VERIFICATION.md"
    report.write_text(
        "\n".join(
            [
                "# CORETAIL-MoE P2 - onafhankelijke verificatie",
                "",
                f"Uitkomst: **{status}** ({passed}/{total} controles).",
                "",
                f"De primaire testtoename in cross-entropy is **{payload['primary_test_relative_ce']:.3%}**.",
                "De preregistratie sluit daardoor P2 en autoriseert geen repair of geïntegreerde wall-clocktest.",
                "",
                "Deze audit herberekent hashes, inputlocks, metriekrekenkunde, laagvolledigheid en de beslisregel onafhankelijk van de evaluator.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "checks": f"{passed}/{total}", "output": str(output)}, indent=2))
    if status != "p2_verification_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
