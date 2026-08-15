from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/bitflow_moe/P0_C1_Q4_PREREGISTRATION.md"
LOCK = ROOT / "reports/bitflow_moe/p0_input_lock.json"
INPUTS = ROOT / "reports/runs/bitflow_moe/p0_input_ids.safetensors"
VALIDATION = ROOT / "reports/bitflow_moe/p0_c1_q4_validation.json"
TEST = ROOT / "reports/bitflow_moe/p0_c1_q4_test.json"
VAL_LOGITS = ROOT / "reports/runs/bitflow_moe/p0_c1_q4_validation_logits.safetensors"
TEST_LOGITS = ROOT / "reports/runs/bitflow_moe/p0_c1_q4_test_logits.safetensors"
REPAIRS = ROOT / "reports/runs/bitflow_moe/p0_c1_q4_repairs"
OUTPUT = ROOT / "reports/bitflow_moe/p0_c1_q4_verification.json"
REPORT = ROOT / "reports/bitflow_moe/P0_C1_Q4_VERIFICATION.md"
HISTORICAL_Q4 = ROOT / "reports/baseline/streamed_model_uniform4_all_layers_all_layers.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float) -> bool:
    # Saved logits are BF16. CPU and CUDA cross-entropy reductions can differ
    # by a few 1e-5 while consuming identical logits and labels.
    return math.isclose(a, b, rel_tol=0.0, abs_tol=5e-5)


def close_recovery(a: float, b: float) -> bool:
    # A CE difference is divided by the much smaller Q4 damage; use a separate
    # tolerance for this derived ratio while keeping raw metrics stricter.
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-3)


def metrics(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    ids = tensors["input_ids"].long()
    teacher = tensors["teacher_logits"].float()
    labels = ids[:, 1:].reshape(-1)
    teacher_ce = F.cross_entropy(teacher[:, :-1].reshape(-1, teacher.shape[-1]), labels)
    output = {}
    for name, key in (("baseline_q4", "baseline_q4_logits"), ("repaired", "repaired_logits")):
        logits = tensors[key].float()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels)
        tlogp = F.log_softmax(teacher, dim=-1)
        slogp = F.log_softmax(logits, dim=-1)
        kl = (tlogp.exp() * (tlogp - slogp)).sum(-1)
        output[name] = {
            "teacher_ce": float(teacher_ce),
            "ce": float(ce),
            "delta": float(ce - teacher_ce),
            "relative": float((ce - teacher_ce) / teacher_ce),
            "top1": float((logits.argmax(-1) == teacher.argmax(-1)).float().mean()),
            "kl_mean": float(kl.mean()),
            "kl_p95": float(torch.quantile(kl, 0.95)),
        }
    damage = output["baseline_q4"]["ce"] - output["baseline_q4"]["teacher_ce"]
    output["recovery"] = {
        "value": (output["baseline_q4"]["ce"] - output["repaired"]["ce"]) / damage
    }
    return output


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite BITFLOW verification")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    test = json.loads(TEST.read_text(encoding="utf-8"))
    checks = {
        "preregistration_hash": lock["preregistration_sha256"] == sha256(PREREG),
        "input_artifact_hash": lock["artifact_sha256"] == sha256(INPUTS),
        "validation_input_hashes": validation["inputs"]["input_artifact_sha256"] == sha256(INPUTS) and validation["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "test_validation_hash": test["inputs"]["validation_result_sha256"] == sha256(VALIDATION),
        "validation_logits_hash": validation["logits_artifact_sha256"] == sha256(VAL_LOGITS),
        "test_logits_hash": test["logits_artifact_sha256"] == sha256(TEST_LOGITS),
        "test_opened_once": test["test_open_count"] == 1,
    }
    repair_hashes = repair_shapes = repair_dtypes = True
    actual_bytes = 0
    for layer in range(1, 27):
        path = REPAIRS / f"layer_{layer:02d}.safetensors"
        expected = validation["repair_artifacts"][str(layer)]
        repair_hashes &= sha256(path) == expected["sha256"] == test["inputs"]["repair_manifest"][str(layer)]["sha256"]
        tensors = load_file(path)
        repair_shapes &= set(tensors) == {"A", "B"} and all(tuple(tensor.shape) == (2048, 2048) for tensor in tensors.values())
        repair_dtypes &= all(tensor.dtype == torch.bfloat16 for tensor in tensors.values())
        actual_bytes += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    checks["all_26_repair_hashes"] = repair_hashes
    checks["all_52_matrix_shapes"] = repair_shapes
    checks["all_repair_dtypes_bf16"] = repair_dtypes
    checks["repair_parameter_accounting"] = validation["repair_parameter_count"] == 26 * 2 * 2048 * 2048 and validation["repair_bf16_bytes"] == actual_bytes
    val_reproduced = metrics(load_file(VAL_LOGITS))
    test_reproduced = metrics(load_file(TEST_LOGITS))
    def matches(reproduced, recorded):
        return (
            close(reproduced["baseline_q4"]["ce"], recorded["baseline_q4"]["next_token_cross_entropy"])
            and close(reproduced["repaired"]["ce"], recorded["repaired"]["next_token_cross_entropy"])
            and close(reproduced["baseline_q4"]["teacher_ce"], recorded["teacher"]["next_token_cross_entropy"])
            and close(reproduced["repaired"]["top1"], recorded["repaired"]["top1_token_agreement"])
            and close(reproduced["repaired"]["relative"], recorded["repaired"]["relative_cross_entropy_increase"])
        )
    checks["validation_metrics_from_raw_logits"] = matches(val_reproduced, validation["final"])
    checks["test_metrics_from_raw_logits"] = matches(test_reproduced, test["final"])
    checks["validation_recovery_formula"] = close_recovery(val_reproduced["recovery"]["value"], validation["final"]["ce_damage_recovery"])
    checks["test_recovery_formula"] = close_recovery(test_reproduced["recovery"]["value"], test["final"]["test_ce_damage_recovery"])
    historical = json.loads(HISTORICAL_Q4.read_text(encoding="utf-8"))["payload"]["final"]
    checks["historical_q4_baseline_exact"] = close(test_reproduced["baseline_q4"]["ce"], historical["student_next_token_cross_entropy"]) and close(test_reproduced["baseline_q4"]["teacher_ce"], historical["teacher_next_token_cross_entropy"]) and close(test_reproduced["baseline_q4"]["top1"], historical["top1_token_agreement"])
    late = max(row["repaired_hidden"]["nrmse"] for row in test["layers"] if row["layer"] >= 21)
    median = float(torch.tensor([row["repaired_hidden"]["nrmse"] for row in test["layers"] if 7 <= row["layer"] <= 20]).median())
    checks["late_explosion_formula"] = close(late / median, test["final"]["late_layer_explosion_ratio"])
    checks["all_validation_decomposition_controls"] = all(row["official_decomposition_max_abs"] == 0.0 for row in validation["layers"])
    checks["all_test_decomposition_controls"] = all(row["official_decomposition_max_abs"] == 0.0 for row in test["layers"])
    checks["validation_progression_fails"] = val_reproduced["recovery"]["value"] < 0.50 and test["gates"]["validation_recovery_ge_0_50"] is False
    checks["test_progression_fails"] = test_reproduced["recovery"]["value"] < 0.50 and test["gates"]["test_recovery_ge_0_50"] is False
    checks["primary_gates_fail"] = not all((test["gates"]["primary_recovery_ge_0_70"], test["gates"]["relative_ce_increase_le_0_01"], test["gates"]["top1_agreement_ge_0_97"], test["gates"]["late_layer_explosion_le_2"]))
    checks["linear_branch_closed"] = test["verdict"] == "p0_linear_branch_negative" and test["full_slate_authorized"] is False and test["p1_authorized"] is False
    passed = sum(checks.values())
    ok = passed == len(checks)
    payload = {
        "kind": "bitflow_moe_p0_c1_q4_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verification_pass": ok,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "final_verdict": "p0_linear_branch_negative_verified" if ok else "verification_failed",
        "reproduced": {"validation": val_reproduced, "test": test_reproduced},
        "source_hashes": {"validation_result": sha256(VALIDATION), "test_result": sha256(TEST), "validation_logits": sha256(VAL_LOGITS), "test_logits": sha256(TEST_LOGITS)},
        "claim_boundary": "Verified data-limited dense C1/Q4 failure. C0/C2/Q3/syndrome/runtime remain unopened.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text("\n".join([
        "# BITFLOW-MoE P0 C1/Q4 — onafhankelijke verificatie", "",
        f"Uitkomst: **{payload['final_verdict']}**; **{passed}/{len(checks)}** controles slagen.", "",
        f"Validation/test CE-schadeherstel uit ruwe logits: {val_reproduced['recovery']['value']:.3%} / {test_reproduced['recovery']['value']:.3%}.",
        f"Test teacher/Q4/repaired CE: {test_reproduced['baseline_q4']['teacher_ce']:.6f} / {test_reproduced['baseline_q4']['ce']:.6f} / {test_reproduced['repaired']['ce']:.6f}.", "",
        "De maximale lineaire C1-kandidaat faalt beide 50%-gates. C0, C2, Q3, syndrome en P1 blijven volgens preregistratie gesloten.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verification_pass": ok, "checks": f"{passed}/{len(checks)}", "final_verdict": payload["final_verdict"]}, indent=2))
