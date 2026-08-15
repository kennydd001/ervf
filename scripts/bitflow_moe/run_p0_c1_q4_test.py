from __future__ import annotations

import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.partial_forward import load_decoder_layer
from moe_lab.reporting import ROOT

from fit_validate_p0_c1_q4 import (
    HIDDEN,
    INPUTS,
    MODEL,
    PREREG,
    REPAIR_DIR,
    apply_repair,
    final_logits,
    final_metrics,
    forward_capture,
    layer_zero,
    quantize_q4,
    sha256,
)


VALIDATION = ROOT / "reports/bitflow_moe/p0_c1_q4_validation.json"
RESULT = ROOT / "reports/bitflow_moe/p0_c1_q4_test.json"
REPORT = ROOT / "reports/bitflow_moe/P0_C1_Q4_TEST.md"
LOGITS = ROOT / "reports/runs/bitflow_moe/p0_c1_q4_test_logits.safetensors"


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if RESULT.exists() or REPORT.exists() or LOGITS.exists():
        raise FileExistsError("refusing to overwrite BITFLOW test")
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    inputs = load_file(INPUTS)
    ids = inputs["test"].long()
    device = torch.device("cuda")
    teacher = layer_zero(MODEL, ids, device)
    baseline = teacher.clone()
    repaired = teacher.clone()
    layers = []
    controls = True
    for layer_index in range(1, 27):
        layer, _ = load_decoder_layer(MODEL, layer_index, device)
        teacher, _, teacher_ids, _ = forward_capture(layer, teacher)
        quantize_q4(layer)
        baseline, _, _, _ = forward_capture(layer, baseline)
        provisional, routed, repair_ids, _ = forward_capture(layer, repaired)
        weights = load_file(REPAIR_DIR / f"layer_{layer_index:02d}.safetensors", device="cuda")
        repaired = apply_repair(provisional, routed, weights["A"], weights["B"])
        control = forward_capture(layer, provisional[:1, :8])[0]
        sequence = 8
        positions = torch.arange(sequence, device=device).unsqueeze(0)
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        mask = _prepare_4d_causal_attention_mask(None, (1, sequence), provisional[:1, :8], 0)
        official = layer(provisional[:1, :8], attention_mask=mask, position_ids=positions, use_cache=False, output_attentions=False)[0]
        max_abs = float((official - control).abs().max().detach())
        controls &= max_abs == 0.0
        layers.append({
            "layer": layer_index,
            "baseline_hidden": regression_metrics(baseline.cpu(), teacher.cpu()),
            "repaired_hidden": regression_metrics(repaired.cpu(), teacher.cpu()),
            "router_overlap_repaired_teacher": topk_overlap(repair_ids.cpu(), teacher_ids.cpu()),
            "official_decomposition_max_abs": max_abs,
            "repair_sha256": sha256(REPAIR_DIR / f"layer_{layer_index:02d}.safetensors"),
        })
        print(json.dumps({"layer": layer_index, "baseline_nrmse": layers[-1]["baseline_hidden"]["nrmse"], "repaired_nrmse": layers[-1]["repaired_hidden"]["nrmse"]}), flush=True)
        del layer, provisional, routed, weights, control, official
        gc.collect()
        torch.cuda.empty_cache()
    teacher_logits = final_logits(teacher)
    baseline_logits = final_logits(baseline)
    repaired_logits = final_logits(repaired)
    teacher_metrics = final_metrics(teacher_logits, teacher_logits, ids)
    baseline_metrics = final_metrics(baseline_logits, teacher_logits, ids)
    repaired_metrics = final_metrics(repaired_logits, teacher_logits, ids)
    damage = baseline_metrics["next_token_cross_entropy"] - teacher_metrics["next_token_cross_entropy"]
    recovery = (baseline_metrics["next_token_cross_entropy"] - repaired_metrics["next_token_cross_entropy"]) / damage if damage > 0 else float("nan")
    mid = torch.tensor([row["repaired_hidden"]["nrmse"] for row in layers if 7 <= row["layer"] <= 20])
    late = max(row["repaired_hidden"]["nrmse"] for row in layers if row["layer"] >= 21)
    explosion = late / float(mid.median())
    validation_recovery = validation["final"]["ce_damage_recovery"]
    progression = validation_recovery >= 0.50 and recovery >= 0.50
    primary = (
        recovery >= 0.70
        and repaired_metrics["relative_cross_entropy_increase"] <= 0.01
        and repaired_metrics["top1_token_agreement"] >= 0.97
        and explosion <= 2.0
        and controls
    )
    save_file({"input_ids": ids.int(), "teacher_logits": teacher_logits.cpu(), "baseline_q4_logits": baseline_logits.cpu(), "repaired_logits": repaired_logits.cpu()}, LOGITS, metadata={"kind": "bitflow_p0_c1_q4_test_logits"})
    verdict = "p0_primary_positive" if primary else ("p0_progression_positive" if progression else "p0_linear_branch_negative")
    payload = {
        "kind": "bitflow_moe_p0_c1_q4_test",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "test_open_count": 1,
        "inputs": {"preregistration_sha256": sha256(PREREG), "validation_result_sha256": sha256(VALIDATION), "repair_manifest": validation["repair_artifacts"]},
        "layers": layers,
        "final": {"teacher": teacher_metrics, "baseline_q4": baseline_metrics, "repaired": repaired_metrics, "validation_ce_damage_recovery": validation_recovery, "test_ce_damage_recovery": recovery, "late_layer_explosion_ratio": explosion},
        "gates": {
            "validation_recovery_ge_0_50": validation_recovery >= 0.50,
            "test_recovery_ge_0_50": recovery >= 0.50,
            "both_progression_gates": progression,
            "primary_recovery_ge_0_70": recovery >= 0.70,
            "relative_ce_increase_le_0_01": repaired_metrics["relative_cross_entropy_increase"] <= 0.01,
            "top1_agreement_ge_0_97": repaired_metrics["top1_token_agreement"] >= 0.97,
            "late_layer_explosion_le_2": explosion <= 2.0,
            "all_official_decomposition_controls_exact": controls,
        },
        "full_slate_authorized": progression,
        "p1_authorized": False,
        "logits_artifact": str(LOGITS.relative_to(ROOT)).replace("\\", "/"),
        "logits_artifact_sha256": sha256(LOGITS),
        "claim_boundary": "One opened test of data-limited dense C1/Q4. No C0/C2/Q3/syndrome/runtime claim.",
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text("\n".join([
        "# BITFLOW-MoE P0 C1/Q4 — eenmalige test", "",
        f"Uitkomst: **{verdict}**.", "",
        f"Teacher CE {teacher_metrics['next_token_cross_entropy']:.6f}; Q4 CE {baseline_metrics['next_token_cross_entropy']:.6f}; repaired CE {repaired_metrics['next_token_cross_entropy']:.6f}.",
        f"Validation/test CE-schadeherstel: {validation_recovery:.3%} / **{recovery:.3%}**.",
        f"Repaired top-1: {repaired_metrics['top1_token_agreement']:.3%}; relatieve CE-toename: {repaired_metrics['relative_cross_entropy_increase']:.3%}; late-layer explosieratio: {explosion:.3f}.", "",
        "Wanneer één 50%-progression gate faalt, sluit de preregistratie de lineaire tak zonder C0-, C2-, Q3- of syndromesweep.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "test_recovery": recovery, "teacher_ce": teacher_metrics["next_token_cross_entropy"], "baseline_ce": baseline_metrics["next_token_cross_entropy"], "repaired_ce": repaired_metrics["next_token_cross_entropy"], "top1": repaired_metrics["top1_token_agreement"], "explosion": explosion}, indent=2))
