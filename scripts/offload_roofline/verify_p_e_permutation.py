from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from moe_lab.moe_layer import load_moe_layer
from moe_lab.reporting import ROOT

sys.path.insert(0, str(ROOT / "scripts/craft_moe"))
from evaluate_atomic_oracle import (  # noqa: E402
    ACTIVE_EXPERTS,
    ATOMS_PER_EXPERT,
    COMPONENT_RELATIVE,
    HIDDEN_SIZE,
    LAYER,
    exact_activations_and_outputs,
    regression_summary,
)


RESULT = ROOT / "reports/offload_roofline/p_e_permutation_result.json"
PREREG = ROOT / "reports/offload_roofline/P_E_PERMUTATION_PREREGISTRATION.md"
OLD_RESULT = ROOT / "reports/craft_moe/atomic_oracle.json"
COMPONENT = ROOT / COMPONENT_RELATIVE
OUT_JSON = ROOT / "reports/offload_roofline/p_e_permutation_verification.json"
OUT_MD = ROOT / "reports/offload_roofline/P_E_PERMUTATION_VERIFICATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def independent_permuted_routed(moe, activations, expert_ids, router_weights, permutations):
    selected = torch.empty(activations.shape[0], ACTIVE_EXPERTS, HIDDEN_SIZE, dtype=activations.dtype)
    device = moe.experts[0].down.device
    for expert_id, expert in enumerate(moe.experts):
        positions = torch.nonzero(expert_ids == expert_id, as_tuple=False)
        if positions.shape[0] == 0:
            continue
        token_indices, slot_indices = positions[:, 0], positions[:, 1]
        permutation = permutations[expert_id].long()
        output = F.linear(
            activations[token_indices, slot_indices][:, permutation].to(device),
            expert.down.index_select(1, permutation.to(device)),
        )
        selected[token_indices, slot_indices] = output.cpu()
    return (selected.float() * router_weights.float().unsqueeze(-1)).sum(1).to(activations.dtype)


def mean_non_null(values):
    selected = [value for value in values if value is not None]
    return float(np.mean(np.asarray(selected, dtype=np.float64)))


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite P-E verification")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    old = json.loads(OLD_RESULT.read_text(encoding="utf-8"))
    permutation_path = ROOT / result["calibration"]["permutation_file"]
    permutations = load_file(permutation_path)["layer26_expert_permutations"]
    checks = {}
    checks["preregistration_hash"] = sha256(PREREG) == result["inputs"]["preregistration_sha256"]
    checks["component_hash"] = sha256(COMPONENT) == result["inputs"]["component_sha256"]
    checks["old_result_hash"] = sha256(OLD_RESULT) == result["inputs"]["old_atomic_result_sha256"]
    checks["permutation_hash"] = sha256(permutation_path) == result["calibration"]["permutation_sha256"]
    checks["permutation_shape_dtype"] = list(permutations.shape) == [64, 1408] and permutations.dtype == torch.int16
    expected = torch.arange(ATOMS_PER_EXPERT)
    checks["all_permutations_bijective"] = all(torch.equal(torch.sort(row.long()).values, expected) for row in permutations)
    observations = result["calibration"]["expert_observations"]
    checks["calibration_observation_accounting"] = len(observations) == 64 and sum(observations.values()) == 768 * 6 and min(observations.values()) > 0
    checks["split_isolation"] = result["windows"] == {"calibration_validation_indices": [256, 1024], "evaluation_validation_indices": [0, 256], "evaluation_test_indices": [0, 256]}
    checks["route_control"] = result["route_control"]["ids_exact"] and result["route_control"]["weights_max_abs"] == 0.0

    baseline_masks_exact = baseline_metrics_exact = raw_metric_aggregates = retained_exact = True
    for split in ("validation", "test"):
        mapping = {
            "global_neuron_25": "global_contribution__f0p250",
            "original_tile64_25": "tile64_contribution__f0p250",
        }
        for new_name, old_name in mapping.items():
            new_row = result["results"][split][new_name]
            old_row = old["results"][split]["policies"][old_name]
            baseline_masks_exact &= new_row["mask_sha256_packed"] == old_row["support"]["sha256_packed"]
            baseline_metrics_exact &= new_row["metrics"]["aggregate"] == old_row["full_model"]["aggregate"]
        for row in result["results"][split].values():
            aggregate, raw = row["metrics"]["aggregate"], row["metrics"]["raw"]
            raw_metric_aggregates &= abs(mean_non_null(raw["candidate_true_token_nll"]) - aggregate["candidate_cross_entropy"]) < 1e-12
            raw_metric_aggregates &= abs(float(np.mean(raw["teacher_to_candidate_kl"])) - aggregate["teacher_to_candidate_kl"]) < 1e-12
            raw_metric_aggregates &= abs(float(np.mean(raw["top1_agreement"])) - aggregate["top1_agreement"]) < 1e-12
            support = row["retained_atoms_per_token"]
            retained_exact &= support["minimum"] == 2112 and support["maximum"] == 2112 and support["mean"] == 2112
    checks["historical_baseline_masks_exact"] = baseline_masks_exact
    checks["historical_baseline_metrics_exact"] = baseline_metrics_exact
    checks["raw_metric_aggregates"] = raw_metric_aggregates
    checks["equal_25pct_atom_budget"] = retained_exact

    quality_arithmetic = True
    for split in ("validation", "test"):
        gate = result["quality_gates"][split]
        neuron = result["results"][split]["global_neuron_25"]["metrics"]["aggregate"]["teacher_to_candidate_kl"]
        spectral = result["results"][split]["spectral_tile64_25"]["metrics"]["aggregate"]["teacher_to_candidate_kl"]
        quality_arithmetic &= gate["neuron_kl"] == neuron and gate["spectral_tile64_kl"] == spectral
        quality_arithmetic &= abs(gate["kl_ratio"] - spectral / neuron) < 1e-12
        quality_arithmetic &= gate["pass_le_1_20x"] == (spectral <= 1.2 * neuron)
    checks["quality_gate_arithmetic"] = quality_arithmetic
    checks["quality_gate_fails_both"] = not any(row["pass_le_1_20x"] for row in result["quality_gates"].values())

    component = load_file(COMPONENT, device="cpu")
    index = torch.tensor(list(range(0, 256)) + list(range(1024, 1280)), dtype=torch.long)
    inputs = component["moe_input"].index_select(0, index)
    ids = component["router_ids"].index_select(0, index).long()
    weights = component["router_weights"].index_select(0, index).float()
    moe = load_moe_layer(ROOT / "models/deepseek-v2-lite", LAYER, torch.device("cuda"))
    activations, selected_outputs, _norms = exact_activations_and_outputs(moe, inputs, ids)
    direct = (selected_outputs.float() * weights.unsqueeze(-1)).sum(1).to(activations.dtype)
    permuted = independent_permuted_routed(moe, activations, ids, weights, permutations)
    regression = regression_summary(direct, permuted)
    checks["independent_permutation_not_bit_exact"] = not torch.equal(direct, permuted) and not result["permutation_full_reconstruction"]["bit_exact"]
    checks["independent_permutation_regression"] = abs(regression["nrmse"] - result["permutation_full_reconstruction"]["regression"]["nrmse"]) < 1e-12 and regression["maximum_absolute_error"] == result["permutation_full_reconstruction"]["regression"]["maximum_absolute_error"]
    checks["official_gates_and_verdict"] = result["gates"] == {"all_permutations_bijective": True, "old_tile64_reproduced": True, "spectral_tile64_kl_le_1_20x_neuron_both_splits": False, "permutation_full_reconstruction_bit_exact": False, "all_pass": False} and result["verdict"] == "p_e_negative"

    checks = {name: bool(value) for name, value in checks.items()}
    passed = sum(checks.values())
    verification = {
        "kind": "offload_roofline_p_e_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "checks_passed": passed, "checks_total": len(checks),
        "all_pass": passed == len(checks),
        "verdict": "p_e_negative_verified" if passed == len(checks) else "verification_failed",
        "independent_permutation_regression": regression,
    }
    OUT_JSON.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# P-E onafhankelijke verificatie", "",
        f"**{verification['verdict']}** — {passed}/{len(checks)} controles geslaagd.", "",
        "De historische maskers en metrics zijn exact gereproduceerd, alle permutaties zijn bijectief, de ruwe evaluatiemetrics en gates zijn herberekend en een tweede BF16-reconstructie bevestigde dat de permutatie niet bit-identiek is.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": verification["verdict"], "checks": f"{passed}/{len(checks)}", "independent_regression": regression}, indent=2))
