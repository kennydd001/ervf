from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from moe_lab.craft_moe.atomic import (
    delta_patched_hidden,
    global_tile_topk_mask,
    global_topk_mask,
    relative_routed_l2,
)
from moe_lab.moe_layer import load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT

sys.path.insert(0, str(ROOT / "scripts/craft_moe"))
from evaluate_atomic_oracle import (  # noqa: E402
    ATOMS_PER_EXPERT,
    CANDIDATE_BATCH,
    COMPONENT_RELATIVE,
    HIDDEN_SIZE,
    LAYER,
    ACTIVE_EXPERTS,
    build_policies,
    corpus_tokens,
    evaluate_hidden,
    exact_activations_and_outputs,
    make_teacher_reference,
    numeric_summary,
    reconstruct_policy_masks,
    regression_summary,
    release_non_down_weights,
    sequence_blocks,
)


PREREG = ROOT / "reports/offload_roofline/P_E_PERMUTATION_PREREGISTRATION.md"
COMPONENT = ROOT / COMPONENT_RELATIVE
OLD_RESULT = ROOT / "reports/craft_moe/atomic_oracle.json"
OUT_JSON = ROOT / "reports/offload_roofline/p_e_permutation_result.json"
OUT_MD = ROOT / "reports/offload_roofline/P_E_PERMUTATION_REPORT.md"
PERM_FILE = ROOT / "reports/runs/offload_roofline/p_e_layer26_permutations.safetensors"
CALIBRATION = slice(256, 1024)
EVAL_VALIDATION = slice(0, 256)
EVAL_TEST = slice(1024, 1280)
FRACTION = 0.25
TILE = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_spectral_groups(features: np.ndarray, indices: np.ndarray, groups: int):
    if groups == 1:
        if indices.size != TILE:
            raise ValueError("terminal spectral group is not tile-sized")
        return [indices]
    left_groups = groups // 2
    left_size = left_groups * TILE
    selected = features[indices].astype(np.float64, copy=False)
    centered = selected - selected.mean(axis=0, keepdims=True)
    if centered.shape[1] == 0 or float(np.square(centered).sum()) == 0.0:
        order = np.sort(indices)
    else:
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ vh[0]
            order = indices[np.lexsort((indices, projection))]
        except np.linalg.LinAlgError:
            order = np.sort(indices)
    return balanced_spectral_groups(features, order[:left_size], left_groups) + balanced_spectral_groups(features, order[left_size:], groups - left_groups)


def learn_permutations(calibration_mask: torch.Tensor, calibration_ids: torch.Tensor):
    permutations = np.empty((64, ATOMS_PER_EXPERT), dtype=np.int16)
    observations = {}
    for expert in range(64):
        positions = (calibration_ids == expert).nonzero(as_tuple=False)
        observations[str(expert)] = int(positions.shape[0])
        if positions.numel():
            matrix = calibration_mask[positions[:, 0], positions[:, 1]].numpy().T
        else:
            matrix = np.empty((ATOMS_PER_EXPERT, 0), dtype=bool)
        groups = balanced_spectral_groups(matrix, np.arange(ATOMS_PER_EXPERT), ATOMS_PER_EXPERT // TILE)
        permutations[expert] = np.concatenate(groups).astype(np.int16)
    return torch.from_numpy(permutations), observations


def permuted_tile_mask(contribution: torch.Tensor, expert_ids: torch.Tensor, permutations: torch.Tensor):
    selected_permutations = permutations[expert_ids.long()].long()
    reordered = torch.gather(contribution, 2, selected_permutations)
    reordered_mask = global_tile_topk_mask(reordered, FRACTION, TILE)
    original_mask = torch.zeros_like(reordered_mask)
    original_mask.scatter_(2, selected_permutations, reordered_mask)
    return original_mask


@torch.inference_mode()
def permuted_full_routed(moe, activations, expert_ids, router_weights, permutations):
    tokens = activations.shape[0]
    selected = torch.empty(tokens, ACTIVE_EXPERTS, HIDDEN_SIZE, dtype=activations.dtype)
    device = moe.experts[0].down.device
    for expert_id, expert in enumerate(moe.experts):
        positions = (expert_ids == expert_id).nonzero(as_tuple=False)
        if not positions.numel():
            continue
        token_indices, slot_indices = positions[:, 0], positions[:, 1]
        permutation = permutations[expert_id].long()
        reordered_activation = activations[token_indices, slot_indices][:, permutation].to(device)
        reordered_down = expert.down[:, permutation.to(device)]
        output = F.linear(reordered_activation, reordered_down)
        selected[token_indices, slot_indices] = output.cpu()
    return (selected.float() * router_weights.float().unsqueeze(-1)).sum(dim=1).to(activations.dtype)


def mask_hash(mask: torch.Tensor):
    packed = np.packbits(mask.numpy().reshape(mask.shape[0], -1), axis=1, bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


if __name__ == "__main__":
    if any(path.exists() for path in (OUT_JSON, OUT_MD, PERM_FILE)):
        raise FileExistsError("refusing to overwrite P-E outputs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P-E")
    torch.set_grad_enabled(False)
    torch.manual_seed(20260811)
    np.random.seed(20260811)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model_dir = ROOT / "models/deepseek-v2-lite"
    old = json.loads(OLD_RESULT.read_text(encoding="utf-8"))
    all_components = load_file(COMPONENT, device="cpu")
    indices = torch.tensor(list(range(256, 1024)) + list(range(0, 256)) + list(range(1024, 1280)), dtype=torch.long)
    components = {key: all_components[key].index_select(0, indices) for key in ("moe_input", "router_ids", "router_weights", "teacher")}
    del all_components
    calibration_tokens = 768
    eval_slice = slice(calibration_tokens, calibration_tokens + 512)
    eval_ids = components["router_ids"][eval_slice].long()
    eval_weights = components["router_weights"][eval_slice].float()

    moe = load_moe_layer(model_dir, LAYER, torch.device("cuda"))
    routed_ids, routed_weights = moe.route(components["moe_input"].to("cuda"))
    route_control = {
        "ids_exact": bool(torch.equal(routed_ids.cpu(), components["router_ids"].long())),
        "weights_max_abs": float((routed_weights.cpu().float() - components["router_weights"].float()).abs().max().item()),
    }
    calibration_activations, _calibration_outputs, calibration_down_norm_bank = exact_activations_and_outputs(
        moe,
        components["moe_input"][:calibration_tokens],
        components["router_ids"][:calibration_tokens].long(),
    )
    eval_activations, selected_outputs, eval_down_norm_bank = exact_activations_and_outputs(
        moe,
        components["moe_input"][eval_slice],
        eval_ids,
    )
    calibration_norms = calibration_down_norm_bank[components["router_ids"][:calibration_tokens].long()]
    calibration_contribution = calibration_activations.float().abs() * calibration_norms.float() * components["router_weights"][:calibration_tokens].float().abs().unsqueeze(-1)
    calibration_mask = global_topk_mask(calibration_contribution, FRACTION)
    permutations, observations = learn_permutations(calibration_mask, components["router_ids"][:calibration_tokens].long())
    bijective = all(torch.equal(torch.sort(row.long()).values, torch.arange(ATOMS_PER_EXPERT)) for row in permutations)
    PERM_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file({"layer26_expert_permutations": permutations.contiguous()}, PERM_FILE)

    eval_norms = eval_down_norm_bank[eval_ids]
    eval_contribution = eval_activations.float().abs() * eval_norms.float() * eval_weights.abs().unsqueeze(-1)
    baseline_specs, baseline_masks = build_policies(eval_activations, eval_weights, eval_norms)
    neuron_index = next(index for index, spec in enumerate(baseline_specs) if spec["method"] == "global_contribution" and spec["requested_fraction"] == FRACTION)
    tile64_index = next(index for index, spec in enumerate(baseline_specs) if spec["method"] == "tile64_contribution" and spec["requested_fraction"] == FRACTION)
    neuron_mask = baseline_masks[neuron_index]
    original_tile_mask = baseline_masks[tile64_index]
    spectral_tile_mask = permuted_tile_mask(eval_contribution, eval_ids, permutations)
    direct_routed = (selected_outputs.float() * eval_weights.unsqueeze(-1)).sum(dim=1).to(eval_activations.dtype)
    permuted_control = permuted_full_routed(moe, eval_activations, eval_ids, eval_weights, permutations)
    permutation_regression = regression_summary(direct_routed, permuted_control)
    permutation_bit_exact = torch.equal(direct_routed, permuted_control)

    release_non_down_weights(moe)
    exact_routed = reconstruct_policy_masks(moe, eval_activations, eval_ids, eval_weights, [baseline_masks[0]], 1)[0]
    baseline_candidate_routed = reconstruct_policy_masks(moe, eval_activations, eval_ids, eval_weights, baseline_masks[1:], 4)
    spectral_routed = reconstruct_policy_masks(moe, eval_activations, eval_ids, eval_weights, [spectral_tile_mask], 1)[0]
    routed = torch.stack((exact_routed, baseline_candidate_routed[neuron_index - 1], baseline_candidate_routed[tile64_index - 1], spectral_routed))
    exact_decomposition = regression_summary(direct_routed, exact_routed)
    del moe, calibration_activations, _calibration_outputs, calibration_down_norm_bank, calibration_norms, calibration_contribution
    del eval_activations, selected_outputs, eval_down_norm_bank, eval_norms, eval_contribution
    gc.collect()
    torch.cuda.empty_cache()

    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to("cuda")
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to("cuda")
    split_slices = {"validation": slice(0, 256), "test": slice(256, 512)}
    token_ids = {split: corpus_tokens(model_dir, split, 256) for split in split_slices}
    results = {}
    policies = (("global_neuron_25", neuron_mask), ("original_tile64_25", original_tile_mask), ("spectral_tile64_25", spectral_tile_mask))
    for split_index, (split, selected) in enumerate(split_slices.items()):
        reference = make_teacher_reference(components["teacher"][eval_slice][selected], token_ids[split], sequence_blocks(256), norm_weight, lm_head, CANDIDATE_BATCH)
        split_rows = {}
        for policy_index, (name, mask) in enumerate(policies, start=1):
            patched = delta_patched_hidden(components["teacher"][eval_slice][selected], exact_routed[selected], routed[policy_index, selected])
            metrics = evaluate_hidden(patched, reference, norm_weight, lm_head, CANDIDATE_BATCH, 10_000, 20260811 + split_index)
            split_rows[name] = {
                "metrics": metrics,
                "routed_relative_l2": numeric_summary(relative_routed_l2(exact_routed[selected], routed[policy_index, selected])),
                "retained_atoms_per_token": numeric_summary(mask[selected].sum(dim=(1, 2)).double()),
                "mask_sha256_packed": mask_hash(mask[selected]),
            }
        results[split] = split_rows

    baseline_reproduction = {}
    quality_gates = {}
    for split in ("validation", "test"):
        old_kl = old["gates"]["tile64_by_split"][split]["tile64_mean_kl"]
        new_old_kl = results[split]["original_tile64_25"]["metrics"]["aggregate"]["teacher_to_candidate_kl"]
        neuron_kl = results[split]["global_neuron_25"]["metrics"]["aggregate"]["teacher_to_candidate_kl"]
        spectral_kl = results[split]["spectral_tile64_25"]["metrics"]["aggregate"]["teacher_to_candidate_kl"]
        baseline_reproduction[split] = {"old_kl": old_kl, "reproduced_kl": new_old_kl, "absolute_difference": abs(new_old_kl - old_kl), "pass_le_5e_6": abs(new_old_kl - old_kl) <= 5e-6}
        quality_gates[split] = {"neuron_kl": neuron_kl, "spectral_tile64_kl": spectral_kl, "kl_ratio": spectral_kl / neuron_kl, "pass_le_1_20x": spectral_kl <= 1.2 * neuron_kl}
    baseline_pass = all(row["pass_le_5e_6"] for row in baseline_reproduction.values())
    quality_pass = all(row["pass_le_1_20x"] for row in quality_gates.values())
    official_pass = bijective and baseline_pass and quality_pass and permutation_bit_exact
    payload = {
        "kind": "offload_roofline_p_e_layer26_spectral_permutation",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "verdict": "p_e_pass" if official_pass else "p_e_negative",
        "evidence_class": "exploratory_existing_trace_out_of_sample_calibration",
        "inputs": {"preregistration_sha256": sha256(PREREG), "component_sha256": sha256(COMPONENT), "old_atomic_result_sha256": sha256(OLD_RESULT), "model_config_sha256": sha256(model_dir / "config.json"), "model_index_sha256": sha256(model_dir / "model.safetensors.index.json")},
        "windows": {"calibration_validation_indices": [256, 1024], "evaluation_validation_indices": [0, 256], "evaluation_test_indices": [0, 256]},
        "route_control": route_control,
        "calibration": {"expert_observations": observations, "all_expert_permutations_bijective": bijective, "permutation_file": str(PERM_FILE.relative_to(ROOT)).replace("\\", "/"), "permutation_sha256": sha256(PERM_FILE)},
        "permutation_full_reconstruction": {"bit_exact": permutation_bit_exact, "regression": permutation_regression},
        "original_full_decomposition": exact_decomposition,
        "results": results,
        "old_tile64_reproduction": baseline_reproduction,
        "quality_gates": quality_gates,
        "gates": {"all_permutations_bijective": bijective, "old_tile64_reproduced": baseline_pass, "spectral_tile64_kl_le_1_20x_neuron_both_splits": quality_pass, "permutation_full_reconstruction_bit_exact": permutation_bit_exact, "all_pass": official_pass},
        "hardware": {"device": torch.cuda.get_device_name(0), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
        "claim_boundary": "Layer-26 exact-activation oracle with held-out calibration; dense masked evaluation, no packed tile kernel or full-depth runtime claim.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# P-E spectrale neuronpermutatie — resultaat", "", f"**Uitkomst: {payload['verdict']}**.", ""]
    for split in ("validation", "test"):
        gate = quality_gates[split]
        old_ratio = old["gates"]["tile64_by_split"][split]["kl_ratio"]
        lines.append(f"- {split}: neuron-KL {gate['neuron_kl']:.6f}; spectrale tile-KL {gate['spectral_tile64_kl']:.6f}; ratio {gate['kl_ratio']:.3f}× (oude tile {old_ratio:.3f}×).")
    lines.extend(["", f"Oude baseline gereproduceerd: {baseline_pass}; permutaties bijectief: {bijective}; volledige reconstructie bit-exact: {permutation_bit_exact}.", f"Numerieke permutatiecontrole: NRMSE {permutation_regression['nrmse']:.3e}, max abs {permutation_regression['maximum_absolute_error']:.6g}.", "", "De run is een laag-26-oracle en geen packed runtime of full-depth bewijs.", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "quality_gates": quality_gates, "gates": payload["gates"], "permutation_regression": permutation_regression, "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
