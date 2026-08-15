from __future__ import annotations

import math
from typing import Any

import torch


def retained_count(size: int, fraction: float) -> int:
    """Return the preregistered conservative ceil-retention count."""

    if size < 1:
        raise ValueError("size must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return min(size, int(math.ceil(fraction * size)))


def _validate_scores(scores: torch.Tensor) -> tuple[int, int, int]:
    if scores.ndim != 3:
        raise ValueError("scores must have shape [tokens, experts, atoms]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores.shape


def per_expert_topk_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep a fixed ceil fraction within every selected expert.

    Stable sorting makes ties follow the original neuron order.
    """

    _, _, atoms = _validate_scores(scores)
    keep = retained_count(atoms, fraction)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if keep == 0:
        return mask
    ranked = scores.argsort(dim=2, descending=True, stable=True)
    mask.scatter_(2, ranked[:, :, :keep], True)
    return mask


def global_topk_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep a fixed ceil fraction across all expert/atom pairs per token."""

    tokens, experts, atoms = _validate_scores(scores)
    flattened = scores.reshape(tokens, experts * atoms)
    keep = retained_count(experts * atoms, fraction)
    mask = torch.zeros_like(flattened, dtype=torch.bool)
    if keep:
        ranked = flattened.argsort(dim=1, descending=True, stable=True)
        mask.scatter_(1, ranked[:, :keep], True)
    return mask.reshape_as(scores)


def global_tile_topk_mask(
    contribution_norm: torch.Tensor, fraction: float, tile_size: int
) -> torch.Tensor:
    """Select complete expert-local tiles by summed squared atom norm."""

    tokens, experts, atoms = _validate_scores(contribution_norm)
    if tile_size < 1 or atoms % tile_size:
        raise ValueError("tile_size must be positive and divide the atom count")
    tiles_per_expert = atoms // tile_size
    tile_scores = (
        contribution_norm.float()
        .square()
        .reshape(tokens, experts, tiles_per_expert, tile_size)
        .sum(dim=3)
    )
    flattened = tile_scores.reshape(tokens, experts * tiles_per_expert)
    keep = retained_count(experts * tiles_per_expert, fraction)
    tile_mask = torch.zeros_like(flattened, dtype=torch.bool)
    if keep:
        ranked = flattened.argsort(dim=1, descending=True, stable=True)
        tile_mask.scatter_(1, ranked[:, :keep], True)
    return (
        tile_mask.reshape(tokens, experts, tiles_per_expert, 1)
        .expand(-1, -1, -1, tile_size)
        .reshape(tokens, experts, atoms)
    )


def atomic_selector_masks(
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    selected_down_norms: torch.Tensor,
    fraction: float,
) -> dict[str, torch.Tensor]:
    """Construct every fixed Stage-A selector from exact routed activations."""

    tokens, experts, atoms = _validate_scores(activations)
    if router_weights.shape != (tokens, experts):
        raise ValueError("router_weights must have shape [tokens, experts]")
    if selected_down_norms.shape != (tokens, experts, atoms):
        raise ValueError("selected_down_norms must match activations")
    if not torch.isfinite(router_weights).all() or not torch.isfinite(
        selected_down_norms
    ).all():
        raise ValueError("router weights and down norms must be finite")
    absolute = activations.float().abs()
    norm_weighted = absolute * selected_down_norms.float()
    contribution = norm_weighted * router_weights.float().abs().unsqueeze(-1)
    return {
        "per_expert_activation": per_expert_topk_mask(absolute, fraction),
        "per_expert_contribution": per_expert_topk_mask(
            norm_weighted, fraction
        ),
        "global_contribution": global_topk_mask(contribution, fraction),
        "tile16_contribution": global_tile_topk_mask(contribution, fraction, 16),
        "tile32_contribution": global_tile_topk_mask(contribution, fraction, 32),
        "tile64_contribution": global_tile_topk_mask(contribution, fraction, 64),
    }


def reconstruct_weighted_atoms(
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    down_columns: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reference atom reconstruction for small exact tests.

    `down_columns` is expert-slot aligned and has shape
    `[tokens, experts, atoms, hidden]`. Production evaluation streams the real
    expert matrices instead of materialising this large tensor.
    """

    tokens, experts, atoms = _validate_scores(activations)
    if router_weights.shape != (tokens, experts):
        raise ValueError("router_weights must have shape [tokens, experts]")
    if down_columns.ndim != 4 or down_columns.shape[:3] != (
        tokens,
        experts,
        atoms,
    ):
        raise ValueError(
            "down_columns must have shape [tokens, experts, atoms, hidden]"
        )
    if mask.shape != activations.shape or mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean and match activations")
    coefficients = (
        activations.double()
        * router_weights.double().unsqueeze(-1)
        * mask.double()
    )
    return torch.einsum("tea,teah->th", coefficients, down_columns.double())


def relative_routed_l2(
    reference_routed: torch.Tensor, candidate_routed: torch.Tensor
) -> torch.Tensor:
    if reference_routed.ndim != 2:
        raise ValueError("reference_routed must have shape [tokens, hidden]")
    if candidate_routed.shape != reference_routed.shape:
        raise ValueError("candidate_routed must match reference_routed")
    reference = reference_routed.double()
    error = candidate_routed.double() - reference
    return (
        error.square().sum(dim=1)
        / reference.square().sum(dim=1).clamp_min(1e-30)
    ).clamp_min(0.0).sqrt()


def delta_patched_hidden(
    official_teacher: torch.Tensor,
    original_routed: torch.Tensor,
    candidate_routed: torch.Tensor,
) -> torch.Tensor:
    if official_teacher.shape != original_routed.shape:
        raise ValueError("official_teacher and original_routed must have equal shape")
    if candidate_routed.shape != original_routed.shape:
        raise ValueError("candidate_routed and original_routed must have equal shape")
    return (
        official_teacher.float()
        + candidate_routed.float()
        - original_routed.float()
    ).to(official_teacher.dtype)


def support_known_accounting(
    retained_atoms: torch.Tensor,
    *,
    atoms_per_expert: int,
    hidden_size: int,
    bytes_per_value: int = 2,
    page_size: int = 4096,
) -> dict[str, Any]:
    """Ideal atom traffic plus tensor-local row-major 4-KiB page traffic.

    The mask-independent down-page result is exact for the DeepSeek-V2-Lite
    geometry used here: one BF16 down row is at most one page, while a fixed
    column traverses every tensor-local page. Gate and up atom rows are exactly
    one page each. The function rejects other geometries instead of silently
    presenting this model-specific page calculation as general.
    """

    if retained_atoms.ndim != 2:
        raise ValueError("retained_atoms must have shape [tokens, experts]")
    if retained_atoms.dtype == torch.bool or retained_atoms.is_floating_point():
        raise ValueError("retained_atoms must use an integer dtype")
    if atoms_per_expert < 1 or hidden_size < 1 or bytes_per_value < 1:
        raise ValueError("dimensions and bytes_per_value must be positive")
    if (retained_atoms < 0).any() or (retained_atoms > atoms_per_expert).any():
        raise ValueError("retained atom counts are outside the expert width")
    row_bytes = hidden_size * bytes_per_value
    down_row_bytes = atoms_per_expert * bytes_per_value
    if row_bytes != page_size or down_row_bytes > page_size:
        raise ValueError(
            "tensor-local page accounting requires one-page gate/up rows and "
            "a down row no larger than one page"
        )

    counts = retained_atoms.to(torch.int64)
    tokens, experts = counts.shape
    total_atoms = experts * atoms_per_expert
    selected = counts.sum(dim=1)
    values_per_atom = 3 * hidden_size
    ideal_bytes = selected * values_per_atom * bytes_per_value
    ideal_macs = selected * values_per_atom
    full_bytes = total_atoms * values_per_atom * bytes_per_value
    full_macs = total_atoms * values_per_atom

    down_tensor_bytes = hidden_size * atoms_per_expert * bytes_per_value
    down_pages = math.ceil(down_tensor_bytes / page_size)
    active = counts.gt(0).to(torch.int64)
    pages = (2 * counts + active * down_pages).sum(dim=1)
    full_pages = experts * (2 * atoms_per_expert + down_pages)
    page_bytes = pages * page_size
    full_page_bytes = full_pages * page_size

    return {
        "assumption": (
            "support known before gate/up/down loads; BF16; tensor-local, "
            "page-aligned row-major matrices; excludes index/selector traffic"
        ),
        "retained_atoms": selected.tolist(),
        "retained_atom_fraction": (selected.double() / total_atoms).tolist(),
        "ideal_weight_bytes": ideal_bytes.tolist(),
        "ideal_weight_byte_fraction": (ideal_bytes.double() / full_bytes).tolist(),
        "ideal_macs": ideal_macs.tolist(),
        "ideal_mac_fraction": (ideal_macs.double() / full_macs).tolist(),
        "tensor_local_pages_4k": pages.tolist(),
        "tensor_local_page_bytes": page_bytes.tolist(),
        "tensor_local_page_byte_fraction": (
            page_bytes.double() / full_page_bytes
        ).tolist(),
        "full_routed_atoms": total_atoms,
        "full_ideal_weight_bytes": full_bytes,
        "full_ideal_macs": full_macs,
        "full_tensor_local_pages_4k": full_pages,
        "full_tensor_local_page_bytes": full_page_bytes,
    }
