from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


@dataclass(frozen=True)
class RouteCandidate:
    subset_index: int
    experts: tuple[int, ...]
    local_kl: float
    mass_loss: float
    natural: bool = False

    @property
    def expert_mask(self) -> int:
        return experts_to_mask(self.experts)


@dataclass(frozen=True)
class BlockSolution:
    route_indices: tuple[int, ...]
    union_mask: int
    total_local_kl: float
    method: str
    diagnostics: dict[str, Any] | None = None

    @property
    def union_count(self) -> int:
        return self.union_mask.bit_count()


def highs_optimal_control(
    diagnostics: dict[str, Any],
    observed_cold_union_count: int,
    *,
    gap_tolerance: float = 1e-12,
    objective_tolerance: float = 1e-6,
) -> bool:
    """Validate an optimal HiGHS result without demanding float bit equality."""

    return bool(
        diagnostics.get("status") == 0
        and diagnostics.get("success") is True
        and abs(float(diagnostics.get("mip_gap", math.inf))) <= gap_tolerance
        and math.isclose(
            float(diagnostics.get("objective", math.inf)),
            float(observed_cold_union_count),
            rel_tol=0.0,
            abs_tol=objective_tolerance,
        )
    )


def experts_to_mask(experts: tuple[int, ...] | list[int]) -> int:
    mask = 0
    seen: set[int] = set()
    for expert in experts:
        expert = int(expert)
        if not 0 <= expert < 64:
            raise ValueError("expert ids must be in [0, 63]")
        if expert in seen:
            raise ValueError("a route cannot contain a duplicate expert")
        seen.add(expert)
        mask |= 1 << expert
    return mask


def mask_to_experts(mask: int) -> list[int]:
    if mask < 0:
        raise ValueError("mask must be non-negative")
    return [expert for expert in range(64) if mask & (1 << expert)]


def original_subset_index(subsets: torch.Tensor) -> int:
    if subsets.ndim != 2 or subsets.shape[1] != 6:
        raise ValueError("subsets must have shape [routes, 6]")
    original = torch.arange(6, dtype=subsets.dtype, device=subsets.device)
    matches = (subsets == original).all(dim=1).nonzero(as_tuple=False).squeeze(1)
    if matches.numel() != 1:
        raise ValueError("subsets must contain the natural first-six route once")
    return int(matches.item())


def build_candidate_slates(
    top12_ids: torch.Tensor,
    top12_weights: torch.Tensor,
    subset_kl: torch.Tensor,
    subsets: torch.Tensor,
    *,
    threshold: float,
    cap: int,
) -> list[list[RouteCandidate]]:
    """Build deterministic KL-eligible capped slates with natural fallback."""

    if top12_ids.ndim != 2 or top12_ids.shape[1] != 12:
        raise ValueError("top12_ids must have shape [tokens, 12]")
    if top12_weights.shape != top12_ids.shape:
        raise ValueError("top12_weights must match top12_ids")
    if subset_kl.shape != (top12_ids.shape[0], subsets.shape[0]):
        raise ValueError("subset_kl shape does not match tokens and subsets")
    if cap < 1 or threshold < 0:
        raise ValueError("cap must be positive and threshold non-negative")
    if cap > subsets.shape[0]:
        raise ValueError("cap exceeds available routes")
    original_index = original_subset_index(subsets)
    slates: list[list[RouteCandidate]] = []
    subsets_cpu = subsets.long().cpu()
    for token in range(top12_ids.shape[0]):
        kl = subset_kl[token].double().cpu().numpy()
        eligible = np.flatnonzero(kl <= threshold).tolist()
        if original_index not in eligible:
            eligible.append(original_index)
        ordered = sorted(eligible, key=lambda index: (float(kl[index]), index))
        selected = ordered[:cap]
        if original_index not in selected:
            selected[-1] = original_index
            selected.sort(key=lambda index: (float(kl[index]), index))
        ids = top12_ids[token].long().cpu()
        weights = top12_weights[token].double().cpu()
        natural_mass = float(weights[:6].sum().item())
        candidates = []
        for subset_index in selected:
            positions = subsets_cpu[subset_index]
            experts = tuple(int(value) for value in ids[positions].tolist())
            if len(set(experts)) != 6:
                raise RuntimeError("top-12 trace contains a duplicate expert")
            selected_mass = float(weights[positions].sum().item())
            candidates.append(
                RouteCandidate(
                    subset_index=subset_index,
                    experts=experts,
                    local_kl=float(kl[subset_index]),
                    mass_loss=max(0.0, natural_mass - selected_mass),
                    natural=subset_index == original_index,
                )
            )
        if sum(candidate.natural for candidate in candidates) != 1:
            raise RuntimeError("every candidate slate must contain natural exactly once")
        slates.append(candidates)
    return slates


def _solution_from_path(
    slates: list[list[RouteCandidate]], path: tuple[int, ...], method: str,
    diagnostics: dict[str, Any] | None = None,
) -> BlockSolution:
    if len(path) != len(slates):
        raise ValueError("path must select one route per token")
    union = 0
    total_kl = 0.0
    for candidates, local_index in zip(slates, path, strict=True):
        if not 0 <= local_index < len(candidates):
            raise ValueError("route index is outside its slate")
        candidate = candidates[local_index]
        union |= candidate.expert_mask
        total_kl += candidate.local_kl
    return BlockSolution(path, union, total_kl, method, diagnostics)


def natural_solution(slates: list[list[RouteCandidate]]) -> BlockSolution:
    path = tuple(
        next(index for index, candidate in enumerate(candidates) if candidate.natural)
        for candidates in slates
    )
    return _solution_from_path(slates, path, "natural")


def marginal_union_greedy(slates: list[list[RouteCandidate]]) -> BlockSolution:
    union = 0
    path = []
    for candidates in slates:
        local_index = min(
            range(len(candidates)),
            key=lambda index: (
                (candidates[index].expert_mask & ~union).bit_count(),
                candidates[index].local_kl,
                candidates[index].subset_index,
            ),
        )
        path.append(local_index)
        union |= candidates[local_index].expert_mask
    return _solution_from_path(slates, tuple(path), "marginal_union_greedy")


def mass_budget_solution(
    slates: list[list[RouteCandidate]], delta: float
) -> BlockSolution:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    union = 0
    path = []
    for candidates in slates:
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.mass_loss <= delta + 1e-12 or candidate.natural
        ]
        local_index = min(
            eligible,
            key=lambda index: (
                (candidates[index].expert_mask & ~union).bit_count(),
                candidates[index].mass_loss,
                candidates[index].local_kl,
                candidates[index].subset_index,
            ),
        )
        path.append(local_index)
        union |= candidates[local_index].expert_mask
    return _solution_from_path(
        slates, tuple(path), f"mass_budget_delta_{delta:g}"
    )


def fixed_cache_prior_solution(
    slates: list[list[RouteCandidate]], cache_mask: int
) -> BlockSolution:
    path = tuple(
        min(
            range(len(candidates)),
            key=lambda index: (
                (candidates[index].expert_mask & ~cache_mask).bit_count(),
                candidates[index].local_kl,
                candidates[index].subset_index,
            ),
        )
        for candidates in slates
    )
    return _solution_from_path(slates, path, "fixed_cache_prior")


def eligible_set_pruning(slates: list[list[RouteCandidate]]) -> BlockSolution:
    """Frequency-ranked smallest expert pool covering every token slate."""

    frequencies = [0] * 64
    for candidates in slates:
        for candidate in candidates:
            for expert in candidate.experts:
                frequencies[expert] += 1
    ranking = sorted(range(64), key=lambda expert: (-frequencies[expert], expert))
    pool = 0
    feasible: list[list[int]] | None = None
    for expert in ranking:
        pool |= 1 << expert
        local_feasible = [
            [
                index
                for index, candidate in enumerate(candidates)
                if candidate.expert_mask & ~pool == 0
            ]
            for candidates in slates
        ]
        if all(local_feasible):
            feasible = local_feasible
            break
    if feasible is None:
        raise RuntimeError("all experts failed to cover candidate slates")
    path = tuple(
        min(
            indices,
            key=lambda index: (
                candidates[index].local_kl,
                candidates[index].subset_index,
            ),
        )
        for candidates, indices in zip(slates, feasible, strict=True)
    )
    return _solution_from_path(
        slates,
        path,
        "eligible_set_pruning",
        {"eligible_pool_mask": pool, "eligible_pool_count": pool.bit_count()},
    )


def beam_union_solution(
    slates: list[list[RouteCandidate]], width: int
) -> BlockSolution:
    if width < 1:
        raise ValueError("beam width must be positive")
    # mask -> (total KL, path); identical masks retain the lower-KL stable path.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    maximum_states = 1
    for candidates in slates:
        expanded: dict[int, tuple[float, tuple[int, ...]]] = {}
        for union, (total_kl, path) in states.items():
            for index, candidate in enumerate(candidates):
                new_union = union | candidate.expert_mask
                value = (total_kl + candidate.local_kl, path + (index,))
                previous = expanded.get(new_union)
                if previous is None or value < previous:
                    expanded[new_union] = value
        ordered = sorted(
            expanded.items(),
            key=lambda item: (
                item[0].bit_count(), item[1][0], item[0], item[1][1]
            ),
        )
        states = dict(ordered[:width])
        maximum_states = max(maximum_states, len(states))
    best_union, (best_kl, best_path) = min(
        states.items(),
        key=lambda item: (
            item[0].bit_count(), item[1][0], item[0], item[1][1]
        ),
    )
    solution = _solution_from_path(
        slates,
        best_path,
        f"beam_{width}",
        {"maximum_retained_states": maximum_states},
    )
    if solution.union_mask != best_union or not math.isclose(
        solution.total_local_kl, best_kl, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("beam state reconstruction mismatch")
    return solution


def exact_ilp_solution(
    slates: list[list[RouteCandidate]],
    *,
    cache_mask: int = 0,
    time_limit_seconds: float = 30.0,
) -> BlockSolution:
    """Solve the binary route-union ILP exactly with SciPy/HiGHS."""

    if not slates or any(not candidates for candidates in slates):
        raise ValueError("slates must contain at least one candidate per token")
    if time_limit_seconds <= 0:
        raise ValueError("time limit must be positive")
    offsets = [0]
    for candidates in slates:
        offsets.append(offsets[-1] + len(candidates))
    x_count = offsets[-1]
    y_offset = x_count
    variable_count = x_count + 64
    objective = np.zeros(variable_count, dtype=np.float64)
    for expert in range(64):
        if not cache_mask & (1 << expert):
            objective[y_offset + expert] = 1.0

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    constraint_row = 0
    for token, candidates in enumerate(slates):
        for variable in range(offsets[token], offsets[token + 1]):
            rows.append(constraint_row)
            columns.append(variable)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        constraint_row += 1
        for local_index, candidate in enumerate(candidates):
            x_variable = offsets[token] + local_index
            for expert in candidate.experts:
                rows.extend((constraint_row, constraint_row))
                columns.extend((x_variable, y_offset + expert))
                values.extend((1.0, -1.0))
                lower.append(-np.inf)
                upper.append(0.0)
                constraint_row += 1
    matrix = coo_array(
        (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
        shape=(constraint_row, variable_count),
    ).tocsc()
    constraint = LinearConstraint(
        matrix, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    )
    started = time.perf_counter()
    result = milp(
        objective,
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=constraint,
        options={
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    seconds = time.perf_counter() - started
    if not result.success or result.status != 0 or result.x is None:
        raise RuntimeError(
            f"HiGHS failed to prove optimality: status={result.status}, "
            f"message={result.message}"
        )
    solver_path = []
    for token, candidates in enumerate(slates):
        local = result.x[offsets[token] : offsets[token + 1]]
        index = int(np.argmax(local))
        if local[index] < 0.5 or not math.isclose(
            float(local.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RuntimeError("ILP route variables are not integral one-hot")
        solver_path.append(index)
    solver_union = 0
    for candidates, index in zip(slates, solver_path, strict=True):
        solver_union |= candidates[index].expert_mask
    # Canonicalize route choices inside the optimal expert set: lowest local
    # KL, then stable subset index. This cannot worsen the union objective.
    path = [
        min(
            (
                index
                for index, candidate in enumerate(candidates)
                if candidate.expert_mask & ~solver_union == 0
            ),
            key=lambda index: (
                candidates[index].local_kl,
                candidates[index].subset_index,
            ),
        )
        for candidates in slates
    ]
    objective_value = float(result.fun)
    solution = _solution_from_path(
        slates,
        tuple(path),
        "exact_ilp" if cache_mask == 0 else "exact_ilp_fixed_cache",
        {
            "solver": "scipy.optimize.milp/HiGHS",
            "status": int(result.status),
            "message": str(result.message),
            "success": bool(result.success),
            "objective": objective_value,
            "mip_gap": float(getattr(result, "mip_gap", 0.0)),
            "mip_node_count": int(getattr(result, "mip_node_count", 0)),
            "solve_seconds": seconds,
            "variables": variable_count,
            "binary_route_variables": x_count,
            "binary_expert_variables": 64,
            "constraints": constraint_row,
            "cache_mask": cache_mask,
            "solver_selected_union_mask": solver_union,
            "route_choices_canonicalized_within_solver_union": True,
        },
    )
    observed_objective = (solution.union_mask & ~cache_mask).bit_count()
    if not math.isclose(
        objective_value, float(observed_objective), rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(
            "ILP objective does not equal the extracted cold expert union"
        )
    return solution


def brute_force_solution(slates: list[list[RouteCandidate]]) -> BlockSolution:
    """Tiny-test oracle; never use for production-sized experiment blocks."""

    combinations = itertools.product(*(range(len(candidates)) for candidates in slates))
    best: BlockSolution | None = None
    for path in combinations:
        candidate = _solution_from_path(slates, tuple(path), "brute_force")
        key = (candidate.union_count, candidate.total_local_kl, candidate.route_indices)
        if best is None or key < (
            best.union_count, best.total_local_kl, best.route_indices
        ):
            best = candidate
    if best is None:
        raise ValueError("empty slates")
    return best


def solution_metrics(
    slates: list[list[RouteCandidate]],
    solution: BlockSolution,
    *,
    natural: BlockSolution | None = None,
    cache_mask: int = 0,
) -> dict[str, Any]:
    if natural is None:
        natural = natural_solution(slates)
    chosen = [
        candidates[index]
        for candidates, index in zip(slates, solution.route_indices, strict=True)
    ]
    natural_candidates = [
        candidates[index]
        for candidates, index in zip(slates, natural.route_indices, strict=True)
    ]
    jaccards = []
    changed = 0
    for candidate, reference in zip(chosen, natural_candidates, strict=True):
        intersection = len(set(candidate.experts) & set(reference.experts))
        jaccards.append(intersection / (12 - intersection))
        changed += int(candidate.subset_index != reference.subset_index)
    return {
        "method": solution.method,
        "route_local_indices": list(solution.route_indices),
        "subset_indices": [candidate.subset_index for candidate in chosen],
        "selected_experts": [list(candidate.experts) for candidate in chosen],
        "union_mask": solution.union_mask,
        "union_experts": mask_to_experts(solution.union_mask),
        "union_count": solution.union_count,
        "cold_union_count": (solution.union_mask & ~cache_mask).bit_count(),
        "total_local_kl": solution.total_local_kl,
        "mean_local_kl": solution.total_local_kl / len(slates),
        "mean_router_mass_loss": sum(candidate.mass_loss for candidate in chosen)
        / len(chosen),
        "changed_token_fraction": changed / len(chosen),
        "mean_natural_route_jaccard": sum(jaccards) / len(jaccards),
        "diagnostics": solution.diagnostics,
    }
