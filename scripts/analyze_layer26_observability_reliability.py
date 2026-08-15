from __future__ import annotations

import time

import torch
from safetensors.torch import load_file

from estimate_layer26_observability import (
    RANKS as INITIAL_RANKS,
    SEED,
    behavioral_metrics,
    corpus_blocks,
)
from moe_lab.behavioral import project, sample_fisher_score_gradient_replicates
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT, envelope, write_json


SAMPLES_PER_STATE = 8
RANKS = (*INITIAL_RANKS, 512, 1024)


def spectrum(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    eigenvectors = eigenvectors.flip(1)
    trace = float(eigenvalues.sum().item())
    fractions = torch.cumsum(eigenvalues, dim=0) / max(trace, 1e-30)

    def threshold_rank(threshold: float) -> int:
        return int(torch.searchsorted(fractions, threshold).item()) + 1

    return eigenvalues, eigenvectors, {
        "trace": trace,
        "effective_rank": float(
            eigenvalues.sum().square().div(eigenvalues.square().sum()).item()
        ),
        "r50": threshold_rank(0.50),
        "r80": threshold_rank(0.80),
        "r90": threshold_rank(0.90),
        "r95": threshold_rank(0.95),
        "captured_trace_fraction": {
            str(rank): float(fractions[rank - 1].item()) for rank in RANKS
        },
        "top_eigenvalues": eigenvalues[:256].cpu().tolist(),
    }


def heldout_capture(basis: torch.Tensor, gram: torch.Tensor, rank: int) -> float:
    selected = basis[:, :rank]
    captured = torch.trace(selected.T @ gram @ selected)
    return float((captured / torch.trace(gram)).item())


def direct_damage_weights(
    teacher: torch.Tensor,
    quantized: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    weights = []
    for start in range(0, teacher.shape[0], batch_size):
        exact = teacher[start : start + batch_size]
        approximate = quantized[start : start + batch_size]
        exact_norm = exact.float()
        exact_norm = exact_norm * torch.rsqrt(
            exact_norm.square().mean(-1, keepdim=True) + 1e-6
        )
        exact_norm = exact_norm.to(exact.dtype) * norm_weight
        approximate_norm = approximate.float()
        approximate_norm = approximate_norm * torch.rsqrt(
            approximate_norm.square().mean(-1, keepdim=True) + 1e-6
        )
        approximate_norm = approximate_norm.to(approximate.dtype) * norm_weight
        exact_logits = torch.nn.functional.linear(exact_norm, lm_head).float()
        approximate_logits = torch.nn.functional.linear(
            approximate_norm, lm_head
        ).float()
        exact_log_probs = torch.log_softmax(exact_logits, dim=-1)
        approximate_log_probs = torch.log_softmax(approximate_logits, dim=-1)
        token_kl = (
            exact_log_probs.exp() * (exact_log_probs - approximate_log_probs)
        ).sum(-1)
        weights.append(token_kl.cpu())
    return torch.cat(weights).clamp_min(0)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("reliability analysis requires CUDA")
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    states = load_file(
        ROOT / "data" / "traces" / "layer26_teacher_quant3_final_states.safetensors",
        device="cpu",
    )
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    train_teacher = states["teacher_train"].reshape(-1, 2048).to(device)
    train_quantized = states["quantized_train"].reshape(-1, 2048).to(device)

    replicates = sample_fisher_score_gradient_replicates(
        train_teacher,
        norm_weight,
        lm_head,
        batch_size=32,
        seed=SEED,
        samples_per_state=SAMPLES_PER_STATE,
    ).to(device)
    print(f"sampled_gradients={tuple(replicates.shape)}", flush=True)
    flattened = replicates.reshape(-1, replicates.shape[-1])
    gram = flattened.T @ flattened / flattened.shape[0]
    eigenvalues, behavioral_basis, main_spectrum = spectrum(gram)
    print(
        f"full r80={main_spectrum['r80']} r90={main_spectrum['r90']} "
        f"effective_rank={main_spectrum['effective_rank']:.2f}",
        flush=True,
    )

    first = replicates[: SAMPLES_PER_STATE // 2].reshape(-1, 2048)
    second = replicates[SAMPLES_PER_STATE // 2 :].reshape(-1, 2048)
    first_gram = first.T @ first / first.shape[0]
    second_gram = second.T @ second / second.shape[0]
    _, first_basis, first_spectrum = spectrum(first_gram)
    _, second_basis, second_spectrum = spectrum(second_gram)
    split_half = {
        "first_half": first_spectrum,
        "second_half": second_spectrum,
        "first_basis_capture_on_second": {
            str(rank): heldout_capture(first_basis, second_gram, rank)
            for rank in RANKS
        },
        "second_basis_capture_on_first": {
            str(rank): heldout_capture(second_basis, first_gram, rank)
            for rank in RANKS
        },
        "subspace_overlap": {
            str(rank): float(
                (
                    (first_basis[:, :rank].T @ second_basis[:, :rank])
                    .square()
                    .sum()
                    / rank
                ).item()
            )
            for rank in RANKS
        },
    }
    print(
        "split_half_r90=",
        first_spectrum["r90"],
        second_spectrum["r90"],
        flush=True,
    )

    train_delta = train_teacher.float() - train_quantized.float()
    damage_weights = direct_damage_weights(
        train_teacher, train_quantized, norm_weight, lm_head
    ).to(device)
    weighted_delta = train_delta * torch.sqrt(
        damage_weights / damage_weights.mean().clamp_min(1e-30)
    ).unsqueeze(-1)
    weighted_covariance = weighted_delta.T @ weighted_delta / weighted_delta.shape[0]
    _, damage_weighted_basis, weighted_spectrum = spectrum(weighted_covariance)
    print(
        f"damage_weighted_error_r90={weighted_spectrum['r90']}", flush=True
    )

    split_ids = {
        split: corpus_blocks(model_dir, split, blocks)
        for split, blocks in {"validation": 8, "test": 8}.items()
    }
    results = {}
    for split in ("validation", "test"):
        teacher = states[f"teacher_{split}"]
        quantized = states[f"quantized_{split}"]
        delta = (teacher.float() - quantized.float()).reshape(-1, 2048).to(device)
        baseline = behavioral_metrics(
            teacher, quantized, split_ids[split], norm_weight, lm_head
        )
        methods = {}
        for method, basis in (
            ("behavioral_8sample", behavioral_basis),
            ("damage_weighted_error_pca", damage_weighted_basis),
        ):
            rows = []
            for rank in RANKS:
                correction = project(delta, basis, rank).view_as(teacher).cpu()
                corrected = (quantized.float() + correction).to(torch.bfloat16)
                metrics = behavioral_metrics(
                    teacher, corrected, split_ids[split], norm_weight, lm_head
                )
                metrics["kl_damage_recovery"] = 1.0 - (
                    metrics["teacher_to_candidate_kl"]
                    / baseline["teacher_to_candidate_kl"]
                )
                metrics["ce_damage_recovery"] = 1.0 - (
                    metrics["cross_entropy_delta"]
                    / baseline["cross_entropy_delta"]
                )
                rows.append({"rank": rank, **metrics})
                print(
                    f"{split} {method} r={rank} "
                    f"KLrec={metrics['kl_damage_recovery']:.3f} "
                    f"CErec={metrics['ce_damage_recovery']:.3f}",
                    flush=True,
                )
            methods[method] = rows
        results[split] = {"quant3_baseline": baseline, "methods": methods}

    report = {
        "status": "complete",
        "experiment": "layer26_h0_observability_reliability",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "train_states": train_teacher.shape[0],
        "fisher_samples_per_train_state": SAMPLES_PER_STATE,
        "spectrum": main_spectrum,
        "split_half": split_half,
        "damage_weighted_error_spectrum": weighted_spectrum,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_behavioral_observability_reliability.json",
        envelope("behavioral_observability_reliability", report),
    )
    print(path)
