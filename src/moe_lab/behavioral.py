from __future__ import annotations

import torch


def rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    hidden_float = hidden_states.float()
    inverse_rms = torch.rsqrt(hidden_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (hidden_float * inverse_rms).to(hidden_states.dtype) * weight


def rmsnorm_pullback(
    hidden_states: torch.Tensor,
    output_gradient: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the transpose Jacobian of RMSNorm to an output-space gradient."""
    hidden = hidden_states.float()
    gradient = output_gradient.float()
    norm_weight = weight.float()
    dimension = hidden.shape[-1]
    rms = torch.sqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps)
    weighted_gradient = gradient * norm_weight
    radial = (weighted_gradient * hidden).sum(dim=-1, keepdim=True)
    return weighted_gradient / rms - hidden * radial / (dimension * rms.pow(3))


@torch.inference_mode()
def sample_fisher_score_gradient_replicates(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int,
    seed: int,
    samples_per_state: int,
) -> torch.Tensor:
    """Sample categorical score gradients with shape [samples, states, hidden].

    The class draws are exact samples from the teacher distribution. The
    probability-weighted LM-head mean is accumulated in the checkpoint dtype,
    matching the numerical precision used by the model forward pass.
    """
    if samples_per_state < 1:
        raise ValueError("samples_per_state must be positive")
    device = hidden_states.device
    generator = torch.Generator(device=device).manual_seed(seed)
    gradients = torch.empty(
        samples_per_state,
        hidden_states.shape[0],
        hidden_states.shape[-1],
        dtype=torch.float32,
        device="cpu",
    )
    for start in range(0, hidden_states.shape[0], batch_size):
        hidden = hidden_states[start : start + batch_size]
        normalized = rmsnorm(hidden, norm_weight)
        logits = torch.nn.functional.linear(normalized, lm_head).float()
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probabilities,
            samples_per_state,
            replacement=True,
            generator=generator,
        )
        expected_weight = (probabilities.to(lm_head.dtype) @ lm_head).float()
        score_at_norm_output = (
            lm_head[sampled].float() - expected_weight.unsqueeze(1)
        )
        pulled_back = rmsnorm_pullback(
            hidden.unsqueeze(1), score_at_norm_output, norm_weight
        )
        stop = start + hidden.shape[0]
        gradients[:, start:stop] = pulled_back.permute(1, 0, 2).cpu()
    return gradients


@torch.inference_mode()
def sample_fisher_score_gradients(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """Sample one categorical score gradient per hidden state."""
    return sample_fisher_score_gradient_replicates(
        hidden_states,
        norm_weight,
        lm_head,
        batch_size,
        seed,
        samples_per_state=1,
    )[0]


def project(tensor: torch.Tensor, basis: torch.Tensor, rank: int) -> torch.Tensor:
    selected = basis[:, :rank]
    return (tensor.float() @ selected.float()) @ selected.float().T
