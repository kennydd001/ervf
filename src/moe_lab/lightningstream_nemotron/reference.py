"""Independent numpy implementations of the nemotron_h module semantics.

Written from the semantics frozen in the N3 preregistration.  Where a different
algorithm is available it is used deliberately: the Mamba-2 scan here is a plain
sequential recurrence, while the official ``torch_forward`` uses a chunked SSD
factorisation.  Agreement between two different algorithms is real evidence;
agreement between two copies of the same algorithm would not be.

Everything runs in float32 on CPU.  No performance claim is attached to any of
this - it is a correctness reference only.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# elementwise helpers
# --------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Branch on sign to avoid overflow in exp for large-magnitude inputs.
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def relu2(x: np.ndarray) -> np.ndarray:
    """ACT2FN['relu2'] == relu(x) ** 2."""
    relu = np.maximum(x, 0.0)
    return relu * relu


def softplus(x: np.ndarray) -> np.ndarray:
    # log1p(exp(x)) via the numerically stable max(x,0) + log1p(exp(-|x|)).
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


# --------------------------------------------------------------------------
# norms
# --------------------------------------------------------------------------

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """NemotronHRMSNorm: variance in float32, rsqrt, float32 weight."""
    x64 = x.astype(np.float64)
    variance = np.mean(x64 * x64, axis=-1, keepdims=True)
    normed = x64 / np.sqrt(variance + eps)
    return (weight.astype(np.float64) * normed)


def gated_rms_norm(x: np.ndarray, gate: np.ndarray, weight: np.ndarray,
                   eps: float, group_size: int) -> np.ndarray:
    """mamba_ssm ``rmsnorm_fn`` with ``norm_before_gate=False``.

    The gate is applied first (``x * silu(z)``), then a grouped RMS norm runs
    over blocks of ``group_size`` along the last axis, then the full-width
    weight is applied.

    NOTE: this single operation is supplied by us for BOTH sides of the N3
    comparison, because ``mamba_ssm`` cannot be installed here.  It is therefore
    NOT independently validated by N3 and is recorded as such.
    """
    x64 = x.astype(np.float64)
    gated = x64 * silu(gate.astype(np.float64))
    shape = gated.shape
    grouped = gated.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    variance = np.mean(grouped * grouped, axis=-1, keepdims=True)
    normed = (grouped / np.sqrt(variance + eps)).reshape(shape)
    return normed * weight.astype(np.float64)


# --------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------

def router(hidden: np.ndarray, weight: np.ndarray, correction_bias: np.ndarray,
           top_k: int, routed_scaling_factor: float,
           norm_topk_prob: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """NemotronHTopkRouter.forward.

    Selection uses ``scores + correction_bias``; the returned weights come from
    the raw ``scores`` WITHOUT the bias.  Getting that backwards is the single
    easiest way to produce a plausible but wrong router.
    """
    tokens = hidden.reshape(-1, hidden.shape[-1]).astype(np.float64)
    logits = tokens @ weight.astype(np.float64).T
    scores = sigmoid(logits)
    scores_for_choice = scores + correction_bias.astype(np.float64)[None, :]

    # n_group == topk_group == 1 makes the official group mask a no-op; the
    # single group is always selected and masked_fill clears nothing.
    order = np.argsort(-scores_for_choice, axis=-1, kind="stable")
    indices = order[:, :top_k]

    weights = np.take_along_axis(scores, indices, axis=-1)
    if norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor

    sorted_choice = np.take_along_axis(scores_for_choice, order, axis=-1)
    diagnostics = {
        "logits": logits,
        "scores": scores,
        "scores_for_choice": scores_for_choice,
        # Margin between the last selected and first rejected expert.
        "tie_margin": sorted_choice[:, top_k - 1] - sorted_choice[:, top_k],
    }
    return indices, weights, diagnostics


def mlp_relu2(x: np.ndarray, up_weight: np.ndarray, down_weight: np.ndarray) -> np.ndarray:
    """NemotronHMLP: down_proj(relu2(up_proj(x))), no bias."""
    hidden = x.astype(np.float64) @ up_weight.astype(np.float64).T
    return relu2(hidden) @ down_weight.astype(np.float64).T


def moe_forward(hidden: np.ndarray, expert_up: dict[int, np.ndarray],
                expert_down: dict[int, np.ndarray], router_weight: np.ndarray,
                correction_bias: np.ndarray, shared_up: np.ndarray,
                shared_down: np.ndarray, top_k: int,
                routed_scaling_factor: float, norm_topk_prob: bool) -> dict:
    """NemotronHMOE.forward: routed sum plus an UNGATED shared expert."""
    orig_shape = hidden.shape
    tokens = hidden.reshape(-1, orig_shape[-1])
    indices, weights, diag = router(
        tokens, router_weight, correction_bias, top_k,
        routed_scaling_factor, norm_topk_prob,
    )

    routed = np.zeros_like(tokens, dtype=np.float64)
    for token in range(tokens.shape[0]):
        for slot in range(top_k):
            expert = int(indices[token, slot])
            out = mlp_relu2(tokens[token:token + 1], expert_up[expert], expert_down[expert])
            routed[token] += out[0] * weights[token, slot]

    shared = mlp_relu2(tokens, shared_up, shared_down)
    return {
        "routed": routed.reshape(orig_shape),
        "shared": shared.reshape(orig_shape),
        "output": (routed + shared).reshape(orig_shape),
        "indices": indices,
        "weights": weights,
        "tie_margin": diag["tie_margin"],
        "router_logits": diag["logits"],
    }


# --------------------------------------------------------------------------
# attention -- NoPE: there is no rotary embedding anywhere in this model
# --------------------------------------------------------------------------

def attention_forward(hidden: np.ndarray, q_w: np.ndarray, k_w: np.ndarray,
                      v_w: np.ndarray, o_w: np.ndarray, num_heads: int,
                      num_kv_heads: int, head_dim: int) -> dict:
    """Causal GQA with no positional encoding and no bias."""
    seq_len, hidden_size = hidden.shape
    x = hidden.astype(np.float64)

    q = (x @ q_w.astype(np.float64).T).reshape(seq_len, num_heads, head_dim)
    k = (x @ k_w.astype(np.float64).T).reshape(seq_len, num_kv_heads, head_dim)
    v = (x @ v_w.astype(np.float64).T).reshape(seq_len, num_kv_heads, head_dim)

    groups = num_heads // num_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    context = np.zeros((seq_len, num_heads, head_dim), dtype=np.float64)

    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    for head in range(num_heads):
        kv = head // groups
        logits = (q[:, head, :] @ k[:, kv, :].T) * scale
        logits = np.where(causal, logits, -np.inf)
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        context[:, head, :] = probs @ v[:, kv, :]

    merged = context.reshape(seq_len, num_heads * head_dim)
    return {
        "output": merged @ o_w.astype(np.float64).T,
        "q": q, "k": k, "v": v,
    }


# --------------------------------------------------------------------------
# Mamba-2 -- sequential recurrence, deliberately not the chunked SSD form
# --------------------------------------------------------------------------

def mamba2_forward(hidden: np.ndarray, weights: dict, cfg: dict) -> dict:
    """NemotronHMamba2Mixer.torch_forward, as a plain sequential scan."""
    seq_len = hidden.shape[0]
    num_heads = cfg["mamba_num_heads"]
    head_dim = cfg["mamba_head_dim"]
    state_size = cfg["ssm_state_size"]
    n_groups = cfg["n_groups"]
    conv_kernel = cfg["conv_kernel"]
    eps = cfg["layer_norm_epsilon"]
    intermediate = num_heads * head_dim
    conv_dim = intermediate + 2 * n_groups * state_size

    x = hidden.astype(np.float64)
    projected = x @ weights["in_proj"].astype(np.float64).T

    # d_mlp is 0 for this config, so the split is [gate, xBC, dt].
    gate = projected[:, :intermediate]
    xbc = projected[:, intermediate:intermediate + conv_dim]
    dt_raw = projected[:, intermediate + conv_dim:]

    # depthwise causal conv1d, kernel 4, padding kernel-1 then truncated
    conv_w = weights["conv1d_weight"].astype(np.float64).reshape(conv_dim, conv_kernel)
    conv_b = weights["conv1d_bias"].astype(np.float64)
    padded = np.concatenate(
        [np.zeros((conv_kernel - 1, conv_dim), dtype=np.float64), xbc], axis=0
    )
    conv_out = np.zeros((seq_len, conv_dim), dtype=np.float64)
    for tap in range(conv_kernel):
        conv_out += padded[tap:tap + seq_len, :] * conv_w[:, tap][None, :]
    conv_out += conv_b[None, :]
    xbc = silu(conv_out)

    x_states = xbc[:, :intermediate].reshape(seq_len, num_heads, head_dim)
    b_states = xbc[:, intermediate:intermediate + n_groups * state_size].reshape(
        seq_len, n_groups, state_size)
    c_states = xbc[:, intermediate + n_groups * state_size:].reshape(
        seq_len, n_groups, state_size)

    dt = softplus(dt_raw + weights["dt_bias"].astype(np.float64)[None, :])
    lo, hi = cfg["time_step_limit"]
    dt = np.clip(dt, lo, hi)

    a_param = -np.exp(weights["A_log"].astype(np.float64))       # [num_heads]
    d_param = weights["D"].astype(np.float64)                    # [num_heads]

    heads_per_group = num_heads // n_groups
    state = np.zeros((num_heads, head_dim, state_size), dtype=np.float64)
    y = np.zeros((seq_len, num_heads, head_dim), dtype=np.float64)

    for t in range(seq_len):
        decay = np.exp(a_param * dt[t])                          # [num_heads]
        for head in range(num_heads):
            group = head // heads_per_group
            state[head] = (
                decay[head] * state[head]
                + (dt[t, head] * x_states[t, head])[:, None] * b_states[t, group][None, :]
            )
            y[t, head] = state[head] @ c_states[t, group] + d_param[head] * x_states[t, head]

    scan_output = y.reshape(seq_len, intermediate)
    normed = gated_rms_norm(scan_output, gate, weights["norm_weight"], eps,
                            intermediate // n_groups)
    return {
        "scan_output": scan_output,
        "output": normed @ weights["out_proj"].astype(np.float64).T,
        "ssm_state": state,
        "dt": dt,
    }
