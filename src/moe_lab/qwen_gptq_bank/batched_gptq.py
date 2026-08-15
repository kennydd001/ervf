from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from moe_lab.fleq_moe.expert_quant import _official_prior


@dataclass(frozen=True)
class BatchedQuantizedProjection:
    weight: torch.Tensor
    scales: torch.Tensor


def official_pure_gptq_projection(
    weight: torch.Tensor,
    calibration_input: torch.Tensor,
    gsq_root: Path,
) -> BatchedQuantizedProjection:
    """Pinned upstream GPTQ with a name that cannot enter the GSQ name branch."""
    GPTQ, Quantizer, _rtn = _official_prior(gsq_root)
    from moe_lab.fleq_moe.expert_quant import gsq_config

    name = "expert_projection"
    if any(marker in name for marker in ("q_proj", "k_proj", "in_proj_qkv")):
        raise RuntimeError("pure GPTQ reference name unexpectedly activates a GSQ branch")
    layer = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False,
        device=weight.device, dtype=weight.dtype,
    )
    layer.weight.data.copy_(weight)
    gptq = GPTQ(layer, name, gsq_config(2), weight.device, weight.dtype)
    gptq.quantizer = Quantizer()
    gptq.quantizer.configure(2, perchannel=True, sym=True, mse=True)
    calibration_input = calibration_input.to(weight.device, dtype=weight.dtype)
    gptq.add_batch(
        calibration_input.unsqueeze(0),
        torch.nn.functional.linear(calibration_input, weight).unsqueeze(0),
    )
    quantized, scales = gptq.fasterquant(
        None, percdamp=0.1, blocksize=128, groupsize=128,
        static_groups=False,
    )
    gptq.free()
    del layer, gptq
    return BatchedQuantizedProjection(
        quantized.to(dtype=weight.dtype).detach(),
        scales.to(dtype=weight.dtype).detach(),
    )


def _official_hinv(calibration: torch.Tensor, dtype: torch.dtype, percdamp: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the pinned GPTQ add_batch/cholesky path for one expert."""
    columns = calibration.shape[-1]
    inp = (math.sqrt(2.0) * calibration.float()).t()
    hessian = inp.matmul(inp.t())
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    damp = percdamp * torch.mean(torch.diag(hessian))
    diagonal = torch.arange(columns, device=calibration.device)
    hessian[diagonal, diagonal] += damp
    factor = torch.linalg.cholesky(hessian).to(dtype)
    inverse = torch.cholesky_inverse(factor.float())
    return torch.linalg.cholesky(inverse, upper=True), dead


def _fastgrid_find_params(
    values: torch.Tensor, maxq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorize only the pinned symmetric 2-bit MSE candidate grid."""
    if values.ndim != 2:
        raise ValueError("weight group must be [rows, columns]")
    rows = values.shape[0]
    zero_floor = torch.zeros(rows, device=values.device)
    xmin = torch.minimum(values.min(1)[0], zero_floor)
    xmax = torch.maximum(values.max(1)[0], zero_floor)
    xmax = torch.maximum(torch.abs(xmin), xmax)
    negative = xmin < 0
    xmin[negative] = -xmax[negative]
    degenerate = (xmin == 0) & (xmax == 0)
    xmin[degenerate] = -1
    xmax[degenerate] = 1
    zero = torch.full_like(xmax, (maxq + 1) / 2)

    shrink = torch.tensor(
        [1 - index / 100 for index in range(80)],
        device=values.device, dtype=values.dtype,
    )
    xmin1 = shrink[:, None] * xmin[None, :]
    xmax1 = shrink[:, None] * xmax[None, :]
    candidate_scale = (
        (xmax1 - xmin1) / maxq
    )
    expanded = values.unsqueeze(0)
    scale = candidate_scale.unsqueeze(2)
    expanded_zero = zero.view(1, rows, 1)
    positive_q = torch.clamp(torch.round(expanded / scale) + expanded_zero, 0, 3)
    positive_q = scale * (positive_q - expanded_zero)
    negative_scale = -scale
    negative_q = torch.clamp(torch.round(expanded / negative_scale) + expanded_zero, 0, 3)
    negative_q = negative_scale * (negative_q - expanded_zero)
    positive_error = (positive_q - expanded).abs().pow(2.4).sum(2)
    negative_error = (negative_q - expanded).abs().pow(2.4).sum(2)
    use_negative = negative_error < positive_error
    errors = torch.where(use_negative, negative_error, positive_error)
    signed_scales = torch.where(use_negative, -candidate_scale, candidate_scale)
    best = errors.argmin(0)
    selected = signed_scales.gather(0, best.unsqueeze(0)).squeeze(0)
    return selected.unsqueeze(1), zero.unsqueeze(1)


def _nosync_find_params(
    values: torch.Tensor, maxq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinned grid/order/shapes, replacing only host-sync guards with where."""
    rows = values.shape[0]
    zero_floor = torch.zeros(rows, device=values.device)
    xmin = torch.minimum(values.min(1)[0], zero_floor)
    xmax = torch.maximum(values.max(1)[0], zero_floor)
    xmax = torch.maximum(torch.abs(xmin), xmax)
    selected = xmin < 0
    xmin[selected] = -xmax[selected]
    selected = (xmin == 0) & (xmax == 0)
    xmin[selected] = -1
    xmax[selected] = 1
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2)
    best = torch.full((rows,), float("inf"), device=values.device)
    for index in range(80):
        shrink = 1 - index / 100
        xmin1 = shrink * xmin
        xmax1 = shrink * xmax
        scale1 = (xmax1 - xmin1) / maxq
        q_pos = torch.clamp(torch.round(values / scale1.unsqueeze(1)) + zero.unsqueeze(1), 0, 3)
        q_pos = scale1.unsqueeze(1) * (q_pos - zero.unsqueeze(1))
        q_neg = torch.clamp(torch.round(values / (-scale1).unsqueeze(1)) + zero.unsqueeze(1), 0, 3)
        q_neg = (-scale1).unsqueeze(1) * (q_neg - zero.unsqueeze(1))
        e_pos = (q_pos - values).abs().pow(2.4).sum(1)
        e_neg = (q_neg - values).abs().pow(2.4).sum(1)
        use_neg = e_neg < e_pos
        error = torch.where(use_neg, e_neg, e_pos)
        chosen = torch.where(use_neg, -scale1, scale1)
        improve = error < best
        best[improve] = error[improve]
        scale[improve] = chosen[improve]
        zero[improve] = zero[improve]
    return scale.unsqueeze(1), zero.unsqueeze(1)


def _where_find_params(
    values: torch.Tensor, maxq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = values.shape[0]
    floor = torch.zeros(rows, device=values.device)
    xmin = torch.minimum(values.min(1)[0], floor)
    xmax = torch.maximum(values.max(1)[0], floor)
    xmax = torch.maximum(torch.abs(xmin), xmax)
    xmin = torch.where(xmin < 0, -xmax, xmin)
    degenerate = (xmin == 0) & (xmax == 0)
    xmin = torch.where(degenerate, torch.full_like(xmin, -1), xmin)
    xmax = torch.where(degenerate, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2)
    best = torch.full((rows,), float("inf"), device=values.device)
    for index in range(80):
        shrink = 1 - index / 100
        xmin1 = shrink * xmin; xmax1 = shrink * xmax
        scale1 = (xmax1 - xmin1) / maxq
        q_pos = torch.clamp(torch.round(values / scale1.unsqueeze(1)) + zero.unsqueeze(1), 0, maxq)
        q_pos = scale1.unsqueeze(1) * (q_pos - zero.unsqueeze(1))
        q_neg = torch.clamp(torch.round(values / (-scale1).unsqueeze(1)) + zero.unsqueeze(1), 0, maxq)
        q_neg = (-scale1).unsqueeze(1) * (q_neg - zero.unsqueeze(1))
        e_pos = (q_pos - values).abs().pow(2.4).sum(1)
        e_neg = (q_neg - values).abs().pow(2.4).sum(1)
        use_neg = e_neg < e_pos
        error = torch.where(use_neg, e_neg, e_pos)
        chosen = torch.where(use_neg, -scale1, scale1)
        improve = error < best
        best = torch.where(improve, error, best)
        scale = torch.where(improve, chosen, scale)
    return scale.unsqueeze(1), zero.unsqueeze(1)


def _finder_pure_gptq_projection(
    weight: torch.Tensor,
    calibration_input: torch.Tensor,
    gsq_root: Path,
    finder,
    *,
    percdamp: float = 0.1,
    blocksize: int = 128,
    groupsize: int = 128,
) -> BatchedQuantizedProjection:
    """Pure GPTQ with identical grid operations but no torch.any host syncs."""
    rows, columns = weight.shape
    _gptq, Quantizer, _rtn = _official_prior(gsq_root)
    probe = Quantizer(); probe.configure(2, perchannel=True, sym=True, mse=True)
    quantize = __import__(probe.__class__.__module__, fromlist=["quantize"]).quantize
    maxq = probe.maxq.to(weight.device)
    hinv, dead = _official_hinv(calibration_input, weight.dtype, percdamp)
    work = weight.float().clone(); work[:, dead] = 0
    groups = (columns + groupsize - 1) // groupsize
    group_scales = torch.zeros((rows, groups), device=weight.device, dtype=torch.float32)
    current_scale = current_zero = None
    for block_start in range(0, columns, blocksize):
        block_end = min(block_start + blocksize, columns); count = block_end - block_start
        block = work[:, block_start:block_end].clone()
        quantized = torch.zeros_like(block); errors = torch.zeros_like(block)
        block_hinv = hinv[block_start:block_end, block_start:block_end]
        for column_in_block in range(count):
            absolute_column = block_start + column_in_block
            if absolute_column % groupsize == 0:
                group_end = min(absolute_column + groupsize, columns)
                current_scale, current_zero = finder(
                    work[:, absolute_column:group_end], maxq
                )
                group_scales[:, absolute_column // groupsize] = current_scale.view(rows)
            value = block[:, column_in_block]; diagonal = block_hinv[column_in_block, column_in_block]
            q = quantize(value.unsqueeze(1), current_scale, current_zero, maxq).flatten()
            quantized[:, column_in_block] = q
            error = (value - q) / diagonal
            block[:, column_in_block:] -= error.unsqueeze(1).matmul(block_hinv[column_in_block, column_in_block:].unsqueeze(0))
            errors[:, column_in_block] = error
        work[:, block_start:block_end] = quantized
        if block_end < columns:
            work[:, block_end:] -= errors.matmul(hinv[block_start:block_end, block_end:])
    return BatchedQuantizedProjection(work.to(weight.dtype).detach(), group_scales.to(weight.dtype).detach())


def nosync_pure_gptq_projection(
    weight: torch.Tensor, calibration_input: torch.Tensor, gsq_root: Path,
    *, percdamp: float = 0.1, blocksize: int = 128, groupsize: int = 128,
) -> BatchedQuantizedProjection:
    return _finder_pure_gptq_projection(
        weight, calibration_input, gsq_root, _nosync_find_params,
        percdamp=percdamp, blocksize=blocksize, groupsize=groupsize,
    )


def where_pure_gptq_projection(
    weight: torch.Tensor, calibration_input: torch.Tensor, gsq_root: Path,
    *, percdamp: float = 0.1, blocksize: int = 128, groupsize: int = 128,
) -> BatchedQuantizedProjection:
    return _finder_pure_gptq_projection(
        weight, calibration_input, gsq_root, _where_find_params,
        percdamp=percdamp, blocksize=blocksize, groupsize=groupsize,
    )


def fastgrid_pure_gptq_projection(
    weight: torch.Tensor,
    calibration_input: torch.Tensor,
    gsq_root: Path,
    *,
    percdamp: float = 0.1,
    blocksize: int = 128,
    groupsize: int = 128,
) -> BatchedQuantizedProjection:
    """Pure pinned GPTQ with only the 80-candidate MSE grid vectorized."""
    if weight.ndim != 2 or calibration_input.ndim != 2:
        raise ValueError("weight and calibration must be matrices")
    rows, columns = weight.shape
    if calibration_input.shape[1] != columns:
        raise ValueError("calibration width mismatch")
    _gptq, Quantizer, _rtn = _official_prior(gsq_root)
    probe = Quantizer()
    probe.configure(2, perchannel=True, sym=True, mse=True)
    quantize = __import__(probe.__class__.__module__, fromlist=["quantize"]).quantize
    maxq = probe.maxq.to(weight.device)
    hinv, dead = _official_hinv(calibration_input, weight.dtype, percdamp)
    work = weight.float().clone()
    work[:, dead] = 0
    groups = (columns + groupsize - 1) // groupsize
    group_scales = torch.zeros((rows, groups), device=weight.device, dtype=torch.float32)
    current_scale = current_zero = None
    for block_start in range(0, columns, blocksize):
        block_end = min(block_start + blocksize, columns)
        count = block_end - block_start
        block = work[:, block_start:block_end].clone()
        quantized = torch.zeros_like(block)
        errors = torch.zeros_like(block)
        block_hinv = hinv[block_start:block_end, block_start:block_end]
        for column_in_block in range(count):
            absolute_column = block_start + column_in_block
            if absolute_column % groupsize == 0:
                group_end = min(absolute_column + groupsize, columns)
                current_scale, current_zero = _fastgrid_find_params(
                    work[:, absolute_column:group_end], maxq
                )
                group_scales[:, absolute_column // groupsize] = current_scale.view(rows)
            value = block[:, column_in_block]
            diagonal = block_hinv[column_in_block, column_in_block]
            q = quantize(
                value.unsqueeze(1), current_scale, current_zero, maxq
            ).flatten()
            quantized[:, column_in_block] = q
            error = (value - q) / diagonal
            block[:, column_in_block:] -= error.unsqueeze(1).matmul(
                block_hinv[column_in_block, column_in_block:].unsqueeze(0)
            )
            errors[:, column_in_block] = error
        work[:, block_start:block_end] = quantized
        if block_end < columns:
            work[:, block_end:] -= errors.matmul(hinv[block_start:block_end, block_end:])
    return BatchedQuantizedProjection(
        weight=work.to(weight.dtype).detach(),
        scales=group_scales.to(weight.dtype).detach(),
    )


def batched_official_gptq_projection(
    weights: torch.Tensor,
    calibration_inputs: torch.Tensor,
    gsq_root: Path,
    *,
    percdamp: float = 0.1,
    blocksize: int = 128,
    groupsize: int = 128,
) -> BatchedQuantizedProjection:
    """Batch independent experts while preserving the pinned GPTQ equations.

    Hessians and their factorizations intentionally run expert-by-expert. The
    column loop is batched; no calibration statistics are shared across experts.
    """
    if weights.ndim != 3 or calibration_inputs.ndim != 3:
        raise ValueError("weights and calibration inputs must be [experts, rows/samples, columns]")
    experts, rows, columns = weights.shape
    if calibration_inputs.shape[0] != experts or calibration_inputs.shape[2] != columns:
        raise ValueError("calibration inputs do not match batched weights")
    if groupsize <= 0 or blocksize <= 0:
        raise ValueError("positive group and block sizes are required")
    if not torch.isfinite(weights).all() or not torch.isfinite(calibration_inputs).all():
        raise ValueError("non-finite GPTQ input")

    _gptq, Quantizer, _rtn = _official_prior(gsq_root)
    quantizers = []
    for _expert in range(experts):
        quantizer = Quantizer()
        quantizer.configure(2, perchannel=True, sym=True, mse=True)
        quantizers.append(quantizer)

    hinv_parts, dead_parts = [], []
    for expert in range(experts):
        hinv, dead = _official_hinv(calibration_inputs[expert], weights.dtype, percdamp)
        hinv_parts.append(hinv)
        dead_parts.append(dead)
    hinv = torch.stack(hinv_parts)
    dead = torch.stack(dead_parts)
    work = weights.float().clone()
    work.masked_fill_(dead.unsqueeze(1), 0)

    groups = (columns + groupsize - 1) // groupsize
    group_scales = torch.zeros(
        (experts, rows, groups), device=weights.device, dtype=torch.float32
    )
    group_zeros = torch.zeros_like(group_scales)

    for block_start in range(0, columns, blocksize):
        block_end = min(block_start + blocksize, columns)
        count = block_end - block_start
        block = work[:, :, block_start:block_end].clone()
        quantized = torch.zeros_like(block)
        errors = torch.zeros_like(block)
        block_hinv = hinv[:, block_start:block_end, block_start:block_end]

        for column_in_block in range(count):
            absolute_column = block_start + column_in_block
            if absolute_column % groupsize == 0:
                group_end = min(absolute_column + groupsize, columns)
                group = absolute_column // groupsize
                for expert, quantizer in enumerate(quantizers):
                    quantizer.find_params(
                        work[expert, :, absolute_column:group_end], weight=True
                    )
                    group_scales[expert, :, group] = quantizer.scale.view(rows)
                    group_zeros[expert, :, group] = quantizer.zero.view(rows)

            value = block[:, :, column_in_block]
            diagonal = block_hinv[:, column_in_block, column_in_block]
            q_parts = []
            for expert, quantizer in enumerate(quantizers):
                q_parts.append(
                    __import__(quantizer.__class__.__module__, fromlist=["quantize"])
                    .quantize(
                        value[expert].unsqueeze(1),
                        quantizer.scale,
                        quantizer.zero,
                        quantizer.maxq,
                    )
                    .flatten()
                )
            q = torch.stack(q_parts)
            quantized[:, :, column_in_block] = q
            error = (value - q) / diagonal.unsqueeze(1)
            # Keep the pinned routine's exact 2-D matmul call per expert.
            # A broadcasted outer product is algebraically identical but can
            # change floating-point rounding and therefore later hard codes.
            for expert in range(experts):
                block[expert, :, column_in_block:] -= error[expert].unsqueeze(1).matmul(
                    block_hinv[expert, column_in_block, column_in_block:].unsqueeze(0)
                )
            errors[:, :, column_in_block] = error

        work[:, :, block_start:block_end] = quantized
        if block_end < columns:
            for expert in range(experts):
                work[expert, :, block_end:] -= errors[expert].matmul(
                    hinv[expert, block_start:block_end, block_end:]
                )

    return BatchedQuantizedProjection(
        weight=work.to(weights.dtype).detach(),
        scales=group_scales.to(weights.dtype).detach(),
    )


def codes_from_quantized(
    quantized_weight: torch.Tensor, scales: torch.Tensor, groupsize: int = 128
) -> torch.Tensor:
    if quantized_weight.ndim != 3 or scales.ndim != 3:
        raise ValueError("batched quantized weights and scales must both be rank three")
    columns = quantized_weight.shape[-1]
    groups = torch.arange(columns, device=quantized_weight.device) // groupsize
    expanded_scales = scales.index_select(-1, groups)
    codes = torch.round(quantized_weight.float() / expanded_scales.float()).to(torch.int8)
    if not bool(((codes >= -2) & (codes <= 1)).all()):
        raise ValueError("recovered GPTQ code outside {-2,-1,0,1}")
    return codes


def pack_2bit_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 4:
        raise ValueError("last code dimension must be divisible by four")
    if not bool(((codes >= -2) & (codes <= 1)).all()):
        raise ValueError("code outside {-2,-1,0,1}")
    values = (codes.to(torch.int16) + 2).to(torch.uint8).reshape(*codes.shape[:-1], -1, 4)
    return (
        values[..., 0]
        | (values[..., 1] << 2)
        | (values[..., 2] << 4)
        | (values[..., 3] << 6)
    ).contiguous()


def unpack_2bit_codes(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise ValueError("packed codes must be uint8")
    shifts = torch.tensor((0, 2, 4, 6), dtype=torch.uint8, device=packed.device)
    values = ((packed.unsqueeze(-1) >> shifts) & 0x03).reshape(*packed.shape[:-1], -1)
    return (values.to(torch.int8) - 2).contiguous()
