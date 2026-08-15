from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from lion_pytorch import Lion

from .gsq_bridge import (
    GSQ_GPTQ_HASH,
    GSQ_PRIOR_QUANT_HASH,
    expert_forward,
    load_official_quantizer_module,
    sha256_file,
)


@dataclass(frozen=True)
class QuantizedProjection:
    weight: torch.Tensor
    scales: torch.Tensor


def select_most_frequent_experts(ids: torch.Tensor, count: int = 8) -> list[int]:
    if ids.ndim != 2:
        raise ValueError("router IDs must be [tokens, top_k]")
    counts = torch.bincount(ids.long().reshape(-1), minlength=128)
    return sorted(range(128), key=lambda expert: (-int(counts[expert]), expert))[:count]


def routed_expert_rows(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    z: torch.Tensor,
    expert: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = (ids.long() == expert).nonzero(as_tuple=False)
    token, slot = positions[:, 0], positions[:, 1]
    return x[token], z[token, slot], weights[token, slot]


def _official_prior(gsq_root: Path):
    gptq_path = gsq_root / "src" / "prior" / "gptq.py"
    quant_path = gsq_root / "src" / "prior" / "quant.py"
    if sha256_file(gptq_path) != GSQ_GPTQ_HASH:
        raise RuntimeError("unpinned GSQ GPTQ implementation")
    if sha256_file(quant_path) != GSQ_PRIOR_QUANT_HASH:
        raise RuntimeError("unpinned GSQ prior quantizer implementation")
    package_name = "fleq_pinned_gsq"
    if package_name not in sys.modules:
        package = __import__("types").ModuleType(package_name)
        package.__path__ = [str((gsq_root / "src").resolve())]
        sys.modules[package_name] = package
    prior_name = f"{package_name}.prior"
    if prior_name not in sys.modules:
        prior = __import__("types").ModuleType(prior_name)
        prior.__path__ = [str((gsq_root / "src" / "prior").resolve())]
        sys.modules[prior_name] = prior

    def load(name: str, path: Path, package_dir: Path | None = None):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[str(package_dir.resolve())]
            if package_dir is not None else None,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load pinned GSQ module {path}")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    load(
        f"{package_name}.quantization",
        gsq_root / "src" / "quantization" / "__init__.py",
        gsq_root / "src" / "quantization",
    )
    load(f"{prior_name}.quant", quant_path)
    module = load(f"{prior_name}.gptq", gptq_path)
    return module.GPTQ, module.Quantizer, module.rtn_quantize


def gsq_config(bits: int | str = 2) -> SimpleNamespace:
    return SimpleNamespace(
        quantization=SimpleNamespace(
            gsq_bits=bits,
            std=0.01,
            strength=6,
            temperature=[2.0, 0.05],
            scale=[100.0, 500.0],
        ),
        gptq=SimpleNamespace(
            wbits=2,
            sym=True,
            trits=bits == "ternary",
            percdamp=0.1,
            blocksize=128,
            groupsize=128,
            static_groups=False,
            prunen=0,
            prunem=0,
        ),
        training=SimpleNamespace(
            lr1=1e-4,
            lr2=5e-5,
            weight_decay=1.0,
            lion_betas=[0.9, 0.95],
        ),
    )


def official_rtn_projection(
    weight: torch.Tensor, gsq_root: Path, bits: int | str = 2
) -> QuantizedProjection:
    _gptq, _quantizer, rtn_quantize = _official_prior(gsq_root)
    layer = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False,
        device=weight.device, dtype=weight.dtype,
    )
    layer.weight.data.copy_(weight)
    q, scales = rtn_quantize(layer, gsq_config(bits), weight.device, weight.dtype)
    del layer
    return QuantizedProjection(q.to(dtype=weight.dtype).detach(), scales.detach())


def official_gptq_projection(
    weight: torch.Tensor,
    calibration_input: torch.Tensor,
    gsq_root: Path,
) -> QuantizedProjection:
    GPTQ, Quantizer, _rtn = _official_prior(gsq_root)
    config = gsq_config(2)
    layer = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False,
        device=weight.device, dtype=weight.dtype,
    )
    layer.weight.data.copy_(weight)
    gptq = GPTQ(layer, "fleq_projection", config, weight.device, weight.dtype)
    gptq.quantizer = Quantizer()
    gptq.quantizer.configure(2, perchannel=True, sym=True, mse=True)
    calibration_input = calibration_input.to(weight.device, dtype=weight.dtype)
    gptq.add_batch(
        calibration_input.unsqueeze(0),
        torch.nn.functional.linear(calibration_input, weight).unsqueeze(0),
    )
    q, scales = gptq.fasterquant(
        None,
        percdamp=0.1,
        blocksize=128,
        groupsize=128,
        static_groups=False,
    )
    gptq.free()
    del layer, gptq
    return QuantizedProjection(q.to(dtype=weight.dtype).detach(), scales.detach())


def _cosine_lr(base: float, step: int, steps: int, minimum_fraction: float = 0.1) -> float:
    progress = step / max(1, steps - 1)
    return base * (
        minimum_fraction
        + 0.5 * (1 - minimum_fraction) * (1 + __import__("math").cos(__import__("math").pi * progress))
    )


def optimize_gsq_expert(
    original: dict[str, torch.Tensor],
    initialized: dict[str, QuantizedProjection],
    calibration_x: torch.Tensor,
    gsq_root: Path,
    *,
    kind: str = "2bit",
    epochs: int = 10,
    batch_size: int = 64,
    seed: int = 260811,
) -> tuple[dict[str, QuantizedProjection], list[float]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    class_name = "GumbelQuantizer2Bit" if kind == "2bit" else "GumbelQuantizerTernary"
    quantizer_module = load_official_quantizer_module(gsq_root, kind)
    quantizer_class = getattr(quantizer_module, class_name)
    quantizers: dict[str, torch.nn.Module] = {}
    for name in ("gate", "up", "down"):
        item = initialized[name]
        quantizers[name] = quantizer_class(
            # The upstream constructor normalizes its initialization tensors
            # in-place.  Clone here so a deterministic repeat really starts
            # from the same immutable caller-owned state.
            item.weight.clone(),
            item.scales.clone(),
            128,
            0.01,
            6,
            item.weight.device,
            item.weight.dtype,
            logits_dtype=torch.bfloat16,
        )
    groups: list[dict[str, Any]] = []
    for quantizer in quantizers.values():
        logits = [p for name, p in quantizer.named_parameters() if name != "scales"]
        groups.extend([
            {"params": logits, "lr": 1e-4, "weight_decay": 1.0, "base_lr": 1e-4},
            {"params": quantizer.scales, "lr": 5e-5, "weight_decay": 0.0, "base_lr": 5e-5},
        ])
    optimizer = Lion(groups, betas=(0.9, 0.95))
    calibration_x = calibration_x.to(original["gate"].device, dtype=original["gate"].dtype)
    with torch.no_grad():
        reference = expert_forward(
            calibration_x, original["gate"], original["up"], original["down"]
        )
    full_batches = calibration_x.shape[0] // batch_size
    if full_batches < 1:
        raise ValueError(f"GSQ needs at least one full batch of {batch_size} rows")
    steps = epochs * full_batches
    losses: list[float] = []
    step = 0
    for _epoch in range(epochs):
        # Match upstream's get_random_batch_indices: permute contiguous full
        # batches and intentionally omit the remainder.
        for batch_index in torch.randperm(full_batches).tolist():
            start = batch_index * batch_size
            stop = start + batch_size
            temperature = 2.0 + (0.05 - 2.0) * step / max(1, steps - 1)
            logit_scale = 100.0 + (500.0 - 100.0) * step / max(1, steps - 1)
            for group in optimizer.param_groups:
                group["lr"] = _cosine_lr(group["base_lr"], step, steps)
            optimizer.zero_grad(set_to_none=True)
            soft = {
                name: quantizer(temperature, logit_scale)
                for name, quantizer in quantizers.items()
            }
            prediction = expert_forward(
                calibration_x[start:stop], soft["gate"], soft["up"], soft["down"]
            )
            loss = torch.nn.functional.mse_loss(
                prediction.float(), reference[start:stop].float()
            )
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for quantizer in quantizers.values()
                for parameter in quantizer.parameters()
            ):
                raise FloatingPointError("non-finite GSQ gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
    output = {}
    for name, quantizer in quantizers.items():
        weight, scales = quantizer.get_hard_weights()
        output[name] = QuantizedProjection(weight.detach(), scales.detach())
    del optimizer, quantizers, reference, calibration_x
    return output, losses


def hard_gsq_initialization(
    initialized: dict[str, QuantizedProjection],
    gsq_root: Path,
    *,
    kind: str,
    seed: int,
) -> dict[str, QuantizedProjection]:
    """Project an upstream initializer onto the quantizer's actual hard grid."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    class_name = "GumbelQuantizer2Bit" if kind == "2bit" else "GumbelQuantizerTernary"
    module = load_official_quantizer_module(gsq_root, kind)
    quantizer_class = getattr(module, class_name)
    output = {}
    for name, item in initialized.items():
        quantizer = quantizer_class(
            item.weight.clone(), item.scales.clone(), 128, 0.01, 6,
            item.weight.device, item.weight.dtype, logits_dtype=torch.bfloat16,
        )
        weight, scales = quantizer.get_hard_weights()
        output[name] = QuantizedProjection(weight.detach(), scales.detach())
    return output


def output_metrics(
    x: torch.Tensor,
    router_weights: torch.Tensor,
    original: dict[str, torch.Tensor],
    candidate: dict[str, QuantizedProjection | torch.Tensor],
) -> dict[str, float]:
    device = original["gate"].device
    x = x.to(device=device, dtype=original["gate"].dtype)
    router_weights = router_weights.to(device=device, dtype=torch.float32)

    def tensor(name: str) -> torch.Tensor:
        value = candidate[name]
        weight = value.weight if isinstance(value, QuantizedProjection) else value
        return weight.to(device=device, dtype=original["gate"].dtype)

    with torch.no_grad():
        reference = expert_forward(x, original["gate"], original["up"], original["down"])
        prediction = expert_forward(x, tensor("gate"), tensor("up"), tensor("down"))
    difference = prediction.float() - reference.float()
    row_error = torch.linalg.vector_norm(difference, dim=1)
    row_reference = torch.linalg.vector_norm(reference.float(), dim=1).clamp_min(1e-30)
    relative_rows = row_error / row_reference
    weighted_error = (router_weights * difference.square().sum(1)).sum()
    weighted_reference = (router_weights * reference.float().square().sum(1)).sum().clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(prediction.float(), reference.float(), dim=1)
    return {
        "rows": int(x.shape[0]),
        "relative_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)),
        "relative_row_p50": float(torch.quantile(relative_rows, 0.50)),
        "relative_row_p95": float(torch.quantile(relative_rows, 0.95)),
        "router_weighted_relative_mse": float(weighted_error / weighted_reference),
        "cosine_mean": float(cosine.mean()),
        "all_finite": bool(torch.isfinite(prediction).all()),
    }
