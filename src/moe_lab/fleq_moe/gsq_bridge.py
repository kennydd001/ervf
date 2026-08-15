from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Literal

import torch


GSQ_COMMIT = "03fc16484c369e3127225615d5e03e8d3a6043e3"
GSQ_FILE_HASHES = {
    "2bit": "81bb1a3ad3b318d4c115a7ba38cd53989d98ea5a3c5943767dac5e6499d7feb0",
    "ternary": "2d4a23117ced5e60359f5432c346187dd38691ab484bfaf725fb7434829cb35e",
}
GSQ_GPTQ_HASH = "b3787929bf0f289c0599e04c70f1b85096f570ab276c32fa3fef51a24c35b4b0"
GSQ_PRIOR_QUANT_HASH = "6a1b3f13ba8974b70b619dd7b0572839b08c4877ee829f2b66d4dd0526614fff"
GSQ_FILENAMES = {
    "2bit": "gumbel_quantizer_2bit.py",
    "ternary": "gumbel_quantizer_ternary.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_quantizer_module(
    gsq_root: Path, kind: Literal["2bit", "ternary"]
) -> ModuleType:
    """Load one pinned upstream quantizer without importing GSQ's full stack."""

    path = gsq_root / "src" / "quantization" / GSQ_FILENAMES[kind]
    actual_hash = sha256_file(path)
    expected_hash = GSQ_FILE_HASHES[kind]
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"unpinned GSQ {kind} quantizer: {actual_hash} != {expected_hash}"
        )
    spec = importlib.util.spec_from_file_location(f"fleq_upstream_gsq_{kind}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GSQ quantizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expert_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(x, gate))
    hidden = hidden * torch.nn.functional.linear(x, up)
    return torch.nn.functional.linear(hidden, down)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()).reshape(-1))
    denominator = torch.linalg.vector_norm(expected.float().reshape(-1)).clamp_min(1e-30)
    return float((numerator / denominator).item())
