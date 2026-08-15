from pathlib import Path

import pytest
import torch

from moe_lab.fleq_moe.gsq_bridge import (
    expert_forward,
    load_official_quantizer_module,
    relative_l2,
)
from moe_lab.reporting import ROOT


def test_expert_forward_matches_explicit_swiglu() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(5, 8, generator=generator)
    gate = torch.randn(4, 8, generator=generator)
    up = torch.randn(4, 8, generator=generator)
    down = torch.randn(8, 4, generator=generator)
    expected = (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T
    torch.testing.assert_close(expert_forward(x, gate, up, down), expected)


def test_relative_l2_identity_and_known_scale() -> None:
    expected = torch.tensor([1.0, -2.0, 3.0])
    assert relative_l2(expected, expected) == 0.0
    assert relative_l2(expected * 2, expected) == pytest.approx(1.0)


@pytest.mark.parametrize("kind,class_name", [
    ("2bit", "GumbelQuantizer2Bit"),
    ("ternary", "GumbelQuantizerTernary"),
])
def test_pinned_official_quantizers_load(kind: str, class_name: str) -> None:
    module = load_official_quantizer_module(Path(ROOT) / "third_party" / "GSQ", kind)
    assert hasattr(module, class_name)
