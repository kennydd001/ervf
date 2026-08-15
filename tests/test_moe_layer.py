import torch

from moe_lab.moe_layer import LoadedMoELayer, ProjectionWeights


def test_expert_forward_matches_explicit_formula() -> None:
    x = torch.tensor([[1.0, -1.0]])
    weights = ProjectionWeights(
        gate=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        up=torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
        down=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    expected = (torch.nn.functional.silu(x @ weights.gate.T) * (x @ weights.up.T)) @ weights.down.T
    assert torch.allclose(LoadedMoELayer.expert_forward(x, weights), expected)
