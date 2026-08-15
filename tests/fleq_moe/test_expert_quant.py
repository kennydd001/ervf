import torch

from moe_lab.fleq_moe.expert_quant import (
    QuantizedProjection,
    output_metrics,
    routed_expert_rows,
    select_most_frequent_experts,
)


def test_select_most_frequent_experts_is_count_then_id() -> None:
    ids = torch.tensor([[5, 2, 5], [2, 3, 5], [3, 2, 9]])
    assert select_most_frequent_experts(ids, 4) == [2, 5, 3, 9]


def test_routed_expert_rows_preserves_token_slot_order() -> None:
    x = torch.tensor([[10.0], [20.0]])
    ids = torch.tensor([[1, 2], [2, 1]])
    weights = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    z = torch.tensor([[[11.0], [12.0]], [[21.0], [22.0]]])
    ex, ez, ew = routed_expert_rows(x, ids, weights, z, 1)
    torch.testing.assert_close(ex, torch.tensor([[10.0], [20.0]]))
    torch.testing.assert_close(ez, torch.tensor([[11.0], [22.0]]))
    torch.testing.assert_close(ew, torch.tensor([0.8, 0.7]))


def test_output_metrics_identity() -> None:
    generator = torch.Generator().manual_seed(1)
    original = {
        "gate": torch.randn(3, 4, generator=generator),
        "up": torch.randn(3, 4, generator=generator),
        "down": torch.randn(4, 3, generator=generator),
    }
    candidate = {
        key: QuantizedProjection(value.clone(), torch.ones(1))
        for key, value in original.items()
    }
    metrics = output_metrics(torch.randn(5, 4, generator=generator), torch.ones(5), original, candidate)
    assert metrics["relative_l2"] == 0.0
    assert metrics["router_weighted_relative_mse"] == 0.0
    assert metrics["cosine_mean"] > 0.999999


def test_output_metrics_casts_dequantized_candidate_to_model_dtype() -> None:
    generator = torch.Generator().manual_seed(2)
    original = {
        "gate": torch.randn(3, 4, generator=generator, dtype=torch.bfloat16),
        "up": torch.randn(3, 4, generator=generator, dtype=torch.bfloat16),
        "down": torch.randn(4, 3, generator=generator, dtype=torch.bfloat16),
    }
    candidate = {
        key: QuantizedProjection(value.float(), torch.ones(1))
        for key, value in original.items()
    }
    metrics = output_metrics(
        torch.randn(5, 4, generator=generator, dtype=torch.bfloat16),
        torch.ones(5),
        original,
        candidate,
    )
    assert metrics["relative_l2"] == 0.0
