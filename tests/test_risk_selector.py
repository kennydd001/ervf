import pytest
import torch

from moe_lab.risk_selector import (
    expert_identity_features,
    finite_sample_upper_quantile,
    route_risk_features,
    subset_mask,
    top_j_slate,
)


def test_subset_mask_and_top_j_slate():
    subsets = torch.tensor([[0, 1, 2], [0, 2, 3], [1, 2, 3]])
    mask = subset_mask(subsets, candidates=4)
    assert mask.tolist() == [
        [True, True, True, False],
        [True, False, True, True],
        [False, True, True, True],
    ]
    assert top_j_slate(subsets, top_j=1).tolist() == [True, True, False]
    assert top_j_slate(subsets, top_j=2).tolist() == [True, False, False]


def test_route_risk_features_are_teacher_free_and_route_sensitive():
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    subsets = torch.tensor([[0, 1], [0, 2]])
    features = route_risk_features(weights, subsets, original_k=2)
    assert features.shape == (1, 2, 32)
    assert torch.isfinite(features).all()
    assert not torch.equal(features[:, 0], features[:, 1])


def test_finite_sample_upper_quantile_uses_corrected_rank():
    scores = torch.arange(1, 10, dtype=torch.float32)
    # ceil((9 + 1) * .8) = 8
    assert finite_sample_upper_quantile(scores, alpha=0.2) == 8.0
    # At 5%, the corrected rank clips to n.
    assert finite_sample_upper_quantile(scores, alpha=0.05) == 9.0


def test_route_risk_features_reject_bad_rank():
    with pytest.raises(ValueError, match="exceeds"):
        route_risk_features(torch.ones(1, 4), torch.tensor([[0, 4]]), original_k=2)


def test_expert_identity_features_track_concrete_swaps():
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    ids = torch.tensor([[5, 1, 3, 7]])
    subsets = torch.tensor([[0, 1], [0, 2]])
    features = expert_identity_features(
        weights, ids, subsets, total_experts=8, original_k=2
    )
    assert features.shape == (1, 2, 24)
    assert torch.isfinite(features).all()
    assert not torch.equal(features[:, 0], features[:, 1])
