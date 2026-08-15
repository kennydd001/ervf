import torch

from moe_lab.behavioral import (
    rmsnorm,
    rmsnorm_pullback,
    sample_fisher_score_gradient_replicates,
)


def test_rmsnorm_pullback_matches_autograd() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, dtype=torch.float64)
    output_gradient = torch.randn(3, 5, dtype=torch.float64)
    output = rmsnorm(hidden, weight)
    expected = torch.autograd.grad((output * output_gradient).sum(), hidden)[0]
    actual = rmsnorm_pullback(hidden.detach(), output_gradient, weight)
    assert torch.allclose(actual.double(), expected, atol=1e-6, rtol=1e-5)


def test_fisher_replicates_have_expected_shape_and_are_reproducible() -> None:
    torch.manual_seed(11)
    hidden = torch.randn(4, 6)
    weight = torch.randn(6)
    lm_head = torch.randn(17, 6)
    first = sample_fisher_score_gradient_replicates(
        hidden, weight, lm_head, batch_size=2, seed=31, samples_per_state=3
    )
    second = sample_fisher_score_gradient_replicates(
        hidden, weight, lm_head, batch_size=2, seed=31, samples_per_state=3
    )
    assert first.shape == (3, 4, 6)
    assert torch.equal(first, second)


def test_fisher_replicates_reject_nonpositive_sample_count() -> None:
    hidden = torch.randn(2, 3)
    weight = torch.ones(3)
    lm_head = torch.randn(5, 3)
    try:
        sample_fisher_score_gradient_replicates(
            hidden, weight, lm_head, batch_size=2, seed=1, samples_per_state=0
        )
    except ValueError as error:
        assert "samples_per_state" in str(error)
    else:
        raise AssertionError("expected ValueError")
