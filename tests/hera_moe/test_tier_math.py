import torch


def test_hot_union_is_not_intersection() -> None:
    general = torch.tensor([[True, False, False]])
    code = torch.tensor([[False, True, False]])
    union = torch.stack([general, code]).any(dim=0)
    assert union.tolist() == [[True, True, False]]


def test_cold_calls_are_counted_per_token_across_layers() -> None:
    hot = torch.tensor([[True, False, True], [False, True, True]])
    routes = [torch.tensor([[0, 1], [2, 2]]), torch.tensor([[0, 1], [2, 0]])]
    calls = sum((~hot[layer])[routes[layer]].sum(dim=1) for layer in range(2))
    assert calls.tolist() == [2, 1]
