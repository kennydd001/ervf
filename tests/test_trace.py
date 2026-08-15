from pathlib import Path

import pytest
import torch

from moe_lab.trace import MoETrace, load_trace, save_trace, slice_trace, trace_baselines


def make_trace() -> MoETrace:
    selected = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 2.0], [4.0, 0.0]],
        ]
    )
    weights = torch.tensor([[0.25, 0.50], [0.40, 0.10]])
    routed = (selected * weights.unsqueeze(-1)).sum(dim=1)
    return MoETrace(
        hidden_states=torch.ones(2, 2),
        router_ids=torch.tensor([[1, 3], [2, 4]], dtype=torch.int16),
        router_weights=weights,
        selected_expert_outputs=selected,
        routed_output=routed,
        shared_output=torch.zeros(2, 2),
    )


def test_trace_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "trace.safetensors"
    validation = save_trace(make_trace(), path, {"layer": 1})
    loaded = load_trace(path)
    assert validation["tokens"] == 2
    assert torch.equal(loaded.routed_output, make_trace().routed_output)


def test_slice_trace_preserves_schema() -> None:
    sliced = slice_trace(make_trace(), 1)
    assert sliced.hidden_states.shape[0] == 1
    sliced.validate()


def test_trace_rejects_incorrect_aggregate() -> None:
    trace = make_trace()
    trace.routed_output += 10
    with pytest.raises(ValueError, match="weighted expert sum"):
        trace.validate()


def test_top1_baselines_respect_non_normalized_router_weights() -> None:
    trace = make_trace()
    baselines = trace_baselines(trace)
    assert torch.equal(baselines["top1_unrenormalized"][0], torch.tensor([0.0, 1.0]))
    assert torch.equal(baselines["top1_renormalized"][0], torch.tensor([0.0, 2.0]))
