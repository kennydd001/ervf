from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "rsiv_moe"
sys.path.insert(0, str(SCRIPT_DIR))
import measure_p1c_qwen as p1c  # noqa: E402


def row(rank: int, threshold: float, fast: float, reduction: float):
    return {
        "rank_cap": rank,
        "threshold": threshold,
        "double_gate_fast_fraction": fast,
        "cold_byte_reduction": reduction,
    }


def test_primary_selection_prefers_lower_rank_before_threshold() -> None:
    selected = p1c.select_candidate(
        [row(8, 0.001, 0.99, 20.0), row(4, 0.10, 0.93, 10.1)]
    )
    assert selected["rank_cap"] == 4
    assert selected["selection_kind"] == "primary_gate_pass"


def test_diagnostic_selection_uses_best_bottleneck_with_rank_cap_32() -> None:
    selected = p1c.select_candidate(
        [
            row(4, 0.10, 0.10, 1.1),
            row(32, 0.10, 0.50, 2.0),
            row(64, 0.10, 0.99, 30.0),
        ]
    )
    assert selected["rank_cap"] == 32
    assert selected["selection_kind"] == "diagnostic_validation_failure"


def test_layer_analysis_keeps_future_invocations_aligned(monkeypatch) -> None:
    monkeypatch.setattr(p1c, "CONTEXTS", 1)
    monkeypatch.setattr(p1c, "PREFIX_TOKENS", 4)
    monkeypatch.setattr(p1c, "FUTURE_TOKENS", 2)
    monkeypatch.setattr(p1c, "CONTEXT_TOKENS", 6)
    monkeypatch.setattr(p1c, "EXPERTS", 2)
    monkeypatch.setattr(p1c, "TOP_K", 1)
    monkeypatch.setattr(p1c, "HIDDEN_SIZE", 3)
    monkeypatch.setattr(p1c, "INTERMEDIATE_SIZE", 2)
    monkeypatch.setattr(p1c, "RANKS", (1, 2))
    monkeypatch.setattr(p1c, "GROWTH_CHECKPOINTS", (2, 4))
    data = {
        "x": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        ),
        "ids": torch.tensor([[0], [1], [0], [1], [0], [1]]),
        "weights": torch.ones(6, 1),
        "z": torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[1.0, -1.0]], [[2.0, 1.0]], [[1.0, 2.0]]]
        ),
    }
    residuals, census = p1c.analyze_layer(0, "validation", data)
    assert census["all_required_controls_pass"]
    assert residuals[1]["x"].shape == (2,)
    assert torch.isfinite(residuals[2]["x"]).all()
    assert torch.isfinite(residuals[2]["z"]).all()
    assert census["contexts"][0]["expert_invocations"] == 4
