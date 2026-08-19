from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pro_research"))

# The distributable test pack intentionally does not copy the parent repo's
# common.py or prompt loader. Inject tiny import stubs so the pure Phase-14F
# algorithms can be validated before the pack is installed into the worktree.
import types
_common = types.ModuleType("common")
_common.REPO = REPO
_common.write_json_atomic = lambda *args, **kwargs: None
_common.utc_now = lambda: "selftest"
sys.modules.setdefault("common", _common)
_prompts = types.ModuleType("s100_phase13b_activation_census")
_prompts.prompts = lambda repo: ([], [])
sys.modules.setdefault("s100_phase13b_activation_census", _prompts)
_shared = types.ModuleType("s100_phase14_common")
_shared.RESULTS = REPO / "pro_research" / "results" / "s100_phase14"
_shared.ensure_results = lambda: _shared.RESULTS
sys.modules.setdefault("s100_phase14_common", _shared)

from s100_phase14f_dflash2_economics import (  # noqa: E402
    CONV_GROUP_SIZE, SELECTOR_RANK, draft_parameter_estimate,
)
from s100_phase14f_dflash2_proxy import (  # noqa: E402
    WindowSet,
    apply_correction,
    fit_base,
    fit_correction,
    hidden_metrics,
    prefix_lengths,
    predict_base,
    selector_paths,
)


def test_parameter_estimate() -> dict:
    row = draft_parameter_estimate(
        hidden=2688,
        q_dim=2688,
        kv_dim=512,
        vocab=131072,
        layers=5,
        mlp_ratio=3.75,
    )
    assert row["total_parameters"] > row["base_dflash_parameters"]
    assert row["candidate_selector_parameters"] == (
        2 * 131072 * SELECTOR_RANK + 2688 * SELECTOR_RANK
    )
    assert CONV_GROUP_SIZE == 16
    assert 0.0 < row["dflash2_overhead_fraction_vs_base"] < 0.30
    return row


def synthetic_windows(seed: int, n: int, hidden: int, slots: int, group: int) -> WindowSet:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, hidden)).astype(np.float32)
    groups = hidden // group
    base_mats = [
        (rng.normal(size=(hidden, hidden)).astype(np.float32) / math.sqrt(hidden))
        for _ in range(slots)
    ]
    gate_w = [
        rng.normal(size=(hidden, groups)).astype(np.float32) / math.sqrt(hidden)
        for _ in range(slots)
    ]
    future = np.empty((n, slots, hidden), np.float32)
    previous = x
    for slot in range(slots):
        linear = x @ base_mats[slot]
        alpha = 0.35 * np.tanh(x @ gate_w[slot])
        coupled = (
            previous.reshape(n, groups, group) * alpha[:, :, None]
        ).reshape(n, hidden)
        future[:, slot] = linear + coupled + 0.01 * rng.normal(size=(n, hidden))
        previous = linear  # parallel first-pass predecessor, not corrected output
    labels = np.zeros((n, slots), np.int32)
    anchors = np.zeros(n, np.int32)
    return WindowSet(
        x=x,
        future_hidden=future,
        labels=labels,
        anchor_ids=anchors,
        prompt_index=np.zeros(n, np.int16),
        anchor_index=np.arange(n, dtype=np.int16),
    )


def test_dynamic_correction() -> dict:
    hidden, slots, group = 64, 4, 8
    train = synthetic_windows(1, 320, hidden, slots, group)
    val = synthetic_windows(1, 320, hidden, slots, group)
    # Use a disjoint deterministic slice from the same generating process.
    train_mask = np.arange(320) < 220
    val_mask = ~train_mask
    base_model = fit_base(
        train.x[train_mask], train.future_hidden[train_mask], rank=64
    )
    train_base, train_z = predict_base(base_model, train.x[train_mask])
    correction = fit_correction(
        train.x[train_mask],
        train.future_hidden[train_mask],
        train_base,
        train_z,
        group,
    )
    base, z = predict_base(base_model, val.x[val_mask])
    corrected, _ = apply_correction(correction, val.x[val_mask], base, z)
    base_score = hidden_metrics(val.future_hidden[val_mask], base)
    corrected_score = hidden_metrics(val.future_hidden[val_mask], corrected)
    assert corrected_score["last3_mean_nrmse"] < base_score["last3_mean_nrmse"] * 0.98
    return {
        "base_last3_nrmse": base_score["last3_mean_nrmse"],
        "corrected_last3_nrmse": corrected_score["last3_mean_nrmse"],
    }


def test_selector_walk() -> dict:
    n, slots, top_k, hidden, rank = 32, 3, 3, 4, 1
    candidates = np.tile(np.array([[[0, 1, 2]]], np.int32), (n, slots, 1))
    labels = np.ones((n, slots), np.int32)
    unary = np.zeros((n, slots, top_k), np.float32)
    unary[:, :, 0] = 0.5  # independent unary picks token 0, which is wrong
    anchors = np.full(n, 9, np.int32)
    h = np.ones((n, slots, hidden), np.float32)
    projection = np.ones((hidden, rank), np.float32) * 0.5
    map_a = {
        9: np.array([1.0], np.float32),
        0: np.array([-1.0], np.float32),
        1: np.array([1.0], np.float32),
        2: np.array([0.0], np.float32),
    }
    map_b = {
        0: np.array([-1.0], np.float32),
        1: np.array([1.0], np.float32),
        2: np.array([0.0], np.float32),
    }
    selected = selector_paths(
        candidate_ids=candidates,
        unary_scores=unary,
        anchor_ids=anchors,
        hidden=h,
        embedding_map_a=map_a,
        embedding_map_b=map_b,
        hidden_projection=projection,
        transition_weight=8.0,
    )
    independent = candidates[:, :, 0]
    before = float(np.mean(prefix_lengths(independent, labels)))
    after = float(np.mean(prefix_lengths(selected, labels)))
    assert after > before + 2.5
    return {"independent_prefix": before, "selected_prefix": after}


def main() -> int:
    result = {
        "kind": "s100_phase14f_dflash2_selftest",
        "status": "PASS",
        "parameter_estimate": test_parameter_estimate(),
        "dynamic_correction": test_dynamic_correction(),
        "selector_walk": test_selector_walk(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
