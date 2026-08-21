from __future__ import annotations

import pytest

from moe_lab.ornith.rolling_prefetch import (
    EXPERT_BYTES,
    RollingPrefetchController,
)


def _routes(*, layers: int = 3, offset: int = 0):
    return tuple(
        tuple(
            tuple((offset + layer * 17 + token * 5 + route) % 64 for route in range(8))
            for token in range(4)
        )
        for layer in range(layers)
    )


def test_prepare_is_speculative_and_ring_is_continuous_across_h4():
    controller = RollingPrefetchController(
        layers=3, experts=64, cache_slots=16, ring_depth=2
    )
    controller.reset_request("req-a")
    empty = controller.cache_snapshot()
    first = controller.prepare_block(_routes())
    second = controller.prepare_block(_routes(offset=1))

    assert controller.cache_snapshot() == empty
    assert [task.task_index for task in first.tasks] == [0, 1, 2]
    assert [task.task_index for task in second.tasks] == [3, 4, 5]
    assert [task.ring_slot for task in first.tasks + second.tasks] == [0, 1, 0, 1, 0, 1]
    assert all(task.bytes == task.groups * EXPERT_BYTES for task in first.tasks)


def test_exact_full_commit_keeps_queued_successor_valid():
    controller = RollingPrefetchController(layers=3, experts=64, cache_slots=16)
    controller.reset_request("req-a")
    routes = _routes()
    first = controller.prepare_block(routes)
    second = controller.prepare_block(_routes(offset=1))
    result = controller.adjudicate(first.block_id, routes)

    assert result.route_accuracy == 1.0
    assert result.uncovered_misses == 0
    assert result.false_prefetches == 0
    assert not result.requires_copy_barrier
    assert controller.pending_block_ids == (second.block_id,)
    assert controller.cache_snapshot() == first.cache_after


def test_wrong_prediction_never_hides_authoritative_cold_experts():
    controller = RollingPrefetchController(layers=3, experts=64, cache_slots=16)
    controller.reset_request("req-a")
    predicted = _routes()
    actual = _routes(offset=32)
    block = controller.prepare_block(predicted)
    plan = controller.plan_layer(block.block_id, 0, actual[0])

    assert plan.staged_hits == ()
    assert plan.uncovered_experts
    assert plan.combined_plan.unique_miss_experts == len(plan.uncovered_experts)
    assert plan.combined_plan.hot_assignments + plan.combined_plan.miss_assignments == 32


def test_start_resident_expert_reloaded_inside_h4_is_a_true_miss():
    controller = RollingPrefetchController(layers=1, experts=64, cache_slots=8)
    controller.reset_request("req-a")
    warm = ((tuple(range(8)),) * 4,)
    warm_block = controller.prepare_block(warm)
    controller.adjudicate(warm_block.block_id, warm)
    actual = ((
        tuple(range(8)),
        tuple(range(8, 16)),
        tuple(range(16, 24)),
        tuple(range(8)),
    ),)
    predicted = ((tuple(range(8, 16)),) * 4,)
    block = controller.prepare_block(predicted)
    plan = controller.plan_layer(block.block_id, 0, actual[0])

    assert set(range(8)).issubset(plan.uncovered_experts)
    assert plan.resident_plan.unique_miss_experts == 24


def test_mismatch_invalidates_speculative_tail_and_requires_barrier():
    controller = RollingPrefetchController(layers=3, experts=64, cache_slots=16)
    controller.reset_request("req-a")
    first = controller.prepare_block(_routes())
    second = controller.prepare_block(_routes(offset=1))
    epoch = controller.schedule_epoch
    result = controller.adjudicate(first.block_id, _routes(offset=2))

    assert result.invalidated_block_ids == (second.block_id,)
    assert result.requires_copy_barrier
    assert controller.pending_block_ids == ()
    assert controller.schedule_epoch == epoch + 1
    with pytest.raises(KeyError):
        controller.plan_layer(second.block_id, 0, _routes(offset=1)[0])


def test_partial_commit_matches_independent_prefix_replay():
    routes = _routes()
    partial = RollingPrefetchController(layers=3, experts=64, cache_slots=16)
    partial.reset_request("partial")
    block = partial.prepare_block(routes)
    result = partial.adjudicate(block.block_id, routes, committed_positions=2)

    reference = RollingPrefetchController(layers=3, experts=64, cache_slots=16)
    reference.reset_request("reference")
    prefix_then_padding = tuple(
        layer[:2] + layer[:2] for layer in routes
    )
    reference_block = reference.prepare_block(prefix_then_padding)
    reference.adjudicate(
        reference_block.block_id, prefix_then_padding, committed_positions=2
    )

    assert partial.cache_snapshot() == reference.cache_snapshot()
    assert result.committed_positions == 2


def test_reset_invalidates_old_blocks_and_clears_cache():
    controller = RollingPrefetchController(layers=3, experts=64, cache_slots=16)
    controller.reset_request("req-a")
    block = controller.prepare_block(_routes())
    controller.adjudicate(block.block_id, _routes())
    assert any(snapshot.expert_to_slot for snapshot in controller.cache_snapshot())

    old_generation = controller.request_generation
    controller.reset_request("req-b")
    assert controller.request_generation == old_generation + 1
    assert controller.pending_block_ids == ()
    assert not any(snapshot.expert_to_slot for snapshot in controller.cache_snapshot())
    with pytest.raises(KeyError):
        controller.plan_layer(block.block_id, 0, _routes()[0])


def test_abort_commits_nothing_and_invalidates_tail():
    controller = RollingPrefetchController(layers=3, experts=64, cache_slots=16)
    controller.reset_request("req-a")
    first = controller.prepare_block(_routes())
    second = controller.prepare_block(_routes(offset=1))
    result = controller.abort(first.block_id)

    assert result.committed_positions == 0
    assert result.invalidated_block_ids == (second.block_id,)
    assert not any(snapshot.expert_to_slot for snapshot in controller.cache_snapshot())
