from __future__ import annotations

import json
from pathlib import Path

import pytest

from moe_lab.craft_moe.repro_audit import (
    AuditCollector,
    disjoint,
    gap_closure,
    load_top_level_members,
    paired_gap_bootstrap_independent,
    paired_load_bootstrap_independent,
    ratio_reduction,
    upper_empirical_quantile,
)


def test_audit_collector_makes_a_misreported_gate_fatal() -> None:
    audit = AuditCollector()
    reported = True
    recalculated = 0.19 >= 0.20
    audit.equal(
        "tampered_gate",
        "synthetic",
        "gate",
        reported,
        recalculated,
        "synthetic evidence",
    )
    assert len(audit.failures) == 1
    assert not audit.summary()["all_required_checks_pass"]


def test_warning_does_not_hide_or_create_required_failure() -> None:
    audit = AuditCollector()
    audit.add(
        "telemetry",
        "synthetic",
        "benchmark",
        False,
        "missing",
        "present",
        "synthetic warning",
        severity="warning",
    )
    assert not audit.failures
    assert len(audit.warnings) == 1
    assert audit.summary()["all_required_checks_pass"]


def test_streaming_top_level_reader_avoids_large_unselected_member(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.json"
    path.write_text(
        json.dumps(
            {"gates": {"passed": False}, "raw": list(range(10_000)), "verdict": "negative"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selected = load_top_level_members(path, ["gates", "verdict"], maximum_member_bytes=100)
    assert selected == {"gates": {"passed": False}, "verdict": "negative"}


def test_gate_arithmetic_and_empirical_quantile() -> None:
    assert ratio_reduction(60, 100) == pytest.approx(0.4)
    assert gap_closure(0.5, 0.1, 0.3) == pytest.approx(0.5)
    assert upper_empirical_quantile([1, 2, 3, 4], 0.5) == 3
    assert disjoint([0, 1], [2, 3])
    assert not disjoint([0, 1], [1, 2])


def test_independent_load_bootstrap_is_deterministic_and_block_based() -> None:
    first = paired_load_bootstrap_independent(
        [10, 20], [4, 8], [3, 6], seed=7, resamples=100
    )
    second = paired_load_bootstrap_independent(
        [10, 20], [4, 8], [3, 6], seed=7, resamples=100
    )
    assert first == second
    assert first["sampling_units"] == 2
    assert first["point_estimates"]["primary_miss_reduction_fraction"] == 0.4


def test_independent_gap_bootstrap_reconciles_point_closure() -> None:
    result = paired_gap_bootstrap_independent(
        [0.5, 0.5, 0.7, 0.7],
        [0.1, 0.1, 0.2, 0.2],
        {"fixed": [0.3, 0.3, 0.45, 0.45]},
        block_size=2,
        seed=9,
        resamples=100,
    )
    assert result["sampling_units"] == 2
    assert result["point_closure"]["fixed"] == pytest.approx(0.5)
    assert len(result["raw"]["fixed"]) == 100

