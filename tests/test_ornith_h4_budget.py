from __future__ import annotations

import pytest

from moe_lab.ornith.h4_budget import OrnithH4Budget, interpolate_curve


def test_budget_properties():
    budget = OrnithH4Budget(
        attention_projection_ms=23.0,
        routed_hot_per_layer_ms=0.5,
        shared_per_layer_ms=0.1,
        head_ms=1.5,
    )
    assert budget.target_h4_ms == pytest.approx(4000 / 65)
    assert budget.known_hot_floor_ms == pytest.approx(48.5)
    assert budget.unmeasured_allowance_ms == pytest.approx(4000 / 65 - 48.5)


def test_curve_interpolation():
    points = {0: 0.0, 4: 0.4, 8: 0.6}
    assert interpolate_curve(points, 2) == pytest.approx(0.2)
    assert interpolate_curve(points, 6) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        interpolate_curve(points, 9)
