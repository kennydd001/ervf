"""Measured-component budget helpers for the Ornith H4 target verifier."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrnithH4Budget:
    attention_projection_ms: float
    routed_hot_per_layer_ms: float
    shared_per_layer_ms: float
    head_ms: float
    layers: int = 40
    target_tokens_per_second: float = 65.0

    @property
    def target_h4_ms(self) -> float:
        return 4000.0 / self.target_tokens_per_second

    @property
    def known_hot_floor_ms(self) -> float:
        return (
            self.attention_projection_ms
            + self.layers * (self.routed_hot_per_layer_ms + self.shared_per_layer_ms)
            + self.head_ms
        )

    @property
    def unmeasured_allowance_ms(self) -> float:
        return self.target_h4_ms - self.known_hot_floor_ms


def interpolate_curve(points: dict[int, float], count: int) -> float:
    """Piecewise-linear interpolation over measured monotonic group counts."""

    count = int(count)
    if count < 0 or count > max(points):
        raise ValueError(count)
    if count in points:
        return float(points[count])
    keys = sorted(points)
    upper = next(key for key in keys if key > count)
    lower = max(key for key in keys if key < count)
    fraction = (count - lower) / (upper - lower)
    return float(points[lower] + fraction * (points[upper] - points[lower]))
