from __future__ import annotations

import pytest

from moe_lab.rsiv_moe.p1_verify import Audit, _verify_metric


def test_metric_verifier_accepts_exact_cold_reciprocals() -> None:
    audit = Audit()
    _verify_metric(
        {
            "invocations": 12,
            "x_fast_fraction": 0.75,
            "z_fast_fraction": 0.5,
            "double_gate_fast_fraction": 0.5,
            "router_mass_double_gate_fast_fraction": 0.4,
            "projected_cold_byte_fraction": 0.25,
            "projected_routed_cold_byte_reduction": 4.0,
            "router_mass_projected_cold_byte_fraction": 0.2,
            "router_mass_projected_cold_byte_reduction": 5.0,
        },
        audit,
        "metric",
    )
    assert not audit.failures


def test_metric_verifier_rejects_inconsistent_reduction() -> None:
    audit = Audit()
    _verify_metric(
        {
            "invocations": 1,
            "x_fast_fraction": 0.0,
            "z_fast_fraction": 0.0,
            "double_gate_fast_fraction": 0.0,
            "router_mass_double_gate_fast_fraction": 0.0,
            "projected_cold_byte_fraction": 0.5,
            "projected_routed_cold_byte_reduction": 3.0,
            "router_mass_projected_cold_byte_fraction": 1.0,
            "router_mass_projected_cold_byte_reduction": 1.0,
        },
        audit,
        "metric",
    )
    assert any("cold_reciprocal" in failure for failure in audit.failures)

