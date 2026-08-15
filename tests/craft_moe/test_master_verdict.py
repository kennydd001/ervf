from __future__ import annotations

from copy import deepcopy

import pytest

from moe_lab.craft_moe.master_verdict import (
    REVOLUTIONARY_GATE_IDS,
    TECHNICAL_IDS,
    build_master_verdict,
    render_master_verdict,
    validate_master_verdict,
)


def test_master_gate_is_conjunctive_and_closed() -> None:
    payload = build_master_verdict()
    gate = payload["revolutionary_v2_gate"]
    assert tuple(row["id"] for row in gate["conditions"]) == REVOLUTIONARY_GATE_IDS
    assert gate["satisfied_count"] == 0
    assert gate["all_satisfied_by_one_candidate"] is False
    assert payload["project_status"] == "closed_no_eureka"


def test_exact_control_rejects_false_eureka_promotion() -> None:
    payload = deepcopy(build_master_verdict())
    for row in payload["revolutionary_v2_gate"]["conditions"]:
        row["satisfied"] = True
    payload["revolutionary_v2_gate"]["satisfied_count"] = len(REVOLUTIONARY_GATE_IDS)
    payload["revolutionary_v2_gate"]["all_satisfied_by_one_candidate"] = True
    with pytest.raises(ValueError, match="unexpected Eureka"):
        validate_master_verdict(payload)


def test_exact_technical_terminal_inventory() -> None:
    payload = build_master_verdict()
    assert tuple(row["id"] for row in payload["technical_program"]) == TECHNICAL_IDS
    assert not any(row["status"] in {"deployable_candidate", "confirmed"} for row in payload["technical_program"])


def test_derived_accounting_forbids_multiplying_oracle_factors() -> None:
    payload = build_master_verdict()
    accounting = " ".join(
        f"{row['derivation']} {row['calculation']} {row['boundary']}"
        for row in payload["derived_accounting"]
    )
    assert "mogen niet worden vermenigvuldigd" in accounting
    assert "25% × 16 bit = 4,0 effectieve bits" in accounting
    assert "geen latency" in accounting


def test_report_has_all_required_orchestrator_sections() -> None:
    report = render_master_verdict(build_master_verdict())
    for heading in (
        "## Gemeten feiten",
        "## Afgeleide boekhouding",
        "## Subjectieve inferentie",
        "## Novelty-status",
        "## Exacte volgende actie",
    ):
        assert heading in report
    assert "closed_no_eureka" in report
    assert "0 van 6" in report
