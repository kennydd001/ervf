from __future__ import annotations

from copy import deepcopy

import pytest

from moe_lab.craft_moe.novelty_audit import (
    ALLOWED_LABELS,
    CLAIM_IDS,
    MANDATORY_FAMILY_IDS,
    build_matrix,
    render_prior_art,
    render_verdict,
    validate_matrix,
)


def test_exact_mandatory_coverage_and_label_vocabulary() -> None:
    matrix = build_matrix()
    assert tuple(row["id"] for row in matrix["mandatory_families"]) == MANDATORY_FAMILY_IDS
    assert tuple(row["id"] for row in matrix["claim_units"]) == CLAIM_IDS
    labels = {
        row["label"]
        for section in ("mandatory_families", "claim_units")
        for row in matrix[section]
    }
    assert labels <= ALLOWED_LABELS
    assert matrix["summary"]["broad_novelty_claim_supported"] is False
    assert matrix["summary"]["eureka_supported"] is False


def test_exact_control_rejects_promoted_or_invented_label() -> None:
    matrix = deepcopy(build_matrix())
    matrix["claim_units"][0]["label"] = "proven novel"
    with pytest.raises(ValueError, match="illegal label"):
        validate_matrix(matrix)


def test_exact_control_rejects_missing_primary_source() -> None:
    matrix = deepcopy(build_matrix())
    del matrix["source_catalog"][matrix["claim_units"][0]["closest_source_ids"][0]]
    with pytest.raises(ValueError, match="missing source"):
        validate_matrix(matrix)


def test_source_catalog_uses_unique_https_urls_and_allowed_types() -> None:
    matrix = build_matrix()
    urls = [source["url"] for source in matrix["source_catalog"].values()]
    assert all(url.startswith("https://") for url in urls)
    assert len(urls) == len(set(urls))
    assert matrix["summary"]["patent_publication_count"] == 4


def test_deterministic_reports_keep_negative_verdict_visible() -> None:
    matrix = build_matrix()
    prior_art = render_prior_art(matrix)
    verdict = render_verdict(matrix)
    assert prior_art == render_prior_art(matrix)
    assert verdict == render_verdict(matrix)
    assert "geen verdedigbare brede nieuwheidsclaim" in prior_art.lower()
    assert "geen eureka" in verdict.lower()
    assert "possibly novel intersection" in verdict
    assert "not searched sufficiently" in verdict

