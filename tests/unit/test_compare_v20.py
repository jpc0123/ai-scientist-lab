"""v2.0.3 comparison conclusion labels."""

from __future__ import annotations

from scientist_lab.services.comparison_labels import (
    CONCLUSION_LABELS,
    conclusion_from_relation,
    enrich_comparison,
)


def test_conclusion_labels_cover_four_outcomes():
    assert set(CONCLUSION_LABELS) == {
        "better",
        "worse",
        "practically_equivalent",
        "inconclusive",
    }
    assert conclusion_from_relation("candidate_better") == "better"
    assert conclusion_from_relation("baseline_better") == "worse"
    assert conclusion_from_relation("practically_equivalent") == "practically_equivalent"
    assert conclusion_from_relation("unclear") == "inconclusive"


def test_enrich_comparison_adds_textual_conclusion():
    payload = enrich_comparison(
        {"hypothesis_status": "supported", "practically_equivalent": False},
        mode="executions",
    )
    assert payload["conclusion"] == "better"
    assert "better" in payload["conclusion_label"]
    assert payload["compare_mode"] == "executions"
