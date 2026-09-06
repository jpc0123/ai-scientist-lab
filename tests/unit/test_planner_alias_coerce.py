"""Planner JSON alias coerce recovers common LLM schema slips."""

from __future__ import annotations

import json

from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.openai_compatible_provider import (
    _try_coerce_planner_content,
    build_repair_request,
)
from scientist_lab.llm.planner_contract import (
    PLANNER_CONTRACT_SCHEMA,
    coerce_planner_json_object,
)
from scientist_lab.llm.schema_parser import validate_against_schema


def test_coerce_planner_json_object_fills_required_aliases() -> None:
    raw = {
        "chosen": {
            "module": "fusion",
            "how_id": "F1",
            "idea": "Replicate early_concat on a new seed.",
            "changes": [
                {
                    "module": "fusion",
                    "description": "Keep F1 early_concat; change seed only.",
                }
            ],
        },
        "alternatives": [],
    }
    coerced = coerce_planner_json_object(raw)
    errors = validate_against_schema(coerced, PLANNER_CONTRACT_SCHEMA)
    assert errors == []
    selected = coerced["selected"]
    assert selected["requested_module"] == "fusion"
    assert selected["hypothesis"].startswith("Replicate")
    assert selected["proposed_changes"][0]["target"] == "fusion"
    assert "early_concat" in selected["proposed_changes"][0]["summary"]


def test_coerce_addresses_coverage_string_and_seed_string() -> None:
    """Regression: string addresses_coverage used to fail provider schema and
    then repair truncation dropped selected → NEED_HUMAN / campaign stop.
    """
    raw = {
        "selected": {
            "requested_module": "fusion",
            "how_id": "F3",
            "hypothesis": "Run F3 on seed 43 for matched-pair coverage.",
            "proposed_changes": [
                {
                    "target": "fusion",
                    "summary": "Run catalog F3 on seed 43.",
                    "detail": {"how_id": "F3", "seed": 43},
                }
            ],
            "seed": "43",
        },
        "verification_plan": {
            "what_to_run": "F3 seed 43 formal",
            "how_to_verify": "Compare APS_lowlight vs F1/A4 on seed 43",
            "success_criterion": "Valid APS_lowlight without claim",
            "controlled_variables": "input_mode=rgbt",
            "addresses_coverage": "F3_seed43",
        },
    }
    coerced = coerce_planner_json_object(raw)
    errors = validate_against_schema(coerced, PLANNER_CONTRACT_SCHEMA)
    assert errors == []
    assert coerced["selected"]["seed"] == 43
    assert coerced["verification_plan"]["addresses_coverage"] == ["F3_seed43"]
    assert coerced["verification_plan"]["controlled_variables"] == ["input_mode=rgbt"]


def test_try_coerce_planner_content_recovers_string_coverage() -> None:
    blob = {
        "selected": {
            "requested_module": "fusion",
            "hypothesis": "Coverage run for F3 seed 43.",
            "proposed_changes": [{"target": "fusion", "summary": "Run F3."}],
            "how_id": "F3",
        },
        "verification_plan": {
            "what_to_run": "F3",
            "how_to_verify": "APS_lowlight",
            "success_criterion": "metric present",
            "addresses_coverage": "F3_seed43",
        },
    }
    content, parsed, errors = _try_coerce_planner_content(
        json.dumps(blob, ensure_ascii=False)
    )
    assert errors == []
    assert parsed is not None
    assert parsed["verification_plan"]["addresses_coverage"] == ["F3_seed43"]
    assert '"selected"' in content


def test_coerce_addresses_question_alias() -> None:
    """Regression: addresses_question used to fail schema (extra property)."""
    raw = {
        "selected": {
            "requested_module": "fusion",
            "how_id": "P3",
            "hypothesis": "Run LLM-authored P3 overlay plugin.",
            "proposed_changes": [
                {"target": "fusion", "summary": "Run plugin P3.", "detail": {"how_id": "P3"}}
            ],
        },
        "verification_plan": {
            "what_to_run": "plugin:P3 seed 43",
            "how_to_verify": "APS_lowlight vs F1",
            "success_criterion": "metric present",
            "addresses_question": "Does true spatial thermal gate help low-light?",
        },
    }
    coerced = coerce_planner_json_object(raw)
    errors = validate_against_schema(coerced, PLANNER_CONTRACT_SCHEMA)
    assert errors == []
    assert "addresses_question" not in coerced["verification_plan"]
    assert coerced["verification_plan"]["addresses_coverage"] == [
        "Does true spatial thermal gate help low-light?"
    ]


def test_build_repair_request_keeps_large_planner_json() -> None:
    fat = {
        "selected": {
            "requested_module": "fusion",
            "hypothesis": "x" * 2000,
            "proposed_changes": [{"target": "fusion", "summary": "Run F3."}],
            "how_id": "F3",
        },
        "verification_plan": {
            "what_to_run": "F3",
            "how_to_verify": "APS",
            "success_criterion": "ok",
            "addresses_coverage": ["F3_seed43"],
        },
    }
    bad = json.dumps(fat, ensure_ascii=False)
    assert len(bad) > 1500
    req = LLMRequest(
        messages=[{"role": "user", "content": "plan"}],
        purpose="planner",
        response_schema=PLANNER_CONTRACT_SCHEMA,
        metadata={"planner_contract": True},
    )
    repair = build_repair_request(
        req,
        bad_content=bad,
        schema_errors=["$.verification_plan.addresses_coverage: x"],
    )
    user = repair.messages[-1]["content"]
    payload = json.loads(user)
    assert "selected" in payload["invalid_output"]
    assert len(payload["invalid_output"]) > 1500
