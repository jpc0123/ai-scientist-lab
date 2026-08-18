"""JSON extraction and lightweight JSON Schema validation (no network)."""

from __future__ import annotations

import json
import re
from typing import Any


class SchemaValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init("; ".join(self.errors) if self.errors else "schema invalid")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON object from raw text or fenced markdown."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty LLM content")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    elif not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM JSON root must be an object")
    return data


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    names = [expected] if isinstance(expected, str) else list(expected)
    actual = _type_name(value)
    for name in names:
        if name == "number" and actual in {"number", "integer"}:
            return True
        if name == actual:
            return True
    return False


def validate_against_schema(
    data: Any,
    schema: dict[str, Any] | None,
    *,
    path: str = "$",
) -> list[str]:
    """Validate data against a minimal JSON Schema subset.

    Supported keywords: type, properties, required, items, enum,
    additionalProperties (bool only), minItems / maxItems.
    """
    if not schema:
        return []

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(data, expected_type):
        errors.append(
            f"{path}: expected type {expected_type}, got {_type_name(data)}"
        )
        return errors

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: value not in enum")

    if isinstance(data, dict) and (expected_type in {None, "object"} or "properties" in schema):
        props = dict(schema.get("properties") or {})
        required = list(schema.get("required") or [])
        for key in required:
            if key not in data:
                errors.append(f"{path}.{key}: missing required property")
        additional = schema.get("additionalProperties", True)
        for key, value in data.items():
            if key in props:
                errors.extend(
                    validate_against_schema(
                        value, props[key], path=f"{path}.{key}"
                    )
                )
            elif additional is False:
                errors.append(f"{path}.{key}: additional property not allowed")

    if isinstance(data, list) and (expected_type in {None, "array"} or "items" in schema):
        if "minItems" in schema and len(data) < int(schema["minItems"]):
            errors.append(f"{path}: fewer than minItems={schema['minItems']}")
        if "maxItems" in schema and len(data) > int(schema["maxItems"]):
            errors.append(f"{path}: more than maxItems={schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(data):
                errors.extend(
                    validate_against_schema(
                        item, item_schema, path=f"{path}[{idx}]"
                    )
                )

    return errors


def parse_and_validate(
    content: str,
    schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Extract JSON and optionally validate. Returns (data, errors).

    Plain chat replies are not JSON. Skip parsing unless a schema was requested,
    and never raise JSONDecodeError for free-form text.
    """
    if not schema:
        return {}, []
    try:
        data = extract_json_object(content)
    except (ValueError, json.JSONDecodeError) as exc:
        return {}, [str(exc)]
    errors = validate_against_schema(data, schema)
    return data, errors


# Minimal planner-shaped schema used by FakeProvider / Replay checks.
PLANNER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["project_id", "reasoning_summary", "candidates"],
    "additionalProperties": True,
    "properties": {
        "project_id": {"type": "string"},
        "reasoning_summary": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "candidate_id",
                    "parent_node_id",
                    "title",
                    "hypothesis",
                    "experiment_type",
                    "parameter_changes",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "parent_node_id": {"type": "string"},
                    "title": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "experiment_type": {
                        "type": "string",
                        "enum": [
                            "improve",
                            "ablation",
                            "debug",
                            "robustness",
                            "efficiency",
                            "replication",
                        ],
                    },
                    "parameter_changes": {"type": "object"},
                    "priority": {"type": "number"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "stop_recommended": {"type": "boolean"},
        "stop_reason": {"type": ["string", "null"]},
    },
}


CRITIC_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["candidate_id", "recommendation"],
    "additionalProperties": True,
    "properties": {
        "candidate_id": {"type": "string"},
        "scientific_validity": {
            "type": "string",
            "enum": ["valid", "weak", "invalid"],
        },
        "recommendation": {
            "type": "string",
            "enum": ["accept", "revise", "reject"],
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
    },
}
