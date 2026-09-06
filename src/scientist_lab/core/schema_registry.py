"""Load and validate Architecture Freeze JSON Schemas."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

# scientist-lab/schemas  (repo layout: src/scientist_lab/core/ -> ../../../schemas)
SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas"

SCHEMA_FILES: dict[str, str] = {
    "common": "common.schema.json",
    "research_protocol": "research_protocol.schema.json",
    "experiment_plan": "experiment_plan.schema.json",
    "experiment_contract": "experiment_contract.schema.json",
    "experiment_run": "experiment_run.schema.json",
    "experiment_result": "experiment_result.schema.json",
    "review_decision": "review_decision.schema.json",
    "research_lesson": "research_lesson.schema.json",
    "strategy": "strategy.schema.json",
    "frozen_fingerprint": "frozen_fingerprint.schema.json",
    "research_event": "research_event.schema.json",
    "claim": "claim.schema.json",
    "claim_gate_result": "claim_gate_result.schema.json",
    "paper_record": "paper_record.schema.json",
    "literature_query": "literature_query.schema.json",
    "literature_evidence": "literature_evidence.schema.json",
    "how_candidate": "how_candidate.schema.json",
    "low_light_subset": "low_light_subset.schema.json",
    "dataset_contract": "dataset_contract.schema.json",
    "r0_baseline": "r0_baseline.schema.json",
    "trajectory_step": "trajectory_step.schema.json",
}


class SchemaValidationError(ValueError):
    def __init__(self, schema_name: str, errors: list[str]) -> None:
        self.schema_name = schema_name
        self.errors = errors
        super().__init__(f"{schema_name}: " + "; ".join(errors))


def list_schema_names() -> list[str]:
    return sorted(SCHEMA_FILES.keys())


@lru_cache(maxsize=1)
def _registry() -> Registry:
    resources: list[tuple[str, Resource]] = []
    for filename in SCHEMA_FILES.values():
        path = SCHEMA_DIR / filename
        raw = json.loads(path.read_text(encoding="utf-8"))
        uri = raw.get("$id") or f"https://scientist-lab.local/schemas/{filename}"
        resources.append((uri, Resource.from_contents(raw, default_specification=DRAFT202012)))
        # Also allow relative ref resolution by filename URI used in $ref
        resources.append(
            (
                filename,
                Resource.from_contents(raw, default_specification=DRAFT202012),
            )
        )
    registry: Registry = Registry()
    for uri, resource in resources:
        registry = registry.with_resource(uri=uri, resource=resource)
    return registry


@lru_cache(maxsize=None)
def _validator(schema_name: str) -> Draft202012Validator:
    if schema_name not in SCHEMA_FILES:
        raise KeyError(f"Unknown schema: {schema_name}")
    path = SCHEMA_DIR / SCHEMA_FILES[schema_name]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Draft202012Validator(raw, registry=_registry())


def validate_named(schema_name: str, document: dict[str, Any]) -> None:
    """Validate document against a named schema; raise SchemaValidationError on failure."""
    validator = _validator(schema_name)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    if errors:
        msgs = [
            f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
            for err in errors
        ]
        raise SchemaValidationError(schema_name, msgs)


def validate_document(schema_name: str, document: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the document unchanged if valid."""
    validate_named(schema_name, document)
    return document


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
