from __future__ import annotations

from pathlib import Path

from scientist_lab.agents.models import PlanningContext
from scientist_lab.llm import (
    AuditingProvider,
    FakeProvider,
    LLMCallRepository,
    PLANNER_OUTPUT_SCHEMA,
    ReplayMissError,
    ReplayProvider,
    parse_and_validate,
    request_fingerprint,
    validate_against_schema,
)
from scientist_lab.llm.context_codec import planning_context_to_planner_request


def _context() -> PlanningContext:
    return PlanningContext(
        project_id="project_rgbt_003",
        research_goal="Offline provider replay test",
        protocol={"protocol_id": "protocol_rgbt_001"},
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        nodes=[
            {
                "node_id": "rgbt_formal_node_003",
                "status": "succeeded",
                "parameters": {"input_mode": "fusion"},
            }
        ],
        remaining_budget={"max_new_nodes": 3, "max_gpu_hours": 10},
        allowed_parameter_changes=["input_mode", "fusion_method"],
        context_sha256="abc123",
    )


def test_schema_parser_validates_planner_shape():
    payload = {
        "project_id": "p",
        "reasoning_summary": "r",
        "candidates": [
            {
                "candidate_id": "c1",
                "parent_node_id": "n1",
                "title": "t",
                "hypothesis": "h",
                "experiment_type": "ablation",
                "parameter_changes": {"input_mode": "rgb"},
            }
        ],
        "stop_recommended": False,
        "stop_reason": None,
    }
    errors = validate_against_schema(payload, PLANNER_OUTPUT_SCHEMA)
    assert errors == []

    bad = dict(payload)
    bad["candidates"] = [{"candidate_id": "c1"}]
    assert validate_against_schema(bad, PLANNER_OUTPUT_SCHEMA)


def test_fake_then_replay_identical_offline(tmp_path: Path):
    context = _context()
    request = planning_context_to_planner_request(context)
    fp = request_fingerprint(request)

    repo = LLMCallRepository(tmp_path / "llm_root")
    audited = AuditingProvider(
        FakeProvider(),
        repo,
        project_id=context.project_id,
        validate_schema=True,
    )

    first = audited.complete(request)
    assert first.provider == "fake"
    assert first.schema_valid is True
    assert first.parsed_json is not None
    assert first.parsed_json["project_id"] == "project_rgbt_003"
    assert first.parsed_json["candidates"]
    assert first.request_fingerprint == fp

    recorded = repo.require_by_fingerprint(fp)
    assert recorded.request_id == first.request_id
    assert recorded.purpose == "planner"
    assert recorded.usage.total_tokens > 0
    assert Path(str(recorded.path)).is_file()

    replay = ReplayProvider(repo)
    second = replay.complete(request)
    assert second.provider == "replay"
    assert second.request_fingerprint == fp
    assert second.content == first.content
    assert second.parsed_json == first.parsed_json
    assert second.schema_valid is True

    # Same request again remains stable.
    third = replay.complete(request)
    assert third.content == first.content


def test_replay_miss_without_prior_audit(tmp_path: Path):
    context = _context()
    request = planning_context_to_planner_request(context)
    replay = ReplayProvider(LLMCallRepository(tmp_path / "empty"))
    try:
        replay.complete(request)
        assert False, "expected ReplayMissError"
    except ReplayMissError:
        pass


def test_fake_provider_modules_have_no_network_imports():
    root = Path(__file__).resolve().parents[2] / "src" / "scientist_lab" / "llm"
    for name in (
        "fake_provider.py",
        "replay_provider.py",
        "audit.py",
        "provider.py",
        "repository.py",
        "schema_parser.py",
    ):
        text = (root / name).read_text(encoding="utf-8")
        for banned in ("import httpx", "import openai", "from openai", "from httpx"):
            assert banned not in text, f"{name} contains {banned}"


def test_parse_fenced_json():
    content = """Here you go:
```json
{"project_id": "p", "reasoning_summary": "r", "candidates": []}
```
"""
    data, errors = parse_and_validate(content, PLANNER_OUTPUT_SCHEMA)
    assert data["project_id"] == "p"
    assert errors == []
