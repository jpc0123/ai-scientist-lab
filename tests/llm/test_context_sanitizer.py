"""v1.4.5 context sanitizer unit tests."""

from __future__ import annotations

from scientist_lab.agents.models import PlanningContext
from scientist_lab.llm.context_sanitizer import sanitize_planning_context


def test_sanitize_redacts_paths_and_secrets():
    context = PlanningContext(
        project_id="project_rgbt_003",
        research_goal="Improve detection; api_key=sk-leak-me-now",
        protocol={"protocol_id": "protocol_rgbt_001"},
        protocol_id="protocol_rgbt_001",
        nodes=[
            {
                "node_id": "n1",
                "status": "succeeded",
                "parameters": {"input_mode": "rgb"},
                "host_path": "D:\\secret\\path\\run.log",
                "api_key": "sk-should-redact",
            }
        ],
        evidence_records=[
            {
                "evidence_id": "e1",
                "claim_id": "c1",
                "summary": "ok",
                "combined_log": "should drop",
            }
        ],
    )
    cleaned = sanitize_planning_context(context)
    dumped = cleaned.model_dump(mode="json")
    assert "sk-leak-me-now" not in dumped["research_goal"]
    assert "[REDACTED]" in dumped["research_goal"] or "api_key=[REDACTED]" in dumped[
        "research_goal"
    ]
    assert dumped["nodes"][0]["node_id"] == "n1"
    assert "host_path" not in dumped["nodes"][0]
    assert dumped["evidence_records"][0]["evidence_id"] == "e1"
    assert "combined_log" not in dumped["evidence_records"][0]
