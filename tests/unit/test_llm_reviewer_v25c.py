"""v2.5-C LLM Reviewer + semantic Memory proposals. No GPU. No API key required."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.cli import main
from scientist_lab.core.claim_gate import evaluate_claim
from scientist_lab.core.decision_rubric import evaluate_rubric
from scientist_lab.core.evidence_validator import EvidenceValidator
from scientist_lab.core.invariants import (
    InvariantError,
    assert_discard_is_not_module_ineffective,
    assert_keep_is_not_claim,
)
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.reviewer import ReviewRefused, Reviewer
from scientist_lab.core.schema_registry import load_json, validate_named
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.gateway import ScriptedProvider, resolve_gateway_provider
from scientist_lab.llm.review_replay import run_llm_review_replay

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "llm_plan_replay" / "m4_rounds3_discard"
)
RUN_ID = "run_plan_round1_neck_hr"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "m4_rounds3_discard"
    shutil.copytree(FIXTURE, dest)
    return dest


def _review_inputs(root: Path) -> dict:
    protocol = load_json(root / "protocol.json")
    plan = load_json(root / "previous_plan.json")
    result = load_json(root / "result.json")
    baseline = load_json(root / "baseline_metrics.json")
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    evidence = EvidenceValidator().validate(
        result,
        protocol=protocol,
        expected_artifacts=list(contract.get("expected_artifacts") or []),
    )
    rubric = evaluate_rubric(
        protocol,
        current_metrics=result.get("metrics") or {},
        baseline_metrics={
            k: v for k, v in baseline.items() if isinstance(v, (int, float))
        },
    )
    return {
        "protocol": protocol,
        "plan": plan,
        "result": result,
        "contract": contract,
        "evidence": evidence,
        "rubric": rubric,
    }


def _proposal_json(*, run_id: str = RUN_ID, **overrides) -> str:
    body = {
        "observation": f"VALID evidence for {run_id}: primary APS judged DISCARD by DecisionRubric.",
        "hypothesis_status": "not_supported_under_current_protocol",
        "interpretation": (
            "DecisionRubric already locked DISCARD. Probe APS declined past discard_if "
            "under the current protocol. This is a next-action DISCARD, not a ClaimGate verdict."
        ),
        "alternative_explanations": [
            "Probe budget can yield APS=0 without a formal matched pair.",
            "Labeled synthetic_control may dominate the delta.",
        ],
        "next_research_priority": (
            "Verify an allowed Adapter HOW module that is not the discarded scope, still probe-class."
        ),
        "evidence_refs": [{"run_id": run_id, "metric": "APS", "delta": -0.6}],
        "created_from": [run_id],
        "confidence": "medium",
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


def test_default_backend_is_rules() -> None:
    reviewer = Reviewer()
    assert reviewer.backend == "rules"
    assert reviewer.fallback_to_rules is False
    assert reviewer.provider is None


def test_rules_reviewer_still_discards_without_llm_trace() -> None:
    ctx = _review_inputs(FIXTURE)
    packet = Reviewer().review(**ctx)
    assert packet.source == "rules_first"
    assert packet.review_decision == "DISCARD"
    assert packet.document["review_decision"] == "DISCARD"
    assert packet.document["hypothesis_status"] == "REJECTED"
    assert packet.semantic_proposal is None
    assert packet.llm_trace is None
    validate_named("review_decision", packet.document)


def test_mock_explains_discard_without_overriding_rubric() -> None:
    ctx = _review_inputs(FIXTURE)
    packet = Reviewer(backend="llm", provider=FakeProvider()).review(**ctx)
    assert packet.review_decision == "DISCARD"
    assert packet.document["review_decision"] == "DISCARD"
    assert packet.document["hypothesis_status"] == "REJECTED"
    proposal = packet.semantic_proposal or {}
    assert proposal["hypothesis_status"] == "not_supported_under_current_protocol"
    assert "DISCARD" in proposal["interpretation"]
    assert "ineffective" not in proposal["interpretation"].lower()
    assert "无效" not in proposal["interpretation"]
    assert proposal["evidence_refs"][0]["run_id"] == RUN_ID
    assert proposal["evidence_refs"][0]["metric"] == "APS"
    assert packet.source == "llm"
    assert packet.llm_trace and packet.llm_trace["rubric_locked"] is True


def test_llm_review_decision_override_fail_closed() -> None:
    ctx = _review_inputs(FIXTURE)
    raw = _proposal_json(review_decision="KEEP")
    reviewer = Reviewer(backend="llm", provider=ScriptedProvider(raw))
    with pytest.raises(ReviewRefused, match="must not override"):
        reviewer.review(**ctx)


def test_llm_module_ineffective_claim_fail_closed() -> None:
    ctx = _review_inputs(FIXTURE)
    raw = _proposal_json(
        interpretation="The neck module is ineffective; DISCARD proves it does not work."
    )
    reviewer = Reviewer(backend="llm", provider=ScriptedProvider(raw))
    with pytest.raises(ReviewRefused, match="module-ineffective|effectiveness"):
        reviewer.review(**ctx)


def test_keep_is_not_written_as_claim_supported() -> None:
    ctx = _review_inputs(FIXTURE)
    raw = _proposal_json(hypothesis_status="SUPPORTED")
    reviewer = Reviewer(backend="llm", provider=ScriptedProvider(raw))
    with pytest.raises(ReviewRefused, match="SUPPORTED|hypothesis_status"):
        reviewer.review(**ctx)


def test_writer_rejects_missing_evidence_refs(tmp_path: Path) -> None:
    writer = MemoryWriter(tmp_path / "memory")
    proposal = json.loads(_proposal_json())
    proposal["evidence_refs"] = []
    with pytest.raises(InvariantError, match="missing evidence_refs"):
        writer.persist_semantic_proposal(
            proposal,
            run_id=RUN_ID,
            review_decision="DISCARD",
            lesson_type="negative_evidence",
            module="neck",
        )


def test_writer_accepts_proposal_with_refs(tmp_path: Path) -> None:
    writer = MemoryWriter(tmp_path / "memory")
    proposal = json.loads(_proposal_json())
    lesson = writer.persist_semantic_proposal(
        proposal,
        run_id=RUN_ID,
        review_decision="DISCARD",
        lesson_type="negative_evidence",
        module="neck",
        metric="APS",
        delta=-0.6,
    )
    validate_named("research_lesson", lesson)
    assert lesson["evidence"][0]["run_id"] == RUN_ID
    assert lesson["created_from"] == [RUN_ID]
    assert "ineffective" not in lesson["statement"].lower()
    stored = writer.load_lessons()
    assert lesson["lesson_id"] in stored


def test_aps_not_impersonated_by_map() -> None:
    ctx = _review_inputs(FIXTURE)
    raw = _proposal_json(
        evidence_refs=[{"run_id": RUN_ID, "metric": "mAP50"}],
        interpretation="Primary metric is mAP50; treat that as APS.",
    )
    reviewer = Reviewer(backend="llm", provider=ScriptedProvider(raw))
    with pytest.raises(ReviewRefused, match="mAP"):
        reviewer.review(**ctx)


def test_claim_gate_still_blocks_discard_as_module_ineffective() -> None:
    ctx = _review_inputs(FIXTURE)
    packet = Reviewer(backend="llm", provider=FakeProvider()).review(**ctx)
    verdict = evaluate_claim(
        {
            "claim_id": "claim_semantic_discard_not_ineffective",
            "claim_type": "component_effectiveness",
            "claim_strength": "C2",
            "claim_text": packet.semantic_proposal["interpretation"],
            "metric": "APS",
        },
        protocol=ctx["protocol"],
        evidence={
            "result": ctx["result"],
            "metrics": ctx["result"]["metrics"],
            "review_decision": packet.review_decision,
            "run_level": "probe",
        },
        plan=ctx["plan"],
        review_decision=packet.review_decision,
    )
    assert verdict["status"] != "SUPPORTED"
    assert_discard_is_not_module_ineffective(
        {**verdict, "review_decision": "DISCARD", "claim_text": packet.semantic_proposal["interpretation"]}
    )
    keep_verdict = evaluate_claim(
        {
            "claim_id": "claim_keep_is_not_supported",
            "claim_type": "comparative",
            "claim_strength": "C1",
            "claim_text": "KEEP means the hypothesis is SUPPORTED",
            "metric": "APS",
            "asserts": {"outperform": True},
        },
        protocol=ctx["protocol"],
        evidence={
            "result": ctx["result"],
            "metrics": ctx["result"]["metrics"],
            "review_decision": "KEEP",
            "run_level": "probe",
        },
        review_decision="KEEP",
    )
    assert keep_verdict["status"] != "SUPPORTED"
    assert_keep_is_not_claim(keep_verdict)


def test_fallback_to_rules_is_opt_in() -> None:
    ctx = _review_inputs(FIXTURE)
    closed = Reviewer(backend="llm", provider=ScriptedProvider("not-json"))
    with pytest.raises(ReviewRefused, match="fail_closed"):
        closed.review(**ctx)
    packet = Reviewer(
        backend="llm",
        provider=ScriptedProvider("not-json"),
        fallback_to_rules=True,
    ).review(**ctx)
    assert packet.source == "rules_first"
    assert packet.review_decision == "DISCARD"
    assert packet.semantic_proposal is None


def test_cli_llm_review_replay_mock_no_gpu(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    source_lessons = set(MemoryWriter(dest / "memory").load_lessons())
    code = main(["llm-review-replay", "--run-dir", str(dest)])
    assert code == 0
    report_path = dest / ".llm_review_replay" / "replay_report.json"
    assert report_path.is_file()
    report = load_json(report_path)
    assert report["ok"] is True
    assert report["gpu"] is False
    assert report["execute"] is False
    assert report["llm_review_decision"] == "DISCARD"
    assert report["document_hypothesis_status"] == "REJECTED"
    assert report["semantic_proposal"]["hypothesis_status"] == (
        "not_supported_under_current_protocol"
    )
    assert report["writer"]["accepted"] is True
    assert report["memory_unchanged"] is True
    assert set(MemoryWriter(dest / "memory").load_lessons()) == source_lessons
    isolated = MemoryWriter(dest / ".llm_review_replay" / "memory")
    assert f"LESSON-{RUN_ID}-semantic-001" in isolated.load_lessons()
    events = (dest / ".llm_review_replay" / "events.jsonl").read_text(encoding="utf-8")
    assert "prompt_hash" in events
    assert "reviewer" in events


def test_cli_live_without_key_does_not_fake_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _copy_fixture(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    code = main(["llm-review-replay", "--run-dir", str(dest), "--live"])
    assert code == 1
    report = load_json(dest / ".llm_review_replay" / "replay_report.json")
    assert report["ok"] is False
    assert report["fail_closed"] is True


def test_run_llm_review_replay_helper(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    report = run_llm_review_replay(dest)
    assert report["ok"] is True
    assert report["rubric_locked"] is True
    assert report["semantic_proposal"]["created_from"] == [RUN_ID]


def test_live_provider_without_key_is_fail_closed() -> None:
    with pytest.raises((MissingAPIKeyError, RealProviderNotEnabledError)):
        resolve_gateway_provider(live=True, environ={"LLM_PROVIDER": "openai-compatible"})
