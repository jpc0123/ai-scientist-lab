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
    assert proposal["claim_stance"] == "no_module_efficacy_claim"
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


def test_llm_module_ineffective_claim_soft_coerced() -> None:
    """Efficacy prose is soft-coerced; campaign must not fail_closed."""
    ctx = _review_inputs(FIXTURE)
    raw = _proposal_json(
        interpretation="The neck module is ineffective; DISCARD proves it does not work.",
        claim_stance="no_module_efficacy_claim",
    )
    packet = Reviewer(backend="llm", provider=ScriptedProvider(raw)).review(**ctx)
    assert packet.review_decision == "DISCARD"
    proposal = packet.semantic_proposal or {}
    assert proposal["claim_stance"] == "no_module_efficacy_claim"
    assert "effectiveness_phrasing_coerced" in (proposal.get("contract_warnings") or [])
    assert "ineffective" not in proposal["interpretation"].lower()
    assert "module-efficacy" in proposal["interpretation"].lower() or "next-action" in proposal[
        "interpretation"
    ].lower()


def test_llm_claim_stance_defaults_and_forces_for_keep() -> None:
    from scientist_lab.llm.reviewer_contract import (
        build_contract_input,
        parse_reviewer_completion,
    )

    ctx = _review_inputs(FIXTURE)
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=ctx["plan"],
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="INCONCLUSIVE",
    )
    raw = _proposal_json(
        hypothesis_status="not_a_claim",
        claim_stance="deferred_to_claim_gate",
    )
    mapped = parse_reviewer_completion(raw, payload)
    assert mapped["claim_stance"] == "no_module_efficacy_claim"
    assert "claim_stance_forced_no_module_efficacy_claim" in mapped["contract_warnings"]


def test_llm_allows_negating_effective_ineffective_disclaimer() -> None:
    """Live M1: 'does not constitute a claim that X is effective/ineffective' is OK."""
    from scientist_lab.llm.reviewer_contract import (
        build_contract_input,
        parse_reviewer_completion,
    )

    ctx = _review_inputs(FIXTURE)
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=ctx["plan"],
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="INCONCLUSIVE",
    )
    raw = _proposal_json(
        hypothesis_status="not_a_claim",
        interpretation=(
            "DecisionRubric already locked KEEP. KEEP is a next-round action, not ClaimGate "
            "SUPPORTED. This does not constitute a formal claim that N1 is effective or ineffective."
        ),
    )
    mapped = parse_reviewer_completion(raw, payload)
    assert mapped["hypothesis_status"] == "not_a_claim"
    assert "ineffective" in mapped["interpretation"].lower()


def test_llm_allows_not_that_effective_module_hedge() -> None:
    """Stage B failure mode: 'not that … as an effective module' must not fail_closed."""
    from scientist_lab.llm.reviewer_contract import (
        _has_banned_effectiveness_claim,
        build_contract_input,
        parse_reviewer_completion,
    )

    hedge = (
        "KEEP here means the baseline is valid for future comparison rounds, "
        "not that the current fusion setting is ClaimGate SUPPORTED as an effective module. "
        "Minor deltas do not indicate module effectiveness or ineffectiveness."
    )
    assert _has_banned_effectiveness_claim(hedge) is False

    ctx = _review_inputs(FIXTURE)
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=ctx["plan"],
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="INCONCLUSIVE",
    )
    raw = _proposal_json(hypothesis_status="not_a_claim", interpretation=hedge)
    mapped = parse_reviewer_completion(raw, payload)
    assert mapped["hypothesis_status"] == "not_a_claim"


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
    monkeypatch.setenv("SCIENTIST_LAB_RUNTIME_DIR", str(tmp_path / "empty_runtime"))
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
    assert report["persist_pack"] is False
    assert (dest / "semantic_review.json").is_file() is False


def test_review_replay_prefers_executed_plan_json(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    executed = load_json(dest / "previous_plan.json")
    executed["plan_id"] = "plan_round1_executed_f3"
    executed["round_index"] = 1
    executed["how_id"] = "F3"
    executed["hypothesis"] = "gated_multiscale may help low-light APS_lowlight vs early_concat."
    (dest / "plan.json").write_text(
        json.dumps(executed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = run_llm_review_replay(dest)
    assert report["ok"] is True
    assert report["plan_id"] == "plan_round1_executed_f3"


def test_review_replay_persist_pack_writes_sidecar_not_rubric(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    before_review = (dest / "review.json").read_text(encoding="utf-8")
    (dest / "reviewer_live_fail_closed.json").write_text(
        json.dumps({"fail_closed": True, "error": "stale"}, ensure_ascii=False),
        encoding="utf-8",
    )
    source_lessons = set(MemoryWriter(dest / "memory").load_lessons())
    report = run_llm_review_replay(dest, persist_pack=True)
    assert report["ok"] is True
    assert report["pack_write"]["review_json_unchanged"] is True
    assert (dest / "semantic_review.json").is_file()
    sidecar = load_json(dest / "semantic_review.json")
    assert sidecar["source"] == "llm"
    assert sidecar["keep_is_not_claim"] is True
    assert sidecar["g2_not_claimed"] is True
    assert sidecar["semantic_proposal"]["hypothesis_status"] == (
        "not_supported_under_current_protocol"
    )
    assert (dest / "review.json").read_text(encoding="utf-8") == before_review
    after = set(MemoryWriter(dest / "memory").load_lessons())
    assert after - source_lessons
    assert report["memory_unchanged"] is False
    assert (dest / "reviewer_live_fail_closed.json").is_file() is False


def test_keep_coerces_inconclusive_budget_to_not_a_claim() -> None:
    """Live M1: KEEP + inconclusive_budget is a common LLM slip; coerce, don't fail."""
    from scientist_lab.llm.reviewer_contract import (
        build_contract_input,
        parse_reviewer_completion,
    )

    ctx = _review_inputs(FIXTURE)
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=ctx["plan"],
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="SUPPORTED",
    )
    proposal = parse_reviewer_completion(
        _proposal_json(hypothesis_status="inconclusive_budget"),
        payload,
    )
    assert proposal["hypothesis_status"] == "not_a_claim"
    assert proposal["locked_review_decision"] == "KEEP"


def test_reviewer_may_cite_executed_plan_how() -> None:
    from scientist_lab.llm.reviewer_contract import (
        ReviewerContractError,
        parse_reviewer_completion,
        build_contract_input,
    )

    ctx = _review_inputs(FIXTURE)
    plan = dict(ctx["plan"])
    plan["proposed_changes"] = [
        {
            "target": "fusion",
            "summary": "Replace F1 early_concat with F3 gated_multiscale.",
            "detail": {"how_id": "F3"},
        }
    ]
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=plan,
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="DISCARD",
        document_hypothesis_status="REJECTED",
    )
    raw = _proposal_json(
        observation=(
            "VALID evidence: F3 was not run; the executed plan still cites early_concat "
            f"as the R0 comparator for {RUN_ID}."
        )
    )
    proposal = parse_reviewer_completion(raw, payload)
    assert "early_concat" in proposal["observation"]

    stray = _proposal_json(
        observation="Switch fusion_method=novel_blend for the next round.",
        interpretation="Set fusion_method to a new operator.",
        next_research_priority="Use fusion_method=novel_blend.",
    )
    with pytest.raises(ReviewerContractError, match="must not emit HOW"):
        parse_reviewer_completion(stray, payload)


def test_reviewer_may_cite_executed_plugin_fusion_method() -> None:
    """Replicating the same plugin:P3 HOW must not trip 'emit HOW'."""
    from scientist_lab.llm.reviewer_contract import (
        parse_reviewer_completion,
        build_contract_input,
    )

    ctx = _review_inputs(FIXTURE)
    plan = dict(ctx["plan"])
    plan["how_id"] = "P3"
    plan["proposed_changes"] = [
        {
            "target": "fusion",
            "summary": "Run overlay plugin P3.",
            "detail": {"how_id": "P3", "fusion_method": "plugin:P3"},
        }
    ]
    plan["hypothesis"] = "P3 thermal spatial gate vs F1 early_concat control."
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=plan,
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="INCONCLUSIVE",
    )
    raw = _proposal_json(
        observation="VALID evidence KEEP for P3; delta within keep threshold.",
        hypothesis_status="not_a_claim",
        claim_stance="no_module_efficacy_claim",
        interpretation=(
            "KEEP is not ClaimGate. Negative delta vs F1 early_concat matched "
            "control is inconclusive under one seed."
        ),
        next_research_priority=(
            "Replicate the P3 overlay plugin (fusion_method=plugin:P3, standard neck) "
            "on additional seeds before any G2 claim."
        ),
    )
    proposal = parse_reviewer_completion(raw, payload)
    assert "plugin:P3" in proposal["next_research_priority"]


def test_reviewer_may_cite_executed_n1_fdpn_live_m1() -> None:
    """Live M1: citing FDPN when plan ran N1 is observation, not invention."""
    from scientist_lab.llm.reviewer_contract import (
        ReviewerContractError,
        parse_reviewer_completion,
        build_contract_input,
    )

    ctx = _review_inputs(FIXTURE)
    plan = dict(ctx["plan"])
    plan["how_id"] = "N1"
    plan["proposed_changes"] = [
        {
            "target": "neck",
            "summary": "Switch to FDPN neck (N1).",
            "detail": {"how_id": "N1", "neck_type": "fdpn"},
        }
    ]
    plan["hypothesis"] = "FDPN neck (N1) improves multi-scale aggregation."
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=plan,
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="SUPPORTED",
    )
    raw = _proposal_json(
        hypothesis_status="not_a_claim",
        interpretation=(
            "The hypothesis that FDPN neck provides richer multi-scale features "
            "is not supported by this single-seed zero delta. KEEP is not a claim."
        ),
        next_research_priority=(
            "Replicate the same N1/FDPN neck protocol on another seed."
        ),
    )
    proposal = parse_reviewer_completion(raw, payload)
    assert "FDPN" in proposal["interpretation"]

    invent = _proposal_json(
        hypothesis_status="not_a_claim",
        interpretation="Next we should invent FDPN-style aggregation without a catalog HOW.",
        next_research_priority="write python for a new network neck.",
    )
    # Plan here still has N1 — invent/write-python must still fail closed.
    with pytest.raises(ReviewerContractError, match="invented operator"):
        parse_reviewer_completion(invent, payload)

    # FDPN prose without executed N1/A4/fdpn remains invention.
    # Fixture plan/protocol may already mention fdpn/A4 — scrub for this case.
    bare_plan = {
        "plan_id": "plan_no_fdpn",
        "round_index": 1,
        "how_id": "F1",
        "hypothesis": "Early concat fusion baseline.",
        "modification_scope": ["fusion"],
        "budget_class": "probe",
    }
    bare_protocol = {
        "protocol_id": ctx["protocol"].get("protocol_id"),
        "protocol_version": ctx["protocol"].get("protocol_version"),
        "objective": dict(ctx["protocol"].get("objective") or {}),
        "decision_policy": dict(ctx["protocol"].get("decision_policy") or {}),
        "editable_scope": list(ctx["protocol"].get("editable_scope") or []),
    }
    bare = build_contract_input(
        protocol=bare_protocol,
        plan=bare_plan,
        result={"metrics": {"APS": 0.1}, "execution": {"status": "success"}},
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="SUPPORTED",
    )
    with pytest.raises(ReviewerContractError, match="invented operator"):
        parse_reviewer_completion(
            _proposal_json(
                hypothesis_status="not_a_claim",
                interpretation="Try FDPN next even though this plan did not run it.",
            ),
            bare,
        )


def test_valid_reviewer_json_without_selected_passes_schema() -> None:
    from scientist_lab.llm.reviewer_contract import REVIEWER_CONTRACT_SCHEMA
    from scientist_lab.llm.schema_parser import parse_and_validate

    raw = _proposal_json()
    data, errors = parse_and_validate(raw, REVIEWER_CONTRACT_SCHEMA)
    assert errors == []
    assert "selected" not in data
    ctx = _review_inputs(FIXTURE)
    packet = Reviewer(backend="llm", provider=ScriptedProvider(raw)).review(**ctx)
    assert packet.semantic_proposal is not None
    assert "selected" not in packet.semantic_proposal
    assert packet.review_decision == "DISCARD"


def test_keep_reviewer_json_without_selected_passes() -> None:
    from scientist_lab.llm.reviewer_contract import (
        build_contract_input,
        parse_reviewer_completion,
    )

    ctx = _review_inputs(FIXTURE)
    payload = build_contract_input(
        protocol=ctx["protocol"],
        plan=ctx["plan"],
        result=ctx["result"],
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision="KEEP",
        document_hypothesis_status="SUPPORTED",
    )
    raw = _proposal_json(
        hypothesis_status="not_a_claim",
        interpretation=(
            "DecisionRubric already locked KEEP. KEEP is a next-round action, "
            "not ClaimGate SUPPORTED, and a single-seed delta is not a G2 claim."
        ),
        next_research_priority="Replicate the same protocol on another seed before any G2 claim.",
    )
    proposal = parse_reviewer_completion(raw, payload)
    assert "selected" not in proposal
    assert proposal["hypothesis_status"] == "not_a_claim"


def test_missing_reviewer_fields_fail_closed() -> None:
    ctx = _review_inputs(FIXTURE)
    reviewer = Reviewer(
        backend="llm",
        provider=ScriptedProvider(json.dumps({"observation": "VALID evidence."})),
    )
    with pytest.raises(ReviewRefused, match="fail_closed"):
        reviewer.review(**ctx)


def test_planner_selected_object_on_reviewer_fail_closed() -> None:
    ctx = _review_inputs(FIXTURE)
    raw = _proposal_json(
        selected={
            "requested_module": "fusion",
            "hypothesis": "x",
            "proposed_changes": [{"target": "fusion", "summary": "F3"}],
        }
    )
    reviewer = Reviewer(backend="llm", provider=ScriptedProvider(raw))
    with pytest.raises(ReviewRefused, match="selected"):
        reviewer.review(**ctx)


def test_persist_pack_fail_closed_writes_sidecar_not_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _copy_fixture(tmp_path)
    before_review = (dest / "review.json").read_text(encoding="utf-8")
    monkeypatch.setattr(
        "scientist_lab.llm.review_replay.resolve_gateway_provider",
        lambda **_kwargs: ScriptedProvider("not-json"),
    )
    report = run_llm_review_replay(dest, persist_pack=True)
    assert report["ok"] is False
    assert report["fail_closed"] is True
    assert (dest / "semantic_review.json").is_file() is False
    fail = load_json(dest / "reviewer_live_fail_closed.json")
    assert fail["fail_closed"] is True
    assert fail["semantic_review_written"] is False
    assert fail["review_json_unchanged"] is True
    assert (dest / "review.json").read_text(encoding="utf-8") == before_review


def test_live_provider_without_key_is_fail_closed() -> None:
    with pytest.raises((MissingAPIKeyError, RealProviderNotEnabledError)):
        resolve_gateway_provider(live=True, environ={"LLM_PROVIDER": "openai-compatible"})


R3_RUN_ID = (
    "run_plan_round3_from_run_plan_round2_from_run_plan_round1_from_exec_31eff20c4e0c"
)
R3_TRUNCATED = (
    "run_plan_round3_from_run_plan_round2_from_run_plan_round1_from_exec_31eff20c"
)
R4_RUN_ID = (
    "run_plan_round4_from_run_plan_round3_from_run_plan_round2_"
    "from_run_plan_round1_from_exec_31eff20c4e0c"
)


def _contract_payload(ctx: dict, *, run_id: str, locked: str = "KEEP"):
    from scientist_lab.llm.reviewer_contract import build_contract_input

    result = dict(ctx["result"])
    result["run_id"] = run_id
    return build_contract_input(
        protocol=ctx["protocol"],
        plan=ctx["plan"],
        result=result,
        evidence=ctx["evidence"].to_dict()
        if hasattr(ctx["evidence"], "to_dict")
        else dict(ctx["evidence"]),
        rubric={
            "objective_check": ctx["rubric"].objective_check,
            "constraint_check": ctx["rubric"].constraint_check,
            "primary_delta": ctx["rubric"].primary_delta,
            "constraints_ok": ctx["rubric"].constraints_ok,
            "suggest_validate": ctx["rubric"].suggest_validate,
            "suggest_discard_threshold": ctx["rubric"].suggest_discard_threshold,
            "keep_threshold_ok": ctx["rubric"].keep_threshold_ok,
        },
        locked_review_decision=locked,
        document_hypothesis_status="SUPPORTED" if locked == "KEEP" else "REJECTED",
    )


def test_canonicalize_cited_run_id_prefix_not_foreign():
    from scientist_lab.llm.reviewer_contract import canonicalize_cited_run_id

    assert canonicalize_cited_run_id(R3_TRUNCATED, R3_RUN_ID) == R3_RUN_ID
    assert canonicalize_cited_run_id(R3_RUN_ID, R3_RUN_ID) == R3_RUN_ID
    assert canonicalize_cited_run_id("run_plan_round3", R3_RUN_ID) == "run_plan_round3"
    assert (
        canonicalize_cited_run_id("run_plan_round1_from_exec_31eff20c4e0c", R3_RUN_ID)
        == "run_plan_round1_from_exec_31eff20c4e0c"
    )
    clipped_r4 = R4_RUN_ID[:76]
    assert R4_RUN_ID.startswith(clipped_r4)
    assert canonicalize_cited_run_id(clipped_r4, R4_RUN_ID) == R4_RUN_ID


def test_truncated_nested_run_id_is_canonicalized_not_fail_closed():
    from scientist_lab.llm.reviewer_contract import parse_reviewer_completion

    ctx = _review_inputs(FIXTURE)
    payload = _contract_payload(ctx, run_id=R3_RUN_ID, locked="KEEP")
    raw = _proposal_json(
        run_id=R3_TRUNCATED,
        hypothesis_status="not_a_claim",
        interpretation=(
            "DecisionRubric already locked KEEP. KEEP is a next-round action, "
            "not ClaimGate SUPPORTED, and a single-seed delta is not a G2 claim."
        ),
        next_research_priority="Replicate the same protocol on another seed before any G2 claim.",
    )
    proposal = parse_reviewer_completion(raw, payload)
    assert proposal["run_id"] == R3_RUN_ID
    assert proposal["evidence_refs"][0]["run_id"] == R3_RUN_ID
    assert proposal["created_from"] == [R3_RUN_ID]
    assert proposal["hypothesis_status"] == "not_a_claim"


def test_foreign_run_id_still_fail_closed():
    from scientist_lab.llm.reviewer_contract import (
        ReviewerContractError,
        parse_reviewer_completion,
    )

    ctx = _review_inputs(FIXTURE)
    payload = _contract_payload(ctx, run_id=R3_RUN_ID, locked="KEEP")
    raw = _proposal_json(
        run_id="run_plan_round1_from_exec_31eff20c4e0c",
        hypothesis_status="not_a_claim",
        interpretation=(
            "DecisionRubric already locked KEEP. KEEP is a next-round action, "
            "not ClaimGate SUPPORTED."
        ),
        next_research_priority="Replicate on another seed.",
    )
    with pytest.raises(ReviewerContractError, match="evidence_refs must cite run_id"):
        parse_reviewer_completion(raw, payload)


def test_persist_pack_accepts_truncated_nested_run_id(tmp_path: Path, monkeypatch):
    dest = _copy_fixture(tmp_path)
    result_path = dest / "result.json"
    result = load_json(result_path)
    result["run_id"] = R3_RUN_ID
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raw = _proposal_json(run_id=R3_TRUNCATED)
    monkeypatch.setattr(
        "scientist_lab.llm.review_replay.resolve_gateway_provider",
        lambda **_kwargs: ScriptedProvider(raw),
    )
    report = run_llm_review_replay(dest, persist_pack=True)
    assert report["ok"] is True
    assert report["fail_closed"] is False
    sidecar = load_json(dest / "semantic_review.json")
    assert sidecar["run_id"] == R3_RUN_ID
    assert sidecar["semantic_proposal"]["evidence_refs"][0]["run_id"] == R3_RUN_ID
    assert sidecar["semantic_proposal"]["created_from"] == [R3_RUN_ID]
    assert (dest / "reviewer_live_fail_closed.json").is_file() is False
