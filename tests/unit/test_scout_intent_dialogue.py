"""Campaign scout-intent dialogue. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.core.how_pending import (
    fallback_scout_query,
    load_store,
    pending_path,
    persist_scout,
    resolve_scout_query,
    scout_literature_for_how,
    set_scout_intent,
)
from scientist_lab.core.planner import PlanRefused, Planner
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.core.scout_dialogue import handle_scout_chat
from scientist_lab.llm.gateway import ScriptedProvider
from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

from tests.unit.test_llm_planner_v25a import _replay_discard, _selected_json

EXAMPLES = SCHEMA_DIR / "examples"


def _campaign(tmp_path: Path, campaign_id: str = "p0_scout") -> Path:
    work = tmp_path / ".run" / "autonomous" / campaign_id
    work.mkdir(parents=True)
    (work / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "status": "paused",
                "ok": True,
                "execute": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return work


def test_human_query_beats_fallback(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    result = handle_scout_chat(path, "往低光模态加权查，不要 neck")
    assert result["refused"] is False
    assert result["action"] == "human_set"
    resolved = resolve_scout_query(load_store(path), protocol)
    assert resolved["source"] == "human"
    assert resolved["fallback"] is False
    assert "weighting" in resolved["query"].lower() or "加权" in resolved["query"]
    assert resolved["query"] != fallback_scout_query(protocol)


def test_llm_proposal_stays_inactive_until_accept(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    handle_scout_chat(
        path,
        "按证据找方向",
        evidence={"last_how_id": "F1", "last_review_decision": "DISCARD"},
        protocol=protocol,
    )
    store = load_store(path)
    assert store["scout_intent"]["status"] == "proposed"
    assert store["scout_intent"]["source"] == "llm"
    resolved = resolve_scout_query(store, protocol)
    assert resolved["source"] == "fallback"
    assert resolved["fallback"] is True
    accepted = handle_scout_chat(path, "接受这个查询")
    assert accepted["refused"] is False
    resolved = resolve_scout_query(load_store(path), protocol)
    assert resolved["source"] == "llm"
    assert resolved["fallback"] is False
    query = resolved["query"].lower()
    assert "f1" in query or "rgb-t" in query or "aps" in query or "small-object" in query
    assert "gated weighting" not in query or "f1" in query


def test_propose_does_not_clobber_human(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    handle_scout_chat(path, "search RGB-T gated weighting low-light")
    held = handle_scout_chat(path, "按证据找方向")
    assert held["action"] == "human_holds"
    resolved = resolve_scout_query(load_store(path), None)
    assert resolved["source"] == "human"
    assert "gated weighting" in resolved["query"]


def test_human_overrides_llm_proposal(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    handle_scout_chat(path, "按证据找方向", evidence={"last_how_id": "F3"})
    handle_scout_chat(path, "search RGB-T gated weighting low-light not neck")
    resolved = resolve_scout_query(load_store(path), None)
    assert resolved["source"] == "human"
    assert "gated weighting" in resolved["query"]


def test_chat_cannot_register_how_or_start_gpu(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    handle_scout_chat(path, "往低光模态加权查")
    before = load_store(path)["scout_intent"]
    refused = handle_scout_chat(path, "批准进目录 F2")
    assert refused["refused"] is True
    assert load_store(path)["scout_intent"]["query"] == before["query"]
    gpu = handle_scout_chat(path, "开始训练并点 GPU")
    assert gpu["refused"] is True
    claim = handle_scout_chat(path, "这是 Claim，可以声称了")
    assert claim["refused"] is True
    assert load_store(path)["scout_intent"]["can_enter_claim_gate"] is False


def test_planner_scout_uses_human_query(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    store = tmp_path / "campaign" / "how_pending.json"
    set_scout_intent(
        store,
        source="human",
        query="RGB-T gated weighting low-light",
        why="human direction",
        status="active",
        drafted_by="human",
    )
    extras = [
        {
            "candidate_id": "cand_alt_a",
            "requested_module": "neck",
            "how_id": "N0",
            "reason_not_selected": "Neck already discarded.",
        },
        {
            "candidate_id": "cand_alt_b",
            "requested_module": "fusion",
            "how_id": "F0",
            "reason_not_selected": "RGB-only control not selected.",
        },
    ]
    raw = _selected_json(
        module="fusion",
        selected_updates={"how_id": "F3"},
        candidates=extras,
        how_candidates=[],
    )
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(raw),
        pending_store=store,
    )
    result = planner.next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
    )
    assert result.plan.get("how_id") == "F3"
    saved = load_store(store)
    assert saved["scout"]["query"] == "RGB-T gated weighting low-light"
    assert saved["scout"]["source"] == "human"
    assert saved["scout"]["intent_source"] == "human"
    assert saved["scout"]["can_enter_claim_gate"] is False
    assert saved["scout"]["fail_closed"] is False
    assert saved["scout"]["live"] is False
    assert saved["scout"]["papers"]
    assert saved["scout"]["literature_query_id"]
    assert all(row.get("title") and row.get("paper_id") for row in saved["scout"]["papers"])


def test_api_scout_intent_chat(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    work = _campaign(tmp_path)
    service = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    service._autonomous_campaigns = campaigns
    client = TestClient(create_app(service=service))
    chat = client.post(
        "/api/v1/autonomous-campaigns/p0_scout/scout-intent/chat",
        json={"message": "按证据找方向"},
    )
    assert chat.status_code == 200, chat.text
    body = chat.json()
    intent = (body.get("how_pending") or {}).get("scout_intent") or {}
    assert intent.get("source") == "llm"
    assert intent.get("status") == "proposed"
    assert intent.get("can_enter_claim_gate") is False
    accept = client.post(
        "/api/v1/autonomous-campaigns/p0_scout/scout-intent",
        json={"action": "accept"},
    )
    assert accept.status_code == 200, accept.text
    active = (accept.json().get("how_pending") or {}).get("scout_intent") or {}
    assert active.get("status") == "active"
    assert active.get("source") == "llm"
    human = client.post(
        "/api/v1/autonomous-campaigns/p0_scout/scout-intent/chat",
        json={"message": "往低光模态加权查，不要 neck"},
    )
    assert human.status_code == 200, human.text
    held = (human.json().get("how_pending") or {}).get("scout_intent") or {}
    assert held.get("source") == "human"
    assert held.get("status") == "active"
    _ = work


def test_planner_persists_scout_when_llm_fails(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    store = tmp_path / "campaign" / "how_pending.json"
    set_scout_intent(
        store,
        source="human",
        query="RGB-T gated weighting low-light",
        why="human direction",
        status="active",
        drafted_by="human",
    )
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider("not-valid-planner-json"),
        pending_store=store,
    )
    with pytest.raises(PlanRefused):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )
    saved = load_store(store)
    scout = saved["scout"]
    assert scout["query"] == "RGB-T gated weighting low-light"
    assert scout["source"] == "human"
    assert scout["intent_source"] == "human"
    assert scout["fail_closed"] is False
    assert scout["papers"]
    assert scout["can_enter_claim_gate"] is False


def test_live_without_key_fail_closed_no_forged_papers(tmp_path: Path) -> None:
    packet = scout_literature_for_how(
        protocol=load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json"),
        live=True,
        round_id="round_no_key",
        provenance_dir=tmp_path / "literature",
        query="RGB-T gated weighting low-light",
        environ={},
    )
    assert packet["fail_closed"] is True
    assert packet["actual_search"] is False
    assert packet["papers"] == []
    assert packet["planner_admissible"] == []
    assert "没有实际检索" in str(packet.get("error") or "")
    store = persist_scout(tmp_path / "how_pending.json", packet)
    scout = store["scout"]
    assert scout["fail_closed"] is True
    assert scout["papers"] == []
    assert scout["query"] == "RGB-T gated weighting low-light"
    assert "source" in scout
    assert "literature_query_id" in scout
    assert "没有实际检索" in str(scout.get("error") or "")
    assert scout["can_enter_claim_gate"] is False


def test_api_get_exposes_scout_papers_and_fail_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    work = _campaign(tmp_path, "p0_scout_get")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    packet = scout_literature_for_how(
        protocol=protocol,
        live=False,
        round_id="round_ui",
        provenance_dir=work / "literature",
        query="RGB-T gated weighting low-light",
    )
    packet["intent_source"] = "human"
    persist_scout(pending_path(work), packet)
    service = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    service._autonomous_campaigns = campaigns
    client = TestClient(create_app(service=service))
    got = client.get("/api/v1/autonomous-campaigns/p0_scout_get")
    assert got.status_code == 200, got.text
    scout = (got.json().get("how_pending") or {}).get("scout") or {}
    assert scout.get("query") == "RGB-T gated weighting low-light"
    assert scout.get("source") == "human"
    assert scout.get("intent_source") == "human"
    assert scout.get("fail_closed") is False
    assert scout.get("papers")
    assert scout.get("literature_query_id")
    assert scout.get("can_enter_claim_gate") is False
    assert "error" in scout

    closed = scout_literature_for_how(
        protocol=protocol,
        live=True,
        round_id="round_ui_live",
        provenance_dir=work / "literature",
        query="RGB-T gated weighting low-light",
        environ={},
    )
    persist_scout(pending_path(work), {**closed, "source": "human", "intent_source": "human"})
    live_got = client.get("/api/v1/autonomous-campaigns/p0_scout_get")
    live_scout = (live_got.json().get("how_pending") or {}).get("scout") or {}
    assert live_scout.get("query") == "RGB-T gated weighting low-light"
    assert live_scout.get("source") == "human"
    assert live_scout.get("fail_closed") is True
    assert live_scout.get("papers") == []
    assert "literature_query_id" in live_scout
    assert "没有实际检索" in str(live_scout.get("error") or "")
    assert live_scout.get("can_enter_claim_gate") is False


def test_rules_planner_scouts_after_pending_store(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    store = tmp_path / "campaign" / "how_pending.json"
    set_scout_intent(
        store,
        source="human",
        query="RGB-T gated weighting low-light",
        why="human direction",
        status="active",
        drafted_by="human",
    )
    planner = Planner(backend="rules", pending_store=store, live=False)
    planner.next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
    )
    scout = load_store(store)["scout"]
    assert scout["query"] == "RGB-T gated weighting low-light"
    assert scout["source"] == "human"
    assert scout["intent_source"] == "human"
    assert scout["papers"]
    assert scout["fail_closed"] is False
    assert scout["can_enter_claim_gate"] is False


def test_planner_live_without_key_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    store = tmp_path / "campaign" / "how_pending.json"
    set_scout_intent(
        store,
        source="human",
        query="RGB-T gated weighting low-light",
        why="human direction",
        status="active",
        drafted_by="human",
    )
    planner = Planner(
        backend="rules",
        pending_store=store,
        live=True,
        literature_environ={},
    )
    planner.next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
    )
    scout = load_store(store)["scout"]
    assert scout["query"] == "RGB-T gated weighting low-light"
    assert scout["source"] == "human"
    assert scout["fail_closed"] is True
    assert scout["actual_search"] is False
    assert scout["papers"] == []
    assert "literature_query_id" in scout
    assert "没有实际检索" in str(scout.get("error") or "")
    assert scout["can_enter_claim_gate"] is False


def test_llm_search_runs_retriever_immediately(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    result = handle_scout_chat(
        path,
        "现在就检索",
        protocol=protocol,
        live=False,
        locale="en",
        provenance_dir=tmp_path / "literature",
    )
    assert result["action"] == "llm_search"
    assert result["refused"] is False
    papers = (result.get("scout") or {}).get("papers") or []
    assert papers
    assert papers[0]["url"]
    assert papers[0]["rank"] == 1
    assert papers[0].get("title_en")
    store = load_store(path)
    assert store["scout_intent"]["status"] == "active"
    assert store["scout"]["can_enter_claim_gate"] is False
    assert store["scout"]["papers"][0]["paper_id"] == papers[0]["paper_id"]


def test_llm_search_does_not_clobber_human_query(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    human_q = "RGB-T gated weighting low-light not neck"
    set_scout_intent(
        path,
        source="human",
        query=human_q,
        why="human direction",
        status="active",
        drafted_by="human",
    )
    result = handle_scout_chat(
        path,
        "你来检索",
        protocol=protocol,
        live=False,
        action="llm_search",
        provenance_dir=tmp_path / "literature",
    )
    assert result["human_holds"] is True
    store = load_store(path)
    assert store["scout_intent"]["source"] == "human"
    assert store["scout_intent"]["query"] == human_q
    assert store["scout"]["papers"]
    assert store["scout"]["can_enter_claim_gate"] is False


def test_llm_search_live_no_key_does_not_invent(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    result = handle_scout_chat(
        path,
        "现在就检索",
        protocol=protocol,
        live=True,
        locale="zh",
        provenance_dir=tmp_path / "literature",
        environ={},
    )
    assert result["action"] == "llm_search"
    scout = result["scout"]
    assert scout["fail_closed"] is True
    assert scout["actual_search"] is False
    assert scout["papers"] == []
    assert scout["can_enter_claim_gate"] is False


def test_api_llm_search_and_scout_display(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    work = _campaign(tmp_path)
    (work / "protocol.json").write_text(
        (EXAMPLES / "research_protocol_rgbt_dfine_v26.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    service = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    service._autonomous_campaigns = campaigns
    client = TestClient(create_app(service=service))
    search = client.post(
        "/api/v1/autonomous-campaigns/p0_scout/scout-intent/chat",
        json={"action": "llm_search", "live": False, "locale": "en"},
    )
    assert search.status_code == 200, search.text
    scout = (search.json().get("how_pending") or {}).get("scout") or {}
    assert scout.get("papers")
    assert scout["papers"][0]["url"]
    assert search.json().get("scout_action") == "llm_search"
    display = client.post(
        "/api/v1/autonomous-campaigns/p0_scout/scout-display",
        json={"locale": "zh", "live": False},
    )
    assert display.status_code == 200, display.text
    shown = (display.json().get("how_pending") or {}).get("scout") or {}
    assert shown["papers"][0]["paper_id"] == scout["papers"][0]["paper_id"]
    assert shown["papers"][0]["url"] == scout["papers"][0]["url"]


def test_library_search_reads_ledger_not_web(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    seed = scout_literature_for_how(
        protocol=protocol,
        live=False,
        round_id="seed_lib",
        provenance_dir=tmp_path / "literature_seed",
        query="brightness-bucket APS RGBT-Tiny",
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
    )
    persist_scout(path, seed)
    human_q = "RGB-T gated weighting low-light not neck"
    set_scout_intent(
        path,
        source="human",
        query=human_q,
        why="human direction",
        status="active",
        drafted_by="human",
    )
    result = handle_scout_chat(
        path,
        "查论文库",
        protocol=protocol,
        live=False,
        action="library_search",
    )
    assert result["action"] == "library_search"
    papers = (result.get("scout") or {}).get("papers") or []
    assert papers
    assert papers[0]["url"]
    assert papers[0]["retrieval_source"] == "library"
    scout = load_store(path)["scout"]
    assert scout["library_only"] is True
    assert scout["actual_search"] is False
    assert scout["fail_closed"] is False
    assert load_store(path)["scout_intent"]["query"] == human_q
    assert scout["can_enter_claim_gate"] is False


def test_library_search_empty_does_not_invent(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    result = handle_scout_chat(
        path,
        "查论文库",
        protocol=protocol,
        live=False,
    )
    assert result["action"] == "library_search"
    scout = result["scout"]
    assert scout["papers"] == []
    assert scout["library_only"] is True
    assert scout["actual_search"] is False
    assert "文献库为空" in str(scout.get("error") or scout.get("note") or "")
    assert scout["can_enter_claim_gate"] is False


def test_llm_search_no_key_does_not_use_library(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    seed = scout_literature_for_how(
        protocol=protocol,
        live=False,
        round_id="seed_not_mixed",
        provenance_dir=tmp_path / "literature_seed2",
        query="brightness-bucket APS RGBT-Tiny",
    )
    persist_scout(path, seed)
    assert load_store(path)["literature_ledger"]
    result = handle_scout_chat(
        path,
        "现在就检索",
        protocol=protocol,
        live=True,
        locale="zh",
        provenance_dir=tmp_path / "literature_llm",
        environ={},
        action="llm_search",
    )
    assert result["action"] == "llm_search"
    scout = result["scout"]
    assert scout["fail_closed"] is True
    assert scout["actual_search"] is False
    assert scout["papers"] == []
    assert not scout.get("library_only")
    assert scout["can_enter_claim_gate"] is False


def test_api_library_search(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    work = _campaign(tmp_path)
    (work / "protocol.json").write_text(
        (EXAMPLES / "research_protocol_rgbt_dfine_v26.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    service = ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    service._autonomous_campaigns = campaigns
    client = TestClient(create_app(service=service))
    empty = client.post(
        "/api/v1/autonomous-campaigns/p0_scout/scout-intent/chat",
        json={"action": "library_search", "live": False, "locale": "zh"},
    )
    assert empty.status_code == 200, empty.text
    scout = (empty.json().get("how_pending") or {}).get("scout") or {}
    assert scout.get("papers") == []
    assert empty.json().get("scout_action") == "library_search"
    assert scout.get("library_only") is True

