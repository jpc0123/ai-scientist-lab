"""HOW candidate freeze: LLM drafts, humans register. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.how_catalog import resolve_how_id
from scientist_lab.api.app import create_app
from scientist_lab.core.how_pending import (
    STATUS_PENDING_ADAPTER,
    STATUS_PROPOSED,
    decide_candidate,
    ingest_llm_candidates,
    literature_paper_ids,
    literature_query_id,
    load_store,
    normalize_llm_candidates,
    pending_path,
    resolve_pending_how,
    scout_literature_for_how,
)
from scientist_lab.core.planner import PlanRefused, Planner
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.llm.gateway import ScriptedProvider
from scientist_lab.llm.planner_contract import PlannerContractError
from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

from tests.unit.test_llm_planner_v25a import _replay_discard, _selected_json

EXAMPLES = SCHEMA_DIR / "examples"


def _f2_draft(qid: str, papers: list[str], **overrides) -> dict:
    row = {
        "how_id": "F2",
        "family": "fusion",
        "mechanism": "Paper reports gated RGB-T weighting; suggest a catalog draft F2.",
        "literature_query_id": qid,
        "paper_refs": papers,
        "invented_operators": [],
        "fusion_method": "weighted_fusion",
        "needs_adapter_work": True,
    }
    row.update(overrides)
    return row


def _packet(tmp_path: Path) -> dict:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    packet = scout_literature_for_how(
        protocol=protocol,
        live=False,
        round_id="round_how_freeze",
        provenance_dir=tmp_path / "literature",
    )
    assert packet.get("fail_closed") is not True
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    assert qid
    assert "S2:lowlight-rgbt-001" in papers or papers
    return packet


def test_llm_f2_draft_lands_proposed_not_catalog(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [_f2_draft(qid, papers[:1])],
        literature=packet,
        round_id="r1",
    )
    rows = store["candidates"]
    assert len(rows) == 1
    assert rows[0]["how_id"] == "F2"
    assert rows[0]["status"] == STATUS_PROPOSED
    assert rows[0]["can_enter_claim_gate"] is False
    with pytest.raises(MaterializeRejected, match="F2"):
        resolve_how_id("F2")
    with pytest.raises(MaterializeRejected, match="F2"):
        resolve_pending_how("F2", path)


def test_human_register_without_adapter_stays_pending(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [_f2_draft(qid, papers[:1])],
        literature=packet,
    )
    cid = store["candidates"][0]["candidate_id"]
    decided = decide_candidate(
        path,
        cid,
        decision="register",
        confirm_human_gate=True,
        note="method is plausible; Adapter has no weighted_fusion",
    )
    row = decided["candidates"][0]
    assert row["status"] == STATUS_PENDING_ADAPTER
    assert row["needs_adapter_work"] is True
    assert "F2" not in (decided.get("registered_overlay") or {})
    with pytest.raises(MaterializeRejected):
        resolve_pending_how("F2", path)
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")


def test_no_provenance_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(PlannerContractError, match="literature_query_id"):
        normalize_llm_candidates(
            [_f2_draft("litq_forged", ["S2:lowlight-rgbt-001"])],
            literature={"fail_closed": False, "planner_admissible": []},
        )


def test_invented_operators_fail_closed(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    with pytest.raises(PlannerContractError, match="invented operators"):
        normalize_llm_candidates(
            [_f2_draft(qid, papers[:1], invented_operators=["weighted_fusion_op"])],
            literature=packet,
        )


def test_python_in_mechanism_fail_closed(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    with pytest.raises(PlannerContractError, match="Python"):
        normalize_llm_candidates(
            [_f2_draft(qid, papers[:1], mechanism="def weighted_fusion(x): return x")],
            literature=packet,
        )


def test_planner_writes_pending_keeps_selected_registered(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    report, writer, protocol, previous = _replay_discard(tmp_path)
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
        how_candidates=[_f2_draft(qid, papers[:1])],
    )
    store = tmp_path / "campaign" / "how_pending.json"
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(raw),
        pending_store=store,
        literature_packet=packet,
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
    assert saved["candidates"]
    assert saved["candidates"][0]["how_id"] == "F2"
    assert saved["candidates"][0]["status"] == STATUS_PROPOSED
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")


def test_selected_f2_goes_pending_not_gpu(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    report, writer, protocol, previous = _replay_discard(tmp_path)
    raw = _selected_json(
        module="fusion",
        selected_updates={"how_id": "F2"},
        candidates=[
            {
                "candidate_id": "cand_alt_a",
                "requested_module": "neck",
                "how_id": "N0",
                "reason_not_selected": "discarded",
            },
            {
                "candidate_id": "cand_alt_b",
                "requested_module": "fusion",
                "how_id": "F0",
                "reason_not_selected": "not selected",
            },
        ],
    )
    store = tmp_path / "how_pending.json"
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(raw),
        pending_store=store,
        literature_packet=packet,
    )
    with pytest.raises(PlanRefused, match="not materializable"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )
    saved = load_store(store)
    assert saved["candidates"]
    assert saved["candidates"][0]["how_id"] == "F2"
    assert saved["candidates"][0]["status"] == STATUS_PROPOSED
    assert saved["candidates"][0]["from_selected"] is True
    assert saved["can_enter_claim_gate"] is False
    with pytest.raises(MaterializeRejected, match="F2"):
        resolve_how_id("F2")


def test_api_register_without_adapter_mapping(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    root = Path(__file__).resolve().parents[2]
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
    work = campaigns.root / "p0_how"
    work.mkdir(parents=True)
    (work / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": "p0_how",
                "status": "paused",
                "ok": True,
                "execute": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ingest_llm_candidates(
        pending_path(work),
        [_f2_draft(qid, papers[:1])],
        literature=packet,
    )
    cid = load_store(pending_path(work))["candidates"][0]["candidate_id"]
    service._autonomous_campaigns = campaigns
    client = TestClient(create_app(service=service))
    refused = client.post(
        f"/api/v1/autonomous-campaigns/p0_how/how-candidates/{cid}/decide",
        json={"decision": "register", "confirm_human_gate": False},
    )
    assert refused.status_code == 400
    ok = client.post(
        f"/api/v1/autonomous-campaigns/p0_how/how-candidates/{cid}/decide",
        json={"decision": "register", "confirm_human_gate": True},
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    rows = (body.get("how_pending") or {}).get("candidates") or []
    assert rows[0]["status"] == STATUS_PENDING_ADAPTER
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")
