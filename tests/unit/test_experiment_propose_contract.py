"""Planner HOW unlock + LLM experiment propose contracts. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.how import list_adapter_capabilities
from scientist_lab.adapters.dfine.how_catalog import (
    ALLOWED_HOW,
    NOT_REGISTERED,
    catalog_payload,
    planner_visible_how_ids,
    resolve_how_id,
)
from scientist_lab.api.app import create_app
from scientist_lab.core.claim_gate import evaluate_claim
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.planner_contract import (
    PlannerContractInput,
    parse_planner_completion,
)
from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
from scientist_lab.services.experiment_propose import (
    materialize_protocol_from_draft,
    propose_experiment_draft,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.registered_experiments import (
    BUILTIN_EXPERIMENT_ID,
    RegisteredExperimentService,
    builtin_science_identity,
    science_identity,
)
from scientist_lab.settings import Settings

EXAMPLES = SCHEMA_DIR / "examples"


def _protocol_plan():
    return (
        load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json"),
        load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json"),
    )


def _payload(**selected):
    body = {
        "requested_module": selected.pop("requested_module", "fusion"),
        "hypothesis": selected.pop("hypothesis", "Probe an allowed Adapter HOW."),
        "proposed_changes": selected.pop(
            "proposed_changes",
            [{"target": "fusion", "summary": "Use catalog HOW."}],
        ),
        "how_id": selected.pop("how_id", "F0"),
        "observation": "Unlock test.",
        "expected_effect": {"primary_metric": "APS_lowlight", "direction": "increase"},
        "budget_class": "formal",
    }
    body.update(selected)
    module = body["requested_module"]
    for row in body["proposed_changes"]:
        row["target"] = module
    return json.dumps({"selected": body, "candidates": [], "invented_operators": []})


def _contract(protocol, plan) -> PlannerContractInput:
    return PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=plan,
        budget={"budget_class": str(plan.get("budget_class") or "formal")},
        adapter_capabilities=list_adapter_capabilities(),
    )


def test_live_planner_visible_set_is_all_materializable() -> None:
    visible = planner_visible_how_ids()
    assert visible == frozenset(ALLOWED_HOW)
    assert visible >= {"F0", "F1", "F3", "N0", "N1", "A4"}
    assert visible.isdisjoint(NOT_REGISTERED)
    how_ids = {row.get("how_id") for row in list_adapter_capabilities() if row.get("how_id")}
    assert how_ids == set(visible)
    assert catalog_payload()["hidden_from_planner"] == {}


def test_n1_and_a4_keep_selected_how_id_and_materialize() -> None:
    protocol, plan = _protocol_plan()
    contract = _contract(protocol, plan)
    for how_id, module in (("N1", "neck"), ("A4", "fusion")):
        mapped = parse_planner_completion(
            _payload(how_id=how_id, requested_module=module),
            contract,
        )
        assert mapped["how_id"] == how_id
        assert mapped["how_executable"] is True
        assert mapped["pending_unmaterializable"] is None
        how = DFINEAdapter()._resolve_how(  # noqa: SLF001
            {
                "how_id": how_id,
                "modification_scope": [module],
                "proposed_changes": mapped["proposed_changes"],
            },
            protocol,
        )
        assert how["how_id"] == how_id
        spec = resolve_how_id(how_id)
        assert how["fusion_method"] == spec["fusion_method"]
        assert how["neck_type"] == spec["neck_type"]


def test_builtin_fingerprint_clone_cannot_be_ready(tmp_path: Path) -> None:
    store = RegisteredExperimentService(tmp_path)
    proto, plan = _protocol_plan()
    proto = dict(proto)
    proto["protocol_id"] = "research_protocol_rgbt_dfine_clone"
    proto["title"] = "Idle clone"
    plan = dict(plan)
    plan["protocol_id"] = proto["protocol_id"]
    created = store.register(protocol=proto, seed_plan=plan, experiment_id="exp_idle_clone")
    assert created["status"] == "draft"
    assert created["experiment_id"] != BUILTIN_EXPERIMENT_ID
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    with pytest.raises(Exception, match="cannot start|idle|fingerprint"):
        campaigns.start(
            experiment_id="exp_idle_clone",
            confirm_human_gate=True,
            execute=False,
            background=False,
        )


def test_legal_different_draft_is_ready_and_start_binds_it(tmp_path: Path) -> None:
    preview = propose_experiment_draft(provider=FakeProvider(), live=False)
    assert preview["gpu"] is False
    assert preview["registered"] is False
    assert preview["is_claim"] is False
    assert preview["status_if_registered"] == "ready"
    assert preview["differs_from_builtin"] is True
    protocol = dict(preview["protocol"])
    plan = dict(preview["seed_plan"])
    assert protocol["protocol_id"] != "research_protocol_rgbt_dfine_v26"
    assert science_identity(protocol, plan) != builtin_science_identity()
    store = RegisteredExperimentService(tmp_path)
    created = store.register(
        protocol=protocol,
        seed_plan=plan,
        experiment_id=str(preview["experiment_id"]),
        title=str(preview["title"]),
    )
    assert created["status"] == "ready"
    assert created["experiment_id"] != BUILTIN_EXPERIMENT_ID
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    campaigns._worker = lambda campaign_id: None  # noqa: ARG005 — no GPU, copy-only
    started = campaigns.start(
        experiment_id=created["experiment_id"],
        confirm_human_gate=True,
        execute=False,
        background=False,
    )
    assert started["experiment_id"] == created["experiment_id"]
    bound = load_json(Path(started["protocol_path"]))
    assert bound["protocol_id"] == protocol["protocol_id"]
    assert bound["protocol_id"] != "research_protocol_rgbt_dfine_v26"


def test_unmaterializable_seed_how_cannot_start(tmp_path: Path) -> None:
    proto, plan = _protocol_plan()
    proto = dict(proto)
    proto["protocol_id"] = "research_protocol_f2_pending"
    proto["objective"] = {
        "primary": {"metric": "APS", "direction": "maximize"},
        "secondary": [],
    }
    proto.pop("condition_slice", None)
    plan = dict(plan)
    plan["protocol_id"] = proto["protocol_id"]
    plan["how_id"] = "F2"
    plan["proposed_changes"] = [
        {"target": "fusion", "summary": "LLM picked F2", "detail": {"how_id": "F2"}}
    ]
    store = RegisteredExperimentService(tmp_path)
    created = store.register(protocol=proto, seed_plan=plan, experiment_id="exp_f2_pending")
    assert created["status"] == "draft"
    assert created["seed_how_id"] == "F2"
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    with pytest.raises(Exception, match="F2|cannot start|不能物化"):
        campaigns.start(
            experiment_id="exp_f2_pending",
            confirm_human_gate=True,
            execute=False,
            background=False,
        )
    with pytest.raises(MaterializeRejected, match="F2"):
        resolve_how_id("F2")


def test_claim_gate_still_blocks_literature() -> None:
    verdict = evaluate_claim(
        {
            "schema_version": "1.0.0",
            "claim_id": "claim_lit_unlock",
            "claim_type": "observational",
            "claim_text": "Papers say fusion helps",
            "claim_strength": "C0",
            "metric": "APS",
        },
        evidence={
            "evidence_kind": "literature",
            "literature_query_id": "litq_unlock",
            "paper_refs": ["S2:lowlight-rgbt-001"],
            "metrics": {},
        },
    )
    assert verdict["status"] == "BLOCKED"
    assert "LiteratureEvidence" in verdict["reason"]
    assert verdict["keep_is_not_claim"] is True


def test_propose_does_not_rewrite_to_builtin_protocol() -> None:
    draft = {
        "research_question": "Does RGB-only F0 lag fusion on full val APS?",
        "dataset_id": "rgbt_tiny_v1",
        "slice_id": None,
        "primary_metric": "APS",
        "adapter_id": "dfine",
        "seed_how_id": "F0",
        "catalog_id": "how_catalog_v26_p2",
        "protocol_id": "research_protocol_rgbt_dfine_v26",
        "title": "Must not become builtin",
    }
    protocol, plan = materialize_protocol_from_draft(draft)
    assert protocol["protocol_id"] != "research_protocol_rgbt_dfine_v26"
    validate_named("research_protocol", protocol)
    validate_named("experiment_plan", plan)
    assert plan["how_id"] == "F0"
    assert protocol["protocol_id"] == plan["protocol_id"]


def test_api_propose_then_register_ready(tmp_path: Path) -> None:
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
    service._autonomous_campaigns = AutonomousCampaignService(project_root=tmp_path)
    client = TestClient(create_app(service=service))
    proposed = client.post("/api/v1/registered-experiments/propose", json={"live": False})
    assert proposed.status_code == 200, proposed.text
    body = proposed.json()
    assert body["gpu"] is False
    assert body["registered"] is False
    assert body["protocol"]["protocol_id"] != "research_protocol_rgbt_dfine_v26"
    created = client.post(
        "/api/v1/registered-experiments",
        json={
            "protocol": body["protocol"],
            "seed_plan": body["seed_plan"],
            "experiment_id": body["experiment_id"],
            "title": body["title"],
        },
    )
    assert created.status_code == 200, created.text
    row = created.json()
    assert row["status"] == "ready"
    assert row["experiment_id"] != BUILTIN_EXPERIMENT_ID
    listed = client.get("/api/v1/autonomous-campaigns")
    assert listed.status_code == 200
    assert all(not item.get("execute") for item in (listed.json().get("items") or []))


def test_fake_provider_experiment_protocol_contract() -> None:
    response = FakeProvider().complete(
        LLMRequest(
            purpose="other",
            messages=[{"role": "user", "content": "{}"}],
            metadata={"planner_contract": "experiment_protocol"},
            response_schema={
                "type": "object",
                "required": [
                    "research_question",
                    "dataset_id",
                    "primary_metric",
                    "adapter_id",
                    "seed_how_id",
                ],
            },
        )
    )
    assert response.schema_valid is True
    parsed = dict(response.parsed_json or {})
    assert parsed["seed_how_id"] == "F0"
    assert parsed["primary_metric"] == "APS"
    assert parsed["slice_id"] is None
