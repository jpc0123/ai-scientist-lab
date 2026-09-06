"""Registered experiments: V26 is one built-in, new protocols get new ids. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.services.autonomous_campaign import (
    AutonomousCampaignError,
    AutonomousCampaignService,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.registered_experiments import (
    BUILTIN_EXPERIMENT_ID,
    RegisteredExperimentError,
    RegisteredExperimentService,
)
from scientist_lab.settings import Settings

EXAMPLES = SCHEMA_DIR / "examples"


def _alt_protocol() -> tuple[dict, dict]:
    proto = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    proto["protocol_id"] = "research_protocol_rgbt_dfine_alt_demo"
    proto["title"] = "Alt RGB-T detection protocol (registered separately)"
    plan = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    plan["protocol_id"] = proto["protocol_id"]
    plan["plan_id"] = "plan_alt_demo_f1"
    return proto, plan


def test_builtin_refreshes_when_schema_example_updates(tmp_path: Path) -> None:
    store = RegisteredExperimentService(tmp_path)
    store.ensure_builtin()
    protocol_path = tmp_path / ".run" / "experiments" / BUILTIN_EXPERIMENT_ID / "protocol.json"
    stale = load_json(protocol_path)
    stale["protocol_version"] = 1
    stale["stop_rules"] = dict(stale.get("stop_rules") or {})
    stale["stop_rules"]["max_rounds"] = 6
    protocol_path.write_text(json.dumps(stale, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    refreshed = store.ensure_builtin()
    canonical = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    on_disk = load_json(refreshed["protocol_path"])
    assert on_disk["protocol_version"] == canonical["protocol_version"]
    assert on_disk["stop_rules"]["max_rounds"] == canonical["stop_rules"]["max_rounds"]


def test_builtin_v26_is_one_experiment(tmp_path: Path) -> None:
    store = RegisteredExperimentService(tmp_path)
    catalog = store.list()
    ids = [row["experiment_id"] for row in catalog["items"]]
    assert BUILTIN_EXPERIMENT_ID in ids
    row = store.get(BUILTIN_EXPERIMENT_ID)
    assert row is not None
    assert row["builtin"] is True
    assert row["task_type"] == "object_detection"
    assert row["status"] == "ready"
    assert row["is_claim"] is False
    assert row["adapter"] == "dfine"


def test_start_without_experiment_id_is_refused(tmp_path: Path) -> None:
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    with pytest.raises(AutonomousCampaignError, match="experiment_id"):
        campaigns.start(confirm_human_gate=True, background=False)


def test_start_copies_bound_experiment_not_side_path(tmp_path: Path) -> None:
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    with pytest.raises(AutonomousCampaignError, match="side protocol"):
        campaigns.start(
            experiment_id=BUILTIN_EXPERIMENT_ID,
            confirm_human_gate=True,
            execute=False,
            background=False,
            protocol_path=EXAMPLES / "research_protocol_rgbt_dfine_v26.json",
            plan_path=EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json",
        )


def test_register_new_protocol_does_not_overwrite_builtin(tmp_path: Path) -> None:
    store = RegisteredExperimentService(tmp_path)
    store.ensure_builtin()
    proto, plan = _alt_protocol()
    created = store.register(
        protocol=proto,
        seed_plan=plan,
        experiment_id="exp_alt_rgbt",
        title="Alt detection experiment",
    )
    assert created["experiment_id"] == "exp_alt_rgbt"
    assert created["builtin"] is False
    assert store.get(BUILTIN_EXPERIMENT_ID)["protocol_id"] == "research_protocol_rgbt_dfine_v26"
    with pytest.raises(RegisteredExperimentError, match="overwrite"):
        store.register(protocol=proto, seed_plan=plan, experiment_id=BUILTIN_EXPERIMENT_ID)


def test_campaign_list_distinguishes_experiments(tmp_path: Path) -> None:
    proto, plan = _alt_protocol()
    campaigns = AutonomousCampaignService(project_root=tmp_path)
    campaigns.experiments.register(protocol=proto, seed_plan=plan, experiment_id="exp_alt_rgbt")
    for eid, title in (
        (BUILTIN_EXPERIMENT_ID, "builtin"),
        ("exp_alt_rgbt", "Alt detection experiment"),
    ):
        dest = campaigns.root / f"{eid}_demo"
        dest.mkdir(parents=True)
        (dest / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"{eid}_demo",
                    "experiment_id": eid,
                    "experiment_title": title,
                    "status": "paused",
                    "ok": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    listed = campaigns.list()
    by_id = {row["campaign_id"]: row for row in listed["items"]}
    assert by_id[f"{BUILTIN_EXPERIMENT_ID}_demo"]["experiment_id"] == BUILTIN_EXPERIMENT_ID
    assert by_id["exp_alt_rgbt_demo"]["experiment_id"] == "exp_alt_rgbt"
    registered = {row["experiment_id"] for row in listed["experiments"]}
    assert BUILTIN_EXPERIMENT_ID in registered
    assert "exp_alt_rgbt" in registered


def test_api_register_and_start_requires_experiment(tmp_path: Path) -> None:
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
    missing = client.post(
        "/api/v1/autonomous-campaigns",
        json={"confirm_human_gate": True, "execute": False},
    )
    assert missing.status_code in {400, 422}
    listed = client.get("/api/v1/registered-experiments")
    assert listed.status_code == 200
    ids = [row["experiment_id"] for row in (listed.json().get("items") or [])]
    assert BUILTIN_EXPERIMENT_ID in ids
    proto, plan = _alt_protocol()
    created = client.post(
        "/api/v1/registered-experiments",
        json={
            "protocol": proto,
            "seed_plan": plan,
            "experiment_id": "exp_api_alt",
            "title": "API alt experiment",
        },
    )
    assert created.status_code == 200, created.text
    assert created.json()["experiment_id"] == "exp_api_alt"
    got = client.get("/api/v1/registered-experiments/exp_api_alt")
    assert got.status_code == 200
    assert got.json()["protocol_id"] == proto["protocol_id"]
    assert got.json()["experiment_id"] != BUILTIN_EXPERIMENT_ID
    assert got.json()["protocol"]["protocol_id"] == proto["protocol_id"]
    assert got.json()["protocol_summary"]["max_rounds"] == proto["stop_rules"]["max_rounds"]
    assert got.json()["protocol_summary"]["allow_scientific_claims"] is False
