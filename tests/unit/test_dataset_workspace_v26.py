"""v2.6 Dataset Workspace: registry/slice contracts. Not an Agent. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.cli import main
from scientist_lab.core.schema_registry import validate_named
from scientist_lab.datasets.low_light_subset import membership_hash, rule_hash
from scientist_lab.datasets.workspace import DatasetContractError, DatasetWorkspace
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

SLICE_ID = "low_light_subset_v1"
DATASET_ID = "rgbt_tiny_v1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _seed_workspace(root: Path) -> DatasetWorkspace:
    processed = root / "processed_ds"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "README.txt").write_text("toy processed root", encoding="utf-8")
    ws = DatasetWorkspace(root / "data")
    ws.ensure_layout()
    _write_json(
        ws.dataset_path(DATASET_ID),
        {
            "schema_version": "1.0.0",
            "dataset_id": DATASET_ID,
            "task": "rgbt_object_detection",
            "version": "toy_v1",
            "annotation_format": "coco",
            "split_reference": "official_split",
            "processed_root": str(processed),
            "raw_root": None,
            "splits": {"train": {"n_rgb": 3}, "val": {"n_rgb": 2}, "test": {"n_rgb": 2}},
            "manifest_hashes": {"train_pairs.csv": "a" * 64},
            "annotation_hashes": {"instances_train.json": "b" * 64},
            "adapters": ["dfine", "rtdetr", "yolo"],
            "read_only": True,
            "llm_may_rewrite": False,
            "llm_may_select": True,
        },
    )
    return ws


def _toy_freeze() -> dict:
    membership = {
        "train": ["dark_t1", "dark_t2"],
        "val": ["dark_v1"],
        "test": ["dark_e1"],
    }
    all_ids = [item for split in ("train", "val", "test") for item in membership[split]]
    return {
        "schema_version": "1.0.0",
        "slice_id": SLICE_ID,
        "official_labels": False,
        "llm_may_rewrite": False,
        "rule": {
            "slice_id": SLICE_ID,
            "version": "v1",
            "method": "rgb_mean_rec709_luminance_train_p25",
            "train_percentile": 25.0,
            "rec709": [0.2126, 0.7152, 0.0722],
            "official_labels": False,
            "threshold_split": "train",
            "dataset": f"dataset:{DATASET_ID}",
        },
        "rule_hash": rule_hash(),
        "threshold": 0.2,
        "membership": membership,
        "membership_hash": membership_hash(all_ids),
    }


@pytest.fixture()
def api_client(tmp_path: Path):
    root = tmp_path
    ws = _seed_workspace(root)
    service = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=root / "lab.db",
            outputs_dir=root / "outputs",
            runtime_dir=root / "runtime",
            data_workspace_dir=ws.root,
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    app = create_app(service=service)
    return TestClient(app), service, ws, root


def test_resolve_dataset_and_slice(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    spec = ws.import_slice_from_freeze(_toy_freeze())
    assert spec["frozen"] is True
    assert spec["llm_may_rewrite"] is False
    contract = ws.resolve(DATASET_ID, slice_id=SLICE_ID, project_root=tmp_path)
    validate_named("dataset_contract", contract)
    assert contract["read_only"] is True
    assert contract["llm_may_rewrite"] is False
    assert contract["fingerprint"]
    assert contract["split_reference"].startswith(f"{SLICE_ID}@")
    assert contract["slice"]["counts"]["train"] == 2
    assert "membership" not in contract["slice"]


def test_mutated_membership_is_rejected(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    ws.import_slice_from_freeze(_toy_freeze())
    ids_path = ws.slice_dir(SLICE_ID) / "train_ids.txt"
    ids_path.write_text(ids_path.read_text(encoding="utf-8") + "sneaky_id\n", encoding="utf-8")
    with pytest.raises(DatasetContractError, match="membership_hash mismatch"):
        ws.load_slice(SLICE_ID)


def test_rule_only_freeze_is_rejected(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    freeze = _toy_freeze()
    freeze["membership"] = {}
    freeze["membership_hash"] = None
    with pytest.raises(DatasetContractError, match="rule-only"):
        ws.import_slice_from_freeze(freeze)


def test_parent_mismatch_is_rejected(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    ws.import_slice_from_freeze(_toy_freeze())
    _write_json(
        ws.dataset_path("other_ds"),
        {
            "schema_version": "1.0.0",
            "dataset_id": "other_ds",
            "task": "rgbt_object_detection",
            "version": "toy",
            "annotation_format": "coco",
            "processed_root": str(tmp_path / "processed_ds"),
            "read_only": True,
            "llm_may_rewrite": False,
        },
    )
    with pytest.raises(DatasetContractError, match="parent_dataset"):
        ws.resolve("other_ds", slice_id=SLICE_ID, project_root=tmp_path)


def test_api_overview_and_slice_omits_membership(api_client) -> None:
    client, service, ws, _root = api_client
    ws.import_slice_from_freeze(_toy_freeze())

    listed = client.get("/api/v1/dataset-workspace")
    assert listed.status_code == 200
    body = listed.json()
    assert body["not_an_agent"] is True
    assert DATASET_ID in [row["dataset_id"] for row in body["datasets"]]
    assert SLICE_ID in [row["slice_id"] for row in body["slices"]]
    blob = json.dumps(body)
    assert "dark_t1" not in blob

    slice_resp = client.get(f"/api/v1/dataset-workspace/slices/{SLICE_ID}")
    assert slice_resp.status_code == 200
    spec = slice_resp.json()
    assert spec["membership_omitted"] is True
    assert "membership" not in spec
    assert "dark_t1" not in slice_resp.text

    resolved = client.post(
        "/api/v1/dataset-workspace/resolve",
        json={"dataset_id": DATASET_ID, "slice_id": SLICE_ID},
    )
    assert resolved.status_code == 200
    contract = resolved.json()
    validate_named("dataset_contract", contract)
    assert contract["sqlite_bound"] is False
    assert "dark_t1" not in resolved.text


def test_api_bind_enable_disable(api_client) -> None:
    client, _service, _ws, _root = api_client
    bound = client.post("/api/v1/dataset-workspace/bind", json={"dataset_id": DATASET_ID})
    assert bound.status_code == 200
    assert bound.json()["ok"] is True
    again = client.post("/api/v1/dataset-workspace/bind", json={"dataset_id": DATASET_ID})
    assert again.json()["already_bound"] is True

    disabled = client.post(f"/api/v1/dataset-workspace/datasets/{DATASET_ID}/disable")
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    enabled = client.post(f"/api/v1/dataset-workspace/datasets/{DATASET_ID}/enable")
    assert enabled.status_code == 200
    assert enabled.json()["enabled"] is True


def test_api_rebind_host_path(api_client, tmp_path: Path) -> None:
    client, service, _ws, _root = api_client
    first = client.post("/api/v1/dataset-workspace/bind", json={"dataset_id": DATASET_ID})
    assert first.status_code == 200
    assert first.json()["already_bound"] is False

    alt = tmp_path / "alt_processed"
    alt.mkdir(parents=True, exist_ok=True)
    (alt / "README.txt").write_text("alt mount", encoding="utf-8")

    blocked = client.post(
        "/api/v1/dataset-workspace/bind",
        json={"dataset_id": DATASET_ID, "host_path": str(alt)},
    )
    assert blocked.status_code == 400

    rebound = client.post(
        "/api/v1/dataset-workspace/bind",
        json={"dataset_id": DATASET_ID, "host_path": str(alt), "rebind": True},
    )
    assert rebound.status_code == 200
    body = rebound.json()
    assert body["ok"] is True
    assert body["rebound"] is True
    assert body["rules"]["path_rebind_allowed"] is True
    assert body["rules"]["dataset_id_change_requires_amendment"] is True
    assert Path(body["dataset"]["host_path"]).resolve() == alt.resolve()

    shown = service.show_dataset(DATASET_ID)
    assert Path(shown["host_path"]).resolve() == alt.resolve()


def test_cli_import_and_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    ws = _seed_workspace(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, _toy_freeze())
    monkeypatch.setenv("SCIENTIST_LAB_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SCIENTIST_LAB_DB_PATH", str(tmp_path / "lab.db"))
    monkeypatch.setenv("SCIENTIST_LAB_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("SCIENTIST_LAB_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("SCIENTIST_LAB_DATA_WORKSPACE_DIR", str(ws.root))
    monkeypatch.setenv("SCIENTIST_LAB_EXPERIMENT_APP_DIR", str(tmp_path / "app"))

    code = main(["dataset-import-slice", "--from-freeze", str(freeze_path)])
    assert code == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["ok"] is True
    assert imported["slice"]["slice_id"] == SLICE_ID

    code = main(["dataset-resolve", "--dataset-id", DATASET_ID, "--slice-id", SLICE_ID])
    assert code == 0
    contract = json.loads(capsys.readouterr().out)
    assert contract["dataset_id"] == DATASET_ID
    assert contract["split_reference"].startswith(f"{SLICE_ID}@")

    code = main(["dataset-workspace"])
    assert code == 0
    overview = json.loads(capsys.readouterr().out)
    assert overview["not_an_agent"] is True
    assert any(row["slice_id"] == SLICE_ID for row in overview["slices"])
