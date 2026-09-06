"""v2.6 V26.4 R0 baseline anchor. No GPU metrics invented."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.cli import main
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.datasets.low_light_subset import SLICE_ID
from scientist_lab.tasks.rgbt_detection.v26_r0 import (
    ANCHOR_ID,
    R0_HOW_ID,
    R0BaselineError,
    bind_metrics_from_run,
    build_r0_freeze,
    default_r0_plan,
    freeze_r0_anchor,
    metric_contract,
    r0_how,
    refuse_forged_metrics,
    run_r0,
)

EXAMPLES = SCHEMA_DIR / "examples"


def _contract() -> dict:
    return {
        "schema_version": "1.0.0",
        "dataset_id": "rgbt_tiny_v1",
        "dataset_ref": "dataset:rgbt_tiny_v1",
        "slice_id": SLICE_ID,
        "split_reference": f"{SLICE_ID}@6aa3cf0a444b0373239c81fec10443e8e067f38f71e4e41dcf57d4d888965c49",
        "fingerprint": "a" * 64,
        "version": "v1_gate_k2c_seq_expand",
        "read_only": True,
        "llm_may_rewrite": False,
        "probe": {"processed_present": True, "raw_present": False},
    }


def test_r0_how_is_f1_not_a4() -> None:
    how = r0_how()
    assert how["how_id"] == "F1"
    assert how["fusion_method"] == "early_concat"
    assert how["neck_type"] == "standard"
    assert "A4" in how["hidden_control"]


def test_r0_plan_and_protocol_validate() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    plan = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    validate_named("research_protocol", protocol)
    validate_named("experiment_plan", plan)
    assert protocol["objective"]["primary"]["metric"] == "APS_lowlight"
    assert protocol["condition_slice"]["id"] == SLICE_ID
    assert "condition_slice" in protocol["frozen_scope"]
    assert plan["how_id"] == R0_HOW_ID
    assert plan["budget_class"] == "formal"
    assert metric_contract()["primary"]["metric"] == "APS_lowlight"


def test_r0_materialize_binds_slice_and_f1() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    plan = default_r0_plan()
    validate_named("experiment_plan", plan)
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    assert contract["dataset"]["split_reference"].startswith(f"{SLICE_ID}@")
    how = (contract.get("materialization") or {}).get("how") or {}
    assert how.get("how_id") == R0_HOW_ID
    assert how.get("fusion_method") == "early_concat"
    assert how.get("neck_type") == "standard"
    assert contract["budget_class"] == "formal"


def test_r0_freeze_metrics_pending(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    plan = default_r0_plan()
    freeze = build_r0_freeze(
        protocol=protocol,
        plan=plan,
        dataset_contract=_contract(),
        doctor={"live_ready": False, "ok": True},
    )
    validate_named("r0_baseline", freeze)
    assert freeze["anchor_id"] == ANCHOR_ID
    assert freeze["status"] == "protocol_frozen_metrics_pending"
    assert freeze["metrics"]["APS_lowlight"] is None
    assert freeze["llm_may_rewrite"] is False
    dest = tmp_path / "R0.json"
    written = freeze_r0_anchor(
        dataset_contract=_contract(),
        doctor={"live_ready": False, "ok": True},
        output=dest,
        metric_contract_output=tmp_path / "metric_contract.json",
    )
    assert dest.is_file()
    assert written["metrics"]["APS_lowlight"] is None


def test_refuse_forged_and_probe_aps() -> None:
    refuse_forged_metrics(None)
    with pytest.raises(R0BaselineError):
        refuse_forged_metrics({"source": "v25d_probe", "APS_lowlight": 0.0})
    with pytest.raises(R0BaselineError):
        refuse_forged_metrics({"APS_lowlight": 0.12, "mAP50_95": 0.12})
    with pytest.raises(R0BaselineError):
        bind_metrics_from_run(Path("definitely-missing-r0-dir"))


def test_bind_metrics_requires_aps_lowlight(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({"mAP50_95": 0.14, "APS": 0.05}),
        encoding="utf-8",
    )
    with pytest.raises(R0BaselineError, match="APS_lowlight"):
        bind_metrics_from_run(tmp_path)
    (tmp_path / "metrics.json").write_text(
        json.dumps({"APS_lowlight": 0.031, "evaluator_backend": "pycocotools"}),
        encoding="utf-8",
    )
    bound = bind_metrics_from_run(tmp_path)
    assert bound["APS_lowlight"] == pytest.approx(0.031)


def test_run_r0_refuses_gpu_without_live_ready(tmp_path: Path) -> None:
    with pytest.raises(R0BaselineError, match="confirm-human-gate"):
        run_r0(
            dataset_contract=_contract(),
            doctor={"live_ready": True, "ok": True},
            output_dir=tmp_path / "out",
            execute=True,
            confirm_human_gate=False,
            require_live_ready=True,
        )
    with pytest.raises(R0BaselineError, match="live_ready"):
        run_r0(
            dataset_contract=_contract(),
            doctor={"live_ready": False, "ok": True},
            output_dir=tmp_path / "out",
            execute=True,
            confirm_human_gate=True,
            require_live_ready=True,
        )


def test_unregistered_how_still_rejected() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    plan = default_r0_plan()
    plan["how_id"] = "F2"
    plan["proposed_changes"][0]["detail"]["how_id"] = "F2"
    with pytest.raises(MaterializeRejected):
        DFINEAdapter().materialize_contract(plan, protocol)


def test_cli_freeze_v26_r0(tmp_path: Path) -> None:
    dest = tmp_path / "R0_BASELINE_FREEZE.json"
    code = main(["freeze-v26-r0", "--no-probe", "--output", str(dest)])
    assert code == 0
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["status"] == "protocol_frozen_metrics_pending"
    assert doc["how"]["how_id"] == "F1"
    assert doc["metrics"]["APS_lowlight"] is None
    assert doc["dataset"]["slice_id"] == SLICE_ID


def test_sync_aps_lowlight_mirrors_into_exec_metrics(tmp_path: Path) -> None:
    from scientist_lab.tasks.rgbt_detection.v26_r0 import (
        _live_exec_output_dirs,
        sync_aps_lowlight_to_dirs,
    )

    campaign = tmp_path / "campaign_run"
    exec_dir = tmp_path / "outputs" / "exec_abc"
    campaign.mkdir()
    exec_dir.mkdir(parents=True)
    (exec_dir / "metrics.json").write_text(
        json.dumps(
            {
                "primary_metric": "APS_lowlight",
                "metrics": {"mAP50": 0.01, "mAP50_95": 0.002},
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (campaign / "live_execution.json").write_text(
        json.dumps({"execution_id": "exec_abc", "output_directory": str(exec_dir)}),
        encoding="utf-8",
    )
    slice_eval = {
        "APS_lowlight": 0.0045,
        "mAP50_95_lowlight": 0.001,
        "AP50_lowlight": 0.003,
        "evaluator_backend": "pycocotools",
        "n_images_lowlight": 100,
    }
    targets = [campaign, *_live_exec_output_dirs(campaign)]
    synced = sync_aps_lowlight_to_dirs(slice_eval, *targets)
    assert str(exec_dir) in synced or any(Path(p) == exec_dir for p in synced)
    assert (exec_dir / "aps_lowlight.json").is_file()
    merged = json.loads((exec_dir / "metrics.json").read_text(encoding="utf-8"))
    assert merged["APS_lowlight"] == 0.0045
    assert merged["metrics"]["APS_lowlight"] == 0.0045
    assert merged["metrics"]["mAP50"] == 0.01
