"""Unit tests for scientist-exec reattach / harvest (no new GPU job)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scientist_lab.runners.exec_reattach import (
    LiveExecBinding,
    contract_identity,
    execution_id_from_container_name,
    harvest_binding,
    identities_match,
    load_contract_identity,
    metrics_ready,
    pick_binding_for_harvest,
    try_harvest_existing,
    write_execution_sidecar,
)


def test_execution_id_from_container_name() -> None:
    assert (
        execution_id_from_container_name("scientist-exec-fb1a7078f954")
        == "exec_fb1a7078f954"
    )
    assert execution_id_from_container_name("/scientist-exec-abc") == "exec_abc"


def test_write_execution_sidecar_and_metrics_ready(tmp_path: Path) -> None:
    out = tmp_path / "exec_x"
    write_execution_sidecar(
        out,
        execution_id="exec_x",
        container_name="scientist-exec-x",
        return_code=0,
    )
    assert (out / "execution.json").is_file()
    assert (out / "live_execution.json").is_file()
    assert metrics_ready(out) is False
    (out / "metrics.json").write_text("{}", encoding="utf-8")
    assert metrics_ready(out) is True


def test_pick_binding_prefers_running() -> None:
    a = LiveExecBinding("scientist-exec-a", "exec_a", Path("a"), "exited", 0)
    b = LiveExecBinding("scientist-exec-b", "exec_b", Path("b"), "running", None)
    assert pick_binding_for_harvest([a, b]) is b


def test_pick_binding_filters_by_expected_node(tmp_path: Path) -> None:
    out_a = tmp_path / "exec_a"
    out_b = tmp_path / "exec_b"
    out_a.mkdir()
    out_b.mkdir()
    (out_a / "contract.json").write_text(
        json.dumps({"node_id": "run_old", "seed": 1, "parameters": {"fusion_method": "none"}}),
        encoding="utf-8",
    )
    (out_b / "contract.json").write_text(
        json.dumps({"node_id": "run_new", "seed": 2, "parameters": {"fusion_method": "gated_multiscale"}}),
        encoding="utf-8",
    )
    a = LiveExecBinding("scientist-exec-a", "exec_a", out_a, "exited", 0)
    b = LiveExecBinding("scientist-exec-b", "exec_b", out_b, "exited", 0)
    picked = pick_binding_for_harvest(
        [a, b],
        expected={"node_id": "run_new"},
    )
    assert picked is b
    assert pick_binding_for_harvest([a, b], expected={"node_id": "missing"}) is None


def test_harvest_binding_exited_with_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "exec_z"
    out.mkdir()
    (out / "metrics.json").write_text(
        json.dumps({"APS_lowlight": 0.01}), encoding="utf-8"
    )
    binding = LiveExecBinding(
        container_name="scientist-exec-z",
        execution_id="exec_z",
        output_dir=out,
        status="exited",
        exit_code=0,
    )

    def _no_wait(name: str, **kwargs: Any) -> int:
        raise AssertionError("should not wait on exited")

    monkeypatch.setattr(
        "scientist_lab.runners.exec_reattach.wait_existing_container",
        _no_wait,
    )
    payload = harvest_binding(binding, wait=True)
    assert payload["reattached"] is True
    assert payload["status"] == "completed"
    assert payload["run"]["output_directory"] == str(out)
    assert (out / "execution.json").is_file()


def test_try_harvest_existing_from_dest_artifacts_requires_match(tmp_path: Path) -> None:
    dest = tmp_path / "run"
    dest.mkdir()
    (dest / "metrics.json").write_text("{}", encoding="utf-8")
    (dest / "execution.json").write_text("{}", encoding="utf-8")
    (dest / "_legacy_fast_eval_contract.json").write_text(
        json.dumps({"node_id": "run_this", "seed": 42, "parameters": {"fusion_method": "none"}}),
        encoding="utf-8",
    )
    # Wrong round leftovers must not be harvested.
    assert (
        try_harvest_existing(
            outputs_root=tmp_path / "outputs",
            dest=dest,
            wait=False,
            expected={"node_id": "run_other"},
        )
        is None
    )
    payload = try_harvest_existing(
        outputs_root=tmp_path / "outputs",
        dest=dest,
        wait=False,
        expected={"node_id": "run_this"},
    )
    assert payload is not None
    assert payload["reattached"] is True
    assert payload["run"]["source"] == "dest_artifacts"


def test_try_harvest_existing_does_not_blind_scan_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs" / "project_rgbt_cuda_001" / "exec_old"
    outputs.mkdir(parents=True)
    (outputs / "metrics.json").write_text(json.dumps({"APS": 0.1}), encoding="utf-8")
    (outputs / "contract.json").write_text(
        json.dumps({"node_id": "run_old", "seed": 1, "parameters": {"fusion_method": "none"}}),
        encoding="utf-8",
    )
    dest = tmp_path / "campaign_run"
    dest.mkdir()
    # New round must start a GPU job, not copy newest unrelated metrics.
    assert (
        try_harvest_existing(
            outputs_root=tmp_path / "outputs",
            dest=dest,
            project_id="project_rgbt_cuda_001",
            wait=False,
            expected={"node_id": "run_new"},
        )
        is None
    )


def test_try_harvest_existing_matches_binding_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = tmp_path / "outputs" / "project_rgbt_cuda_001" / "exec_match"
    outputs.mkdir(parents=True)
    (outputs / "metrics.json").write_text(json.dumps({"APS_lowlight": 0.02}), encoding="utf-8")
    (outputs / "contract.json").write_text(
        json.dumps(
            {
                "node_id": "run_match",
                "seed": 44,
                "parameters": {"fusion_method": "gated_multiscale", "input_mode": "rgbt", "neck": {"type": "standard"}},
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "run"
    dest.mkdir()

    binding = LiveExecBinding(
        "scientist-exec-match",
        "exec_match",
        outputs,
        "exited",
        0,
    )
    monkeypatch.setattr(
        "scientist_lab.runners.exec_reattach.discover_live_exec_bindings",
        lambda *a, **k: [binding],
    )
    payload = try_harvest_existing(
        outputs_root=tmp_path / "outputs",
        dest=dest,
        project_id="project_rgbt_cuda_001",
        wait=False,
        expected={"node_id": "run_match"},
    )
    assert payload is not None
    assert payload["run"]["execution_id"] == "exec_match"
    assert (dest / "metrics.json").is_file()


def test_identities_match_node_id() -> None:
    assert identities_match({"node_id": "a"}, {"node_id": "a"})
    assert not identities_match({"node_id": "a"}, {"node_id": "b"})
    assert identities_match(
        {"seed": 1, "fusion_method": "none", "neck_type": "standard"},
        {"seed": 1, "fusion_method": "none", "neck_type": "standard"},
    )


def test_identities_match_rejects_same_node_wrong_fusion() -> None:
    """plan P2 must not harvest a same-node P3A GPU artifact."""
    expected = {
        "node_id": "run_plan_r9_25fe24b137",
        "fusion_method": "plugin:P2",
        "neck_type": "standard",
        "input_mode": "rgbt",
    }
    actual_p3a = {
        "node_id": "run_plan_r9_25fe24b137",
        "fusion_method": "plugin:P3A",
        "neck_type": "standard",
        "input_mode": "rgbt",
    }
    actual_p2 = {
        "node_id": "run_plan_r9_25fe24b137",
        "fusion_method": "plugin:p2",
        "neck_type": "standard",
        "input_mode": "rgbt",
    }
    assert not identities_match(expected, actual_p3a)
    assert identities_match(expected, actual_p2)


def test_identities_match_rejects_same_node_wrong_epochs() -> None:
    expected = {
        "node_id": "run_plan_r3_c704dec973",
        "fusion_method": "plugin:P3",
        "input_mode": "rgbt",
        "epochs": 8,
    }
    actual_f0 = {
        "node_id": "run_plan_r3_c704dec973",
        "fusion_method": "none",
        "input_mode": "rgb",
        "epochs": 2,
    }
    assert not identities_match(expected, actual_f0)


def test_load_contract_identity_prefers_artifact_knobs(tmp_path: Path) -> None:
    """Stamped contract node_id must not hide wrong train knobs on disk."""
    out = tmp_path / "exec_bad"
    out.mkdir()
    (out / "contract.json").write_text(
        json.dumps(
            {
                "node_id": "run_plan_r3",
                "seed": 42,
                "parameters": {
                    "fusion_method": "plugin:P3",
                    "input_mode": "rgbt",
                    "epochs": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps({"fusion_method": "none", "input_mode": "rgb", "epochs": 2}),
        encoding="utf-8",
    )
    ident = load_contract_identity(out)
    assert ident["node_id"] == "run_plan_r3"
    assert ident["fusion_method"] == "none"
    assert ident["epochs"] == 2
    assert not identities_match(
        {
            "node_id": "run_plan_r3",
            "fusion_method": "plugin:P3",
            "input_mode": "rgbt",
            "epochs": 8,
        },
        ident,
    )


def test_cuda_live_runner_harvests_only_matching_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scientist_lab.adapters.dfine import cuda_runner as cr

    dest = tmp_path / "run"
    dest.mkdir()
    (dest / "metrics.json").write_text("{}", encoding="utf-8")
    (dest / "execution.json").write_text("{}", encoding="utf-8")
    (dest / "_legacy_fast_eval_contract.json").write_text(
        json.dumps({"node_id": "run_stale", "parameters": {"fusion_method": "none"}}),
        encoding="utf-8",
    )

    called = {"eval": False}

    def _eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
        called["eval"] = True
        return {
            "status": "completed",
            "orchestrator": "dfine_cuda",
            "run": {"status": "completed", "execution_id": "exec_new"},
            "dry_run": False,
        }

    class _Legacy:
        project_id = "project_rgbt_cuda_001"
        resources = type("R", (), {"timeout_seconds": 60})()

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "project_id": self.project_id,
                "node_id": "run_fresh",
                "seed": 7,
                "parameters": {"fusion_method": "none", "input_mode": "rgb", "neck": {"type": "standard"}},
            }

    class _Exp:
        settings = type("S", (), {"outputs_dir": tmp_path / "outputs"})()

    monkeypatch.setattr(cr, "freeze_to_legacy_contract", lambda c: {})
    monkeypatch.setattr(cr.ExperimentContract, "model_validate", staticmethod(lambda x: _Legacy()))
    runner = cr.make_cuda_live_runner(_Exp(), execute=True, call_eval=_eval)
    out = runner({"project_id": "project_rgbt_cuda_001", "run_id": "run_fresh"}, dest)
    assert called["eval"] is True
    assert out.get("reattached") is not True
    assert out["status"] == "completed"


def test_cuda_live_runner_prefers_matching_dest_harvest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scientist_lab.adapters.dfine import cuda_runner as cr

    dest = tmp_path / "run"
    dest.mkdir()
    (dest / "metrics.json").write_text("{}", encoding="utf-8")
    (dest / "execution.json").write_text("{}", encoding="utf-8")
    (dest / "_legacy_fast_eval_contract.json").write_text(
        json.dumps({"node_id": "run_same", "parameters": {"fusion_method": "none"}}),
        encoding="utf-8",
    )

    def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("must not start a new GPU job")

    class _Legacy:
        project_id = "project_rgbt_cuda_001"
        resources = type("R", (), {"timeout_seconds": 60})()

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {
                "project_id": self.project_id,
                "node_id": "run_same",
                "parameters": {"fusion_method": "none"},
            }

    class _Exp:
        settings = type("S", (), {"outputs_dir": tmp_path / "outputs"})()

    monkeypatch.setattr(cr, "freeze_to_legacy_contract", lambda c: {})
    monkeypatch.setattr(cr.ExperimentContract, "model_validate", staticmethod(lambda x: _Legacy()))
    runner = cr.make_cuda_live_runner(_Exp(), execute=True, call_eval=_boom)
    out = runner({"project_id": "project_rgbt_cuda_001", "run_id": "run_same"}, dest)
    assert out.get("reattached") is True
    assert out["status"] == "completed"
