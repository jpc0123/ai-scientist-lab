"""Unit tests for experiment failure classifier."""

from pathlib import Path

from scientist_lab.services.failure_classifier import (
    classify_failure,
    user_approved_resume,
    write_classification,
)


def test_binding_json_corrupt_is_engineering_recover():
    clf = classify_failure(
        cli_text=(
            "pydantic_core._pydantic_core.ValidationError: 1 validation error for "
            "RemoteJobBinding\n  Invalid JSON: EOF while parsing a value at line 1 "
            "column 0"
        ),
        worker_status="running",
    )
    assert clf.kind == "engineering"
    assert clf.action == "auto_recover_wait_worker"
    assert clf.reason_code in {
        "worker_still_active",
        "binding_json_corrupt",
    }


def test_binding_corrupt_when_worker_dead_still_engineering():
    clf = classify_failure(
        cli_text="Invalid JSON: EOF while parsing a value at line 1 column 0",
        worker_status="failed",
    )
    assert clf.kind == "engineering"
    assert clf.action == "auto_recover_wait_worker"
    assert clf.reason_code == "binding_json_corrupt"
    assert clf.auto_rerun_allowed is False


def test_deterministic_op_allows_one_retry():
    clf = classify_failure(
        log_text=(
            "RuntimeError: grid_sampler_2d_backward_cuda does not have a "
            "deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True)'."
        )
    )
    assert clf.kind == "engineering"
    assert clf.action == "auto_retry_once"
    assert clf.auto_rerun_allowed is True


def test_scientific_nan_waits_for_user():
    clf = classify_failure(log_text="Epoch 3 loss became NaN; training collapsed")
    assert clf.kind == "scientific"
    assert clf.action == "wait_user"
    assert clf.auto_rerun_allowed is False


def test_user_cancel_waits():
    clf = classify_failure(exec_status="cancelled", worker_status="cancelled")
    assert clf.kind == "user_pause"
    assert clf.action == "wait_user"


def test_metrics_present_short_circuits():
    clf = classify_failure(metrics_present=True, cli_text="anything")
    assert clf.reason_code == "metrics_already_present"
    assert clf.action == "auto_recover_wait_worker"


def test_write_wait_user_and_resume_gate(tmp_path: Path):
    clf = classify_failure(log_text="diagnose_seed_instability required")
    write_classification(tmp_path, clf, context={"execution_id": "exec_x"})
    assert (tmp_path / "FAILURE_CLASSIFICATION.json").is_file()
    assert (tmp_path / "WAIT_USER.json").is_file()
    assert user_approved_resume(tmp_path) is None
    (tmp_path / "RESUME_APPROVED.json").write_text(
        '{"approve": true, "action": "retry"}\n', encoding="utf-8"
    )
    approved = user_approved_resume(tmp_path)
    assert approved is not None
    assert approved["action"] == "retry"


def test_unclassified_defaults_to_wait_user():
    clf = classify_failure(log_text="something totally novel broke")
    assert clf.kind == "unknown"
    assert clf.action == "wait_user"
