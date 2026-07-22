"""v1.6.1–1.6.4 restricted patching tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.patching.diff_parser import DiffParseError, parse_unified_diff
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.service import PatchingService, build_mock_unified_diff
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "test.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def test_parse_mock_diff():
    diff = build_mock_unified_diff()
    parsed = parse_unified_diff(diff)
    assert len(parsed.files) == 1
    assert "adapters/mock_patch_note.md" in parsed.files[0].path.replace("\\", "/")
    assert parsed.files[0].is_new_file is True


def test_path_policy_allows_and_denies():
    policy = PathPolicy()
    ok, _ = policy.is_allowed(
        "src/scientist_lab/tasks/rgbt_detection/feedback_rules.py"
    )
    assert ok is True
    bad, reason = policy.is_allowed("src/scientist_lab/llm/config.py")
    assert bad is False
    assert "denied" in reason or "explicit" in reason
    escape, _ = policy.is_allowed(
        "src/scientist_lab/tasks/rgbt_detection/../../llm/config.py"
    )
    assert escape is False
    abs_ok, _ = policy.is_allowed("D:/secret/file.py")
    assert abs_ok is False
    docker, _ = policy.is_allowed(
        "experiment_apps/rgbt_detection_real/adapters/Dockerfile"
    )
    assert docker is False


def test_verifier_rejects_forbidden_and_binary():
    verifier = PatchVerifier()
    bad = (
        "diff --git a/pyproject.toml b/pyproject.toml\n"
        "--- a/pyproject.toml\n"
        "+++ b/pyproject.toml\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    result = verifier.verify(bad)
    assert result.ok is False
    assert any(i.code == "path_policy_violation" for i in result.issues)

    binary = (
        "diff --git a/src/scientist_lab/tasks/rgbt_detection/x.bin "
        "b/src/scientist_lab/tasks/rgbt_detection/x.bin\n"
        "Binary files a/x.bin and b/x.bin differ\n"
    )
    result2 = verifier.verify(binary)
    assert result2.ok is False
    assert any(i.code == "binary_file" for i in result2.issues)


def test_fingerprint_stable():
    diff = build_mock_unified_diff()
    assert fingerprint_diff(diff) == fingerprint_diff(diff + "\n")


def test_propose_verify_approve_reject_no_main_apply(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_rgbt_003")
    assert proposed["status"] == "verified"
    assert proposed["can_apply"] is False
    assert proposed["can_apply_main"] is False
    assert proposed["provider"] == "mock"
    assert proposed["verification"]["ok"] is True
    assert proposed["files_touched"]

    shown = service.patches.show(proposed["patch_id"])
    assert shown["patch_id"] == proposed["patch_id"]

    verified = service.patches.verify(proposed["patch_id"])
    assert verified["status"] == "verified"

    approved = service.patches.approve(proposed["patch_id"], reason="ok for sandbox")
    assert approved["status"] == "approved"
    assert approved["can_apply_main"] is False
    assert approved["can_apply_sandbox"] is True

    second = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/other_note.md"
            )
        ),
    )
    rejected = service.patches.reject(second["patch_id"], reason="not needed")
    assert rejected["status"] == "rejected"


def test_duplicate_fingerprint_blocked(tmp_path: Path):
    service = _service(tmp_path)
    first = service.patches.propose_mock("project_rgbt_003")
    assert first["status"] == "verified"
    second = service.patches.propose_mock("project_rgbt_003")
    assert second["status"] == "rejected_by_verifier"
    assert second["verification"]["duplicate_of"] == first["patch_id"]


def test_apply_sandbox_writes_only_sandbox(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_rgbt_003")
    service.patches.approve(proposed["patch_id"])
    applied = service.patches.apply_sandbox(proposed["patch_id"])
    assert applied["status"] == "applied_sandbox"
    assert applied["sandbox"]["ok"] is True
    assert applied["sandbox"]["main_workspace_modified"] is False
    sandbox_dir = Path(applied["sandbox"]["sandbox_dir"])
    assert sandbox_dir.is_dir()
    assert str(tmp_path / "outputs") in str(sandbox_dir)
    note = sandbox_dir / "experiment_apps/rgbt_detection_real/adapters/mock_patch_note.md"
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "fusion ablation" in text
    # Main tree must not contain this new file.
    main_note = (
        root
        / "experiment_apps"
        / "rgbt_detection_real"
        / "adapters"
        / "mock_patch_note.md"
    )
    assert not main_note.exists()


def test_apply_sandbox_requires_approval(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path="experiment_apps/rgbt_detection_real/configs/demo.yaml"
        ),
    )
    with pytest.raises(ValueError, match="approved"):
        service.patches.apply_sandbox(proposed["patch_id"])


def test_test_sandbox_smoke_and_mock_experiment(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_rgbt_003")
    service.patches.approve(proposed["patch_id"])
    service.patches.apply_sandbox(proposed["patch_id"])

    smoke = service.patches.test_sandbox(proposed["patch_id"], profile="smoke")
    assert smoke["can_test_sandbox"] is True
    assert smoke["sandbox_tests"]["ok"] is True
    assert smoke["sandbox_tests"]["main_workspace_modified"] is False
    report_path = Path(smoke["sandbox_tests"]["report_path"])
    assert report_path.is_file()

    mock_exp = service.patches.test_sandbox(
        proposed["patch_id"], profile="mock_experiment"
    )
    assert mock_exp["sandbox_tests"]["ok"] is True
    assert mock_exp["sandbox_tests_ok"] is True
    names = {c["name"] for c in mock_exp["sandbox_tests"]["checks"]}
    assert "mock_experiment_no_shell" in names
    assert "mock_experiment_marker" in names


def test_test_sandbox_requires_applied(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_rgbt_003")
    service.patches.approve(proposed["patch_id"])
    with pytest.raises(ValueError, match="applied_sandbox"):
        service.patches.test_sandbox(proposed["patch_id"])


def test_record_evidence_and_decide_merge_no_main_apply(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_rgbt_003")
    pid = proposed["patch_id"]
    service.patches.approve(pid)
    service.patches.apply_sandbox(pid)
    service.patches.test_sandbox(pid, profile="mock_experiment")

    recorded = service.patches.record_evidence(pid, require_tests=True)
    assert recorded["status"] == "evidence_recorded"
    assert recorded["can_apply_main"] is False
    evidence = recorded["patch_evidence"]
    assert evidence["main_workspace_modified"] is False
    assert evidence["sandbox_tests_ok"] is True
    assert evidence["evidence_strength"] == "moderate"
    artifact = Path(evidence["artifact_path"])
    assert artifact.is_file()
    assert str(tmp_path / "outputs") in str(artifact)

    merged = service.patches.decide_merge(
        pid, decision="merge", reason="accept for later human git apply"
    )
    assert merged["status"] == "merged"
    assert merged["can_apply_main"] is False
    assert merged["applied_main"] is False
    assert merged["merge_decision"]["applied_main"] is False
    assert "not modified" in merged["warning"].lower()

    # Main tree still untouched.
    main_note = (
        root
        / "experiment_apps"
        / "rgbt_detection_real"
        / "adapters"
        / "mock_patch_note.md"
    )
    assert not main_note.exists()


def test_decide_merge_discard(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/discard_note.md"
            )
        ),
    )
    pid = proposed["patch_id"]
    service.patches.approve(pid)
    service.patches.apply_sandbox(pid)
    service.patches.record_evidence(pid)
    discarded = service.patches.decide_merge(pid, decision="discard", reason="noise")
    assert discarded["status"] == "discarded"
    assert discarded["applied_main"] is False


def test_record_evidence_require_tests(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path="experiment_apps/rgbt_detection_real/configs/req.yaml"
        ),
    )
    pid = proposed["patch_id"]
    service.patches.approve(pid)
    service.patches.apply_sandbox(pid)
    with pytest.raises(ValueError, match="sandbox tests required"):
        service.patches.record_evidence(pid, require_tests=True)


def test_empty_diff_parse_error():
    with pytest.raises(DiffParseError):
        parse_unified_diff("")
