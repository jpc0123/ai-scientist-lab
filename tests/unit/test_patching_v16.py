"""v1.6.1–1.6.3 restricted patching tests."""

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
    assert parsed.files[0].path.endswith("feedback_rules.py")
    assert parsed.files[0].hunks


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


def test_propose_verify_approve_reject_no_apply(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_rgbt_003")
    assert proposed["status"] == "verified"
    assert proposed["can_apply"] is False
    assert proposed["applied"] is False
    assert proposed["provider"] == "mock"
    assert proposed["verification"]["ok"] is True
    assert proposed["files_touched"]

    shown = service.patches.show(proposed["patch_id"])
    assert shown["patch_id"] == proposed["patch_id"]

    verified = service.patches.verify(proposed["patch_id"])
    assert verified["status"] == "verified"

    approved = service.patches.approve(proposed["patch_id"], reason="ok for later sandbox")
    assert approved["status"] == "approved"
    assert approved["can_apply"] is False
    assert approved["metadata"]["applied"] is False

    # Reject path on a second patch
    second = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path="src/scientist_lab/tasks/rgbt_detection/claim_gate.py"
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


def test_empty_diff_parse_error():
    with pytest.raises(DiffParseError):
        parse_unified_diff("")
