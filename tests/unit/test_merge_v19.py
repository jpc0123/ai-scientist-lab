"""v1.9.1 merge-prepare / GitAdapter / worktree isolation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.patching.service import build_mock_unified_diff
from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "merge.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def _evidenced_merge_ready(service: ExperimentService) -> str:
    proposed = service.patches.propose_mock(
        "project_merge_v19",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/merge_v19_note.md"
            )
        ),
    )
    patch_id = proposed["patch_id"]
    service.patches.approve(patch_id, reason="v19 test")
    service.patches.apply_sandbox(patch_id)
    service.patches.test_sandbox(patch_id, profile="smoke")
    service.patches.record_evidence(patch_id)
    service.patches.decide_merge(patch_id, decision="merge", reason="intent only")
    return patch_id


def test_git_adapter_denies_reset_hard():
    root = Path(__file__).resolve().parents[2]
    git = GitAdapter(root)
    with pytest.raises(GitAdapterError):
        git._run(["reset", "--hard", "HEAD"])  # noqa: SLF001


def test_merge_prepare_requires_evidence(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_merge_v19")
    with pytest.raises(ValueError, match="PatchEvidence"):
        service.merge_prepare(proposed["patch_id"])


def test_merge_prepare_creates_worktree_without_commit(tmp_path: Path):
    service = _service(tmp_path)
    root = Path(service.settings.project_root)
    before = GitAdapter(root).status_porcelain()
    patch_id = _evidenced_merge_ready(service)
    prepared = service.merge_prepare(patch_id)
    assert prepared["merge_candidate_id"]
    assert prepared["can_commit"] is False
    assert prepared["can_merge"] is False
    assert prepared["main_workspace_modified"] is False
    assert prepared["status"] in {"created", "merge_conflict"}
    workspace = Path(prepared["workspace_path"])
    assert workspace.is_dir()
    assert (workspace / "PENDING_PATCH.diff").is_file()
    assert workspace.is_relative_to(root / ".scientist-worktrees") or str(
        workspace
    ).startswith(str(root / ".scientist-worktrees"))
    after = GitAdapter(root).status_porcelain()
    # Main tree status must not gain unexpected tracked edits from prepare.
    # (registry.json under .scientist-worktrees is gitignored)
    assert before == after or ".scientist-worktrees" in after

    shown = service.merge_show(prepared["merge_candidate_id"])
    assert shown["patch_id"] == patch_id
    assert shown["patch_evidence_id"]


def test_unapproved_cannot_prepare(tmp_path: Path):
    service = _service(tmp_path)
    proposed = service.patches.propose_mock("project_merge_v19")
    # verified but not approved / no evidence
    with pytest.raises(ValueError):
        service.merge_prepare(proposed["patch_id"])
