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


def test_merge_apply_and_syntax_test(tmp_path: Path):
    service = _service(tmp_path)
    root = Path(service.settings.project_root)
    before = GitAdapter(root).status_porcelain()
    patch_id = _evidenced_merge_ready(service)
    prepared = service.merge_prepare(patch_id)
    assert prepared["status"] == "created"
    assert prepared["can_apply"] is True

    applied = service.merge_apply(prepared["merge_candidate_id"])
    assert applied["workspace_applied"] is True
    assert applied["can_commit"] is False
    assert applied["status"] == "preparing"
    changed = (applied.get("metadata") or {}).get("changed_paths") or []
    assert any("merge_v19_note.md" in p for p in changed)

    tested = service.merge_test(prepared["merge_candidate_id"], profile_id="syntax")
    assert tested["status"] == "waiting_approval"
    assert tested["test_result"]["ok"] is True
    assert tested["can_approve"] is True

    after = GitAdapter(root).status_porcelain()
    assert before == after or ".scientist-worktrees" in after

    listed = service.list_merge_candidates(patch_id=patch_id)
    assert len(listed) >= 1


def test_merge_approve_commit_keeps_main_head(tmp_path: Path):
    service = _service(tmp_path)
    root = Path(service.settings.project_root)
    git = GitAdapter(root)
    head_before = git.rev_parse("HEAD")
    patch_id = _evidenced_merge_ready(service)
    mc = service.merge_prepare(patch_id)["merge_candidate_id"]
    service.merge_apply(mc)
    service.merge_test(mc, profile_id="syntax")

    with pytest.raises(ValueError, match="commit requires approved"):
        # cannot commit before approve
        service.merge_commit(mc)

    approved = service.merge_approve(mc, reason="ok to commit in worktree")
    assert approved["status"] == "approved"
    assert approved["can_commit"] is True

    committed = service.merge_commit(mc)
    assert committed["commit_sha"]
    assert committed["commit_sha"] != head_before
    assert committed["status"] == "committing"
    # Main branch HEAD must not move on commit (only worktree).
    assert git.rev_parse("HEAD") == head_before

    # Finalize would merge into current branch — refuse dirty/wrong context tests
    # by only checking gate: finalize allowed flag is set.
    assert committed["can_finalize"] is True


def test_merge_reject_from_waiting_approval(tmp_path: Path):
    service = _service(tmp_path)
    patch_id = _evidenced_merge_ready(service)
    mc = service.merge_prepare(patch_id)["merge_candidate_id"]
    service.merge_apply(mc)
    service.merge_test(mc, profile_id="syntax")
    rejected = service.merge_reject(mc, reason="not now")
    assert rejected["status"] == "rejected"
    with pytest.raises(ValueError):
        service.merge_approve(mc, reason="too late")


def test_merge_finalize_on_disposable_repo(tmp_path: Path):
    import subprocess

    repo = tmp_path / "mini_repo"
    adapters = repo / "experiment_apps" / "rgbt_detection_real" / "adapters"
    adapters.mkdir(parents=True)
    (adapters / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "merge-test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Merge Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    service = ExperimentService(
        settings=Settings(
            project_root=repo,
            db_path=tmp_path / "mini.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=repo,
        ).resolve()
    )
    proposed = service.patches.propose_mock(
        "project_mini",
        unified_diff=build_mock_unified_diff(
            relative_path="experiment_apps/rgbt_detection_real/adapters/mini_note.md"
        ),
    )
    patch_id = proposed["patch_id"]
    service.patches.approve(patch_id, reason="mini")
    service.patches.apply_sandbox(patch_id)
    service.patches.test_sandbox(patch_id, profile="smoke")
    service.patches.record_evidence(patch_id)
    service.patches.decide_merge(patch_id, decision="merge", reason="intent")

    mc = service.merge_prepare(patch_id, target_branch="main")["merge_candidate_id"]
    service.merge_apply(mc)
    # Inject passing test state (mini repo has no full source tree for syntax profile).
    candidate = service.merges._repo.require(mc)
    candidate.status = "waiting_approval"
    candidate.metadata = {
        **dict(candidate.metadata or {}),
        "last_test": {"ok": True, "profile_id": "syntax"},
        "workspace_applied": True,
    }
    service.merges._repo.upsert(candidate)

    service.merge_approve(mc, reason="mini approve")
    committed = service.merge_commit(mc)
    assert committed["commit_sha"]
    head_before = GitAdapter(repo).rev_parse("HEAD")
    finalized = service.merge_finalize(mc, post_merge_profile=None)
    assert finalized["status"] == "merged"
    assert finalized["merge_commit_sha"]
    assert GitAdapter(repo).rev_parse("HEAD") != head_before
    assert GitAdapter(repo).rev_parse("HEAD") == finalized["merge_commit_sha"]

    rolled = service.merge_rollback(mc, reason="manual rollback test")
    assert rolled["status"] == "rolled_back"
    assert rolled["rollback"]["rollback_commit_sha"]
    assert rolled["rollback"]["original_commit_sha"] == finalized["merge_commit_sha"]
    assert GitAdapter(repo).rev_parse("HEAD") == rolled["rollback"]["rollback_commit_sha"]


def test_finalize_auto_rollback_when_post_merge_fails(tmp_path: Path):
    import subprocess

    repo = tmp_path / "mini_repo2"
    adapters = repo / "experiment_apps" / "rgbt_detection_real" / "adapters"
    adapters.mkdir(parents=True)
    (adapters / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "merge-test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Merge Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    service = ExperimentService(
        settings=Settings(
            project_root=repo,
            db_path=tmp_path / "mini2.db",
            runtime_dir=tmp_path / "runtime2",
            outputs_dir=tmp_path / "outputs2",
            experiment_app_dir=repo,
        ).resolve()
    )
    proposed = service.patches.propose_mock(
        "project_mini2",
        unified_diff=build_mock_unified_diff(
            relative_path="experiment_apps/rgbt_detection_real/adapters/mini2_note.md"
        ),
    )
    patch_id = proposed["patch_id"]
    service.patches.approve(patch_id, reason="mini")
    service.patches.apply_sandbox(patch_id)
    service.patches.test_sandbox(patch_id, profile="smoke")
    service.patches.record_evidence(patch_id)
    service.patches.decide_merge(patch_id, decision="merge", reason="intent")
    mc = service.merge_prepare(patch_id, target_branch="main")["merge_candidate_id"]
    service.merge_apply(mc)
    candidate = service.merges._repo.require(mc)
    candidate.status = "waiting_approval"
    candidate.metadata = {
        **dict(candidate.metadata or {}),
        "last_test": {"ok": True, "profile_id": "syntax"},
        "workspace_applied": True,
    }
    service.merges._repo.upsert(candidate)
    service.merge_approve(mc, reason="approve")
    service.merge_commit(mc)
    # unit profile runs pytest on missing tests in mini repo → fail → auto rollback
    result = service.merge_finalize(mc, post_merge_profile="unit")
    assert result["status"] == "rolled_back"
    assert result.get("auto_rolled_back") is True
    assert result["rollback"]["trigger"] == "post_merge_failure"
    assert result["post_merge_check"]["ok"] is False


def test_merge_test_rejects_unknown_profile(tmp_path: Path):
    service = _service(tmp_path)
    patch_id = _evidenced_merge_ready(service)
    prepared = service.merge_prepare(patch_id)
    service.merge_apply(prepared["merge_candidate_id"])
    with pytest.raises(ValueError, match="unknown test profile"):
        service.merge_test(prepared["merge_candidate_id"], profile_id="arbitrary_shell")
