"""Controlled merge prepare / apply / test / show (v1.9.1–1.9.4)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.models import new_id, utc_now_iso
from scientist_lab.patching.models import PatchProposal
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.repository import PatchRepository
from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError
from scientist_lab.release.models import MergeCandidate
from scientist_lab.release.repository import MergeCandidateRepository
from scientist_lab.release.test_profiles import list_profile_ids, require_profile
from scientist_lab.release.verifier import reverify_patch, sha256_text
from scientist_lab.release.workspace_service import WorkspaceRegistry


_IGNORE_WORKSPACE_FILES = frozenset({"PENDING_PATCH.diff"})


class MergeService:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        project_root: Path,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._patches = PatchRepository(session_factory)
        self._repo = MergeCandidateRepository(session_factory)
        self.git = GitAdapter(self.project_root)
        self.workspaces = WorkspaceRegistry(self.project_root, git=self.git)

    def show(self, merge_candidate_id: str) -> dict[str, Any]:
        return self._view(self._repo.require(merge_candidate_id))

    def list_candidates(
        self,
        *,
        project_id: str | None = None,
        patch_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return [
            self._view(item)
            for item in self._repo.list_all(
                project_id=project_id, patch_id=patch_id, limit=limit
            )
        ]

    def list_profiles(self) -> list[dict[str, Any]]:
        return list_profile_ids(self.project_root)

    def prepare(
        self,
        patch_id: str,
        *,
        target_branch: str | None = None,
    ) -> dict[str, Any]:
        """Create isolated worktree + MergeCandidate. Does not commit or merge."""
        proposal = self._patches.require(patch_id)
        self._require_ready_for_prepare(proposal)

        evidence = dict((proposal.metadata or {}).get("patch_evidence") or {})
        evidence_id = str(
            evidence.get("evidence_id")
            or (proposal.metadata or {}).get("patch_evidence_id")
            or ""
        )
        if not evidence_id:
            raise ValueError("PatchEvidence required before merge-prepare")

        reverify = reverify_patch(proposal)
        if not reverify.get("ok"):
            raise ValueError(
                "patch re-verification failed; refuse MergeCandidate creation: "
                f"{reverify.get('issues')}"
            )

        branch = (target_branch or self.git.current_branch()).strip()
        if not branch or branch == "HEAD":
            branch = self.git.current_branch()
        source_commit = self.git.rev_parse(branch if target_branch else "HEAD")

        merge_candidate_id = new_id("mc")
        main_before = self.git.status_porcelain()
        try:
            ws = self.workspaces.create(
                merge_candidate_id=merge_candidate_id,
                source_commit=source_commit,
                patch_id=patch_id,
                target_branch=branch,
            )
        except GitAdapterError as exc:
            raise ValueError(f"failed to create isolated worktree: {exc}") from exc

        workspace_path = Path(ws["path"])
        patch_file = workspace_path / "PENDING_PATCH.diff"
        patch_file.write_text(proposal.unified_diff, encoding="utf-8")
        check = self.git.apply_check(patch_file, cwd=workspace_path)
        apply_ok = check.ok
        detail = check.stderr or check.stdout or ""

        status: str = "created"
        error: str | None = None
        if not apply_ok:
            status = "merge_conflict"
            error = detail or "git apply --check failed"

        main_after = self.git.status_porcelain()
        if main_before != main_after:
            try:
                self.workspaces.remove(merge_candidate_id, force=True)
            except GitAdapterError:
                pass
            raise ValueError("main workspace changed during merge-prepare; aborted")

        candidate = MergeCandidate(
            merge_candidate_id=merge_candidate_id,
            project_id=proposal.project_id,
            patch_id=proposal.patch_id,
            patch_evidence_id=evidence_id,
            source_commit=source_commit,
            target_branch=branch,
            workspace_path=str(workspace_path),
            patch_sha256=str(
                reverify.get("fingerprint_sha256") or proposal.fingerprint_sha256
            ),
            diff_sha256=str(
                reverify.get("diff_sha256") or sha256_text(proposal.unified_diff)
            ),
            status=status,  # type: ignore[arg-type]
            apply_check_ok=apply_ok,
            apply_check_detail=detail[:2000],
            reverify=reverify,
            metadata={
                "main_workspace_modified": False,
                "workspace_applied": False,
                "can_commit": False,
                "can_merge": False,
                "v19_stage": "1.9.2-prepare",
                "merge_decision": (proposal.metadata or {}).get("merge_decision"),
                "approved_files": list(proposal.files_touched),
            },
            error=error,
        )
        self._repo.upsert(candidate)
        return self._view(candidate)

    def apply_workspace(self, merge_candidate_id: str) -> dict[str, Any]:
        """Apply approved patch inside the isolated worktree only (v1.9.3)."""
        candidate = self._repo.require(merge_candidate_id)
        if candidate.status == "merge_conflict":
            raise ValueError("cannot apply: merge_conflict (re-prepare with a new patch)")
        if candidate.status not in {"created", "preparing", "failed"}:
            raise ValueError(
                f"apply not allowed from status={candidate.status!r}"
            )
        if candidate.apply_check_ok is False:
            raise ValueError("apply --check previously failed; refuse apply")
        if (candidate.metadata or {}).get("workspace_applied") is True:
            return self._view(candidate)

        proposal = self._patches.require(candidate.patch_id)
        reverify = reverify_patch(proposal)
        if not reverify.get("ok"):
            candidate.status = "failed"
            candidate.error = "fingerprint/reverify failed before apply"
            candidate.reverify = reverify
            self._repo.upsert(candidate)
            raise ValueError(candidate.error)
        if reverify.get("fingerprint_sha256") != candidate.patch_sha256:
            candidate.status = "failed"
            candidate.error = "patch fingerprint changed since prepare"
            self._repo.upsert(candidate)
            raise ValueError(candidate.error)

        workspace = Path(candidate.workspace_path)
        if not workspace.is_dir():
            raise ValueError(f"workspace missing: {workspace}")
        patch_file = workspace / "PENDING_PATCH.diff"
        if not patch_file.is_file():
            patch_file.write_text(proposal.unified_diff, encoding="utf-8")

        candidate.status = "preparing"
        self._repo.upsert(candidate)

        main_before = self.git.status_porcelain()
        try:
            self.git.apply(patch_file, cwd=workspace)
        except GitAdapterError as exc:
            candidate.status = "merge_conflict"
            candidate.error = str(exc)
            candidate.metadata = {
                **dict(candidate.metadata or {}),
                "workspace_applied": False,
            }
            self._repo.upsert(candidate)
            raise ValueError(f"git apply failed: {exc}") from exc

        main_after = self.git.status_porcelain()
        if main_before != main_after:
            candidate.status = "failed"
            candidate.error = "main workspace changed during apply; aborted"
            self._repo.upsert(candidate)
            raise ValueError(candidate.error)

        changed = self._collect_changed_paths(workspace)
        policy = PathPolicy()
        approved = {
            str(p).replace("\\", "/")
            for p in (candidate.metadata or {}).get("approved_files")
            or proposal.files_touched
        }
        unexpected: list[str] = []
        denied: list[str] = []
        for path in changed:
            if path in _IGNORE_WORKSPACE_FILES:
                continue
            ok, reason = policy.is_allowed(path)
            if not ok:
                denied.append(f"{path}:{reason}")
            if approved and path not in approved:
                # New-file patches may list the new path; allow if policy ok and
                # path was in reverify files_touched.
                touched = set(reverify.get("files_touched") or [])
                if path not in touched:
                    unexpected.append(path)

        if denied or unexpected:
            candidate.status = "failed"
            candidate.error = (
                f"post-apply path gate failed denied={denied} unexpected={unexpected}"
            )
            candidate.metadata = {
                **dict(candidate.metadata or {}),
                "workspace_applied": False,
                "changed_paths": changed,
            }
            self._repo.upsert(candidate)
            raise ValueError(candidate.error)

        candidate.status = "preparing"
        candidate.error = None
        candidate.metadata = {
            **dict(candidate.metadata or {}),
            "workspace_applied": True,
            "changed_paths": changed,
            "applied_at": utc_now_iso(),
            "v19_stage": "1.9.3-applied",
            "main_workspace_modified": False,
        }
        self._repo.upsert(candidate)
        return self._view(candidate)

    def run_tests(
        self,
        merge_candidate_id: str,
        *,
        profile_id: str = "smoke",
    ) -> dict[str, Any]:
        """Run a registry TestProfile inside the worktree (v1.9.4)."""
        candidate = self._repo.require(merge_candidate_id)
        if not (candidate.metadata or {}).get("workspace_applied"):
            raise ValueError("apply patch in worktree before merge-test")
        if candidate.status in {"merge_conflict", "rejected", "rolled_back", "merged"}:
            raise ValueError(f"cannot test from status={candidate.status!r}")

        profile = require_profile(self.project_root, profile_id)
        workspace = Path(candidate.workspace_path)
        if not workspace.is_dir():
            raise ValueError(f"workspace missing: {workspace}")

        candidate.status = "testing"
        candidate.test_run_id = new_id("testrun")
        self._repo.upsert(candidate)

        main_before = self.git.status_porcelain()
        command_results: list[dict[str, Any]] = []
        ok = True
        for cmd in profile.commands:
            if not cmd or not isinstance(cmd, list):
                raise ValueError("invalid profile command")
            try:
                completed = subprocess.run(
                    list(cmd),
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    shell=False,
                    check=False,
                    timeout=int(profile.timeout_seconds),
                )
            except subprocess.TimeoutExpired as exc:
                ok = False
                command_results.append(
                    {
                        "command": list(cmd),
                        "returncode": -1,
                        "stdout_tail": "",
                        "stderr_tail": f"timeout after {profile.timeout_seconds}s: {exc}",
                    }
                )
                break
            entry = {
                "command": list(cmd),
                "returncode": int(completed.returncode),
                "stdout_tail": (completed.stdout or "")[-2000:],
                "stderr_tail": (completed.stderr or "")[-2000:],
            }
            command_results.append(entry)
            if completed.returncode != 0:
                ok = False
                break

        main_after = self.git.status_porcelain()
        if main_before != main_after:
            candidate.status = "failed"
            candidate.error = "main workspace changed during merge-test; aborted"
            self._repo.upsert(candidate)
            raise ValueError(candidate.error)

        test_payload = {
            "test_run_id": candidate.test_run_id,
            "profile_id": profile.profile_id,
            "ok": ok,
            "commands": command_results,
            "finished_at": utc_now_iso(),
        }
        candidate.metadata = {
            **dict(candidate.metadata or {}),
            "last_test": test_payload,
            "v19_stage": "1.9.4-tested",
            "main_workspace_modified": False,
        }
        if ok:
            candidate.status = "waiting_approval"
            candidate.error = None
        else:
            candidate.status = "failed"
            candidate.error = f"test profile {profile.profile_id!r} failed"
        self._repo.upsert(candidate)
        view = self._view(candidate)
        view["test_result"] = test_payload
        return view

    def _collect_changed_paths(self, workspace: Path) -> list[str]:
        names = set(self.git.diff_name_only(cwd=workspace))
        for path in self.git.untracked_files(cwd=workspace):
            names.add(path)
        porcelain = self.git.status_porcelain(cwd=workspace)
        for line in porcelain.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip().replace("\\", "/")
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path and path not in _IGNORE_WORKSPACE_FILES:
                names.add(path)
        cleaned: list[str] = []
        for path in sorted(names):
            if not path or path in _IGNORE_WORKSPACE_FILES:
                continue
            # Ignore directory placeholders from porcelain (trailing slash).
            if path.endswith("/"):
                continue
            full = workspace / path
            if full.is_dir():
                continue
            cleaned.append(path)
        return cleaned

    def _require_ready_for_prepare(self, proposal: PatchProposal) -> None:
        meta = dict(proposal.metadata or {})
        evidence = meta.get("patch_evidence")
        if not evidence:
            raise ValueError("PatchEvidence missing; run patch-record-evidence first")
        decision = dict(meta.get("merge_decision") or {})
        if decision.get("decision") != "merge":
            raise ValueError(
                "merge decision must be 'merge' (run patch-decide-merge --decision merge)"
            )
        if proposal.status not in {"merged", "evidence_recorded"}:
            raise ValueError(
                "merge-prepare requires evidenced patch with merge intent; "
                f"got status={proposal.status!r}"
            )
        if meta.get("applied_main") is True:
            raise ValueError("patch already applied to main; refuse new MergeCandidate")

    def _view(self, candidate: MergeCandidate) -> dict[str, Any]:
        meta = dict(candidate.metadata or {})
        applied = bool(meta.get("workspace_applied"))
        data = candidate.model_dump(mode="json")
        data["can_apply"] = (
            candidate.status in {"created", "preparing", "failed"}
            and candidate.apply_check_ok is not False
            and not applied
            and candidate.status != "merge_conflict"
        )
        data["can_test"] = applied and candidate.status in {
            "preparing",
            "testing",
            "failed",
            "waiting_approval",
        }
        data["can_approve"] = False  # v1.9.5
        data["can_commit"] = False
        data["can_merge"] = False
        data["main_workspace_modified"] = False
        data["workspace_exists"] = Path(candidate.workspace_path).exists()
        data["workspace_applied"] = applied
        return data
