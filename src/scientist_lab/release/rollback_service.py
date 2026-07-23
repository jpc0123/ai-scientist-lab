"""Post-merge verification and revert-based rollback (v1.9.7)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError
from scientist_lab.release.models import MergeCandidate, RollbackRecord
from scientist_lab.release.repository import MergeCandidateRepository
from scientist_lab.release.test_profiles import require_profile


class RollbackService:
    """Never uses reset --hard; only git revert (with -m 1 for merge commits)."""

    def __init__(
        self,
        *,
        project_root: Path,
        git: GitAdapter,
        candidates: MergeCandidateRepository,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.git = git
        self._repo = candidates

    def run_post_merge_check(
        self,
        candidate: MergeCandidate,
        *,
        profile_id: str = "syntax",
    ) -> dict[str, Any]:
        """Run a registry profile on the main repo after finalize."""
        profile = require_profile(self.project_root, profile_id)
        command_results: list[dict[str, Any]] = []
        ok = True
        for cmd in profile.commands:
            try:
                completed = subprocess.run(
                    list(cmd),
                    cwd=str(self.project_root),
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
                        "stderr_tail": f"timeout: {exc}",
                        "stdout_tail": "",
                    }
                )
                break
            command_results.append(
                {
                    "command": list(cmd),
                    "returncode": int(completed.returncode),
                    "stdout_tail": (completed.stdout or "")[-1500:],
                    "stderr_tail": (completed.stderr or "")[-1500:],
                }
            )
            if completed.returncode != 0:
                ok = False
                break
        return {
            "ok": ok,
            "profile_id": profile.profile_id,
            "commands": command_results,
            "finished_at": utc_now_iso(),
        }

    def rollback(
        self,
        merge_candidate_id: str,
        *,
        reason: str = "",
        trigger: str = "human",
        verify_profile: str = "syntax",
        run_verify: bool = True,
    ) -> dict[str, Any]:
        candidate = self._repo.require(merge_candidate_id)
        if candidate.status not in {"merged", "failed"}:
            raise ValueError(
                f"rollback requires merged (or failed post-merge); got {candidate.status!r}"
            )
        finalize = dict((candidate.metadata or {}).get("finalize") or {})
        merge_sha = str(
            finalize.get("merge_commit_sha") or candidate.commit_sha or ""
        )
        if not merge_sha:
            raise ValueError("no merge commit sha available to revert")

        current = self.git.current_branch()
        if current != candidate.target_branch:
            raise ValueError(
                f"check out {candidate.target_branch!r} before rollback "
                f"(currently on {current!r})"
            )

        dirty = [
            line
            for line in self.git.status_porcelain().splitlines()
            if line.strip() and ".scientist-worktrees" not in line
        ]
        if dirty:
            raise ValueError(
                "main working tree is dirty; refuse rollback: "
                + "; ".join(dirty[:5])
            )

        record = RollbackRecord(
            merge_candidate_id=merge_candidate_id,
            original_commit_sha=merge_sha,
            trigger=trigger,
            reason=reason or "post-merge rollback via git revert",
            status="pending",
        )
        msg = (
            f"revert: roll back merge candidate {merge_candidate_id}\n"
            f"\n"
            f"Original merge: {merge_sha}\n"
            f"Trigger: {trigger}\n"
            f"Reason: {record.reason}\n"
        )
        msg_file = (
            self.project_root
            / ".scientist-worktrees"
            / f"{merge_candidate_id}_REVERT_MSG.txt"
        )
        msg_file.parent.mkdir(parents=True, exist_ok=True)
        msg_file.write_text(msg, encoding="utf-8")

        try:
            # Merge commits from --no-ff require -m 1.
            revert_sha = self.git.revert_commit(
                merge_sha, mainline=1, message_file=msg_file
            )
        except GitAdapterError as exc:
            record.status = "failed"
            candidate.error = f"rollback revert failed: {exc}"
            candidate.metadata = {
                **dict(candidate.metadata or {}),
                "rollback": record.model_dump(mode="json"),
            }
            self._repo.upsert(candidate)
            raise ValueError(candidate.error) from exc

        record.rollback_commit_sha = revert_sha
        record.status = "reverted"
        verify: dict[str, Any] | None = None
        if run_verify:
            verify = self.run_post_merge_check(candidate, profile_id=verify_profile)
            if not verify.get("ok"):
                record.status = "reverted_verify_failed"
                candidate.status = "rolled_back"
                candidate.error = (
                    "revert succeeded but verification profile failed; "
                    "manual inspection required (no reset --hard)"
                )
                candidate.metadata = {
                    **dict(candidate.metadata or {}),
                    "rollback": record.model_dump(mode="json"),
                    "rollback_verify": verify,
                    "v19_stage": "1.9.7-rolled-back-verify-failed",
                    "pushed": False,
                }
                self._repo.upsert(candidate)
                view = candidate.model_dump(mode="json")
                view["rollback"] = record.model_dump(mode="json")
                view["rollback_verify"] = verify
                return view

        candidate.status = "rolled_back"
        candidate.error = None
        candidate.metadata = {
            **dict(candidate.metadata or {}),
            "rollback": record.model_dump(mode="json"),
            "rollback_verify": verify,
            "v19_stage": "1.9.7-rolled-back",
            "pushed": False,
            "used_reset_hard": False,
        }
        self._repo.upsert(candidate)
        view = candidate.model_dump(mode="json")
        view["rollback"] = record.model_dump(mode="json")
        view["rollback_verify"] = verify
        view["can_rollback"] = False
        view["main_workspace_modified"] = False
        return view
