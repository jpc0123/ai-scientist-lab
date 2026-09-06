"""GitManager: experiment history ≠ best-model history.

KEEP updates best_sha. DISCARD restores the worktree to best_sha.
Never deletes experiment commits. Never reset --hard.
Does not invent KEEP/DISCARD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError
from scientist_lab.core.state_machine import ReviewDecisionValue

_POINTER_NAME = "scientist_lab_git_pointers.json"


def is_git_repo(path: Path | str) -> bool:
    """True only when ``path/.git`` exists. Does not walk parents.

    CLI ``.run/`` output dirs are not repos; walking up would hit the
    Scientist Lab checkout and accidentally commit into it.
    """
    return (Path(path) / ".git").exists()


class GitManagerError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitPointers:
    experiment_sha: str
    best_sha: str
    runs: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_sha": self.experiment_sha,
            "best_sha": self.best_sha,
            "runs": dict(self.runs),
        }


class GitManager:
    def __init__(self, repo_root: Path | str, *, pointers_path: Path | str | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.git = GitAdapter(self.repo_root)
        self.pointers_path = Path(pointers_path) if pointers_path else self.repo_root / _POINTER_NAME
        self._ensure_pointers()

    def _ensure_pointers(self) -> GitPointers:
        if self.pointers_path.is_file():
            return self.load()
        sha = self.git.rev_parse("HEAD")
        pointers = GitPointers(experiment_sha=sha, best_sha=sha, runs={})
        self._save(pointers)
        return pointers

    def load(self) -> GitPointers:
        data = json.loads(self.pointers_path.read_text(encoding="utf-8"))
        return GitPointers(
            experiment_sha=str(data["experiment_sha"]),
            best_sha=str(data["best_sha"]),
            runs={str(k): str(v) for k, v in dict(data.get("runs") or {}).items()},
        )

    def _save(self, pointers: GitPointers) -> None:
        self.pointers_path.parent.mkdir(parents=True, exist_ok=True)
        self.pointers_path.write_text(
            json.dumps(pointers.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def current_sha(self) -> str:
        return self.git.rev_parse("HEAD")

    def commit_reachable(self, sha: str) -> bool:
        try:
            resolved = self.git.rev_parse(sha)
        except GitAdapterError:
            return False
        return bool(resolved)

    def record_experiment(
        self,
        *,
        run_id: str,
        message: str | None = None,
        paths: Sequence[str] | None = None,
        allow_empty: bool = False,
    ) -> GitPointers:
        """Commit current experiment state. Does not move best_sha."""
        staged = False
        if paths:
            self.git.add_paths(list(paths), cwd=self.repo_root)
            staged = True
        if staged or allow_empty:
            msg = message or f"experiment {run_id}"
            msg_file = self.repo_root / ".scientist_lab_commit_msg.txt"
            msg_file.write_text(msg + "\n", encoding="utf-8")
            try:
                sha = self.git.commit_message_file(
                    msg_file, cwd=self.repo_root, allow_empty=allow_empty and not staged
                )
            finally:
                if msg_file.exists():
                    msg_file.unlink()
        else:
            sha = self.current_sha()
        pointers = self.load()
        runs = dict(pointers.runs)
        runs[run_id] = sha
        updated = GitPointers(
            experiment_sha=sha,
            best_sha=pointers.best_sha,
            runs=runs,
        )
        self._save(updated)
        return updated

    def rollback_workspace(self, to_sha: str) -> str:
        """Restore worktree to sha. Does not delete the experiment commit."""
        if not to_sha or to_sha.startswith("-"):
            raise GitManagerError("invalid rollback sha")
        if not self.commit_reachable(to_sha):
            raise GitManagerError(f"rollback target not reachable: {to_sha}")
        return self.git.switch_detach(to_sha)

    def apply_review(
        self,
        review_decision: str,
        *,
        experiment_sha: str | None = None,
    ) -> GitPointers:
        """Apply KEEP/DISCARD pointers. Other decisions only record experiment_sha."""
        pointers = self.load()
        sha = experiment_sha or pointers.experiment_sha
        if not self.commit_reachable(sha):
            raise GitManagerError(f"experiment_sha not reachable: {sha}")
        decision = str(review_decision)
        runs = dict(pointers.runs)
        if decision == ReviewDecisionValue.KEEP.value:
            updated = GitPointers(experiment_sha=sha, best_sha=sha, runs=runs)
            self._save(updated)
            return updated
        if decision == ReviewDecisionValue.DISCARD.value:
            self.rollback_workspace(pointers.best_sha)
            updated = GitPointers(
                experiment_sha=sha,
                best_sha=pointers.best_sha,
                runs=runs,
            )
            self._save(updated)
            if self.git.rev_parse("HEAD") != self.git.rev_parse(pointers.best_sha):
                raise GitManagerError("DISCARD did not restore best_sha worktree")
            if not self.commit_reachable(sha):
                raise GitManagerError("DISCARD deleted experiment commit")
            return updated
        updated = GitPointers(experiment_sha=sha, best_sha=pointers.best_sha, runs=runs)
        self._save(updated)
        return updated

    def snapshot(self) -> dict[str, Any]:
        pointers = self.load()
        payload = pointers.to_dict()
        payload["head"] = self.current_sha()
        return payload
