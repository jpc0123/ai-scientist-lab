"""Isolated Git worktree registry under .scientist-worktrees/ (v1.9.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError


WORKTREE_DIRNAME = ".scientist-worktrees"


class WorkspaceRegistry:
    """One MergeCandidate → one worktree; never writes the main working tree files."""

    def __init__(self, repo_root: Path, git: GitAdapter | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.git = git or GitAdapter(self.repo_root)
        self.root = self.repo_root / WORKTREE_DIRNAME
        self._index_path = self.root / "registry.json"

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        gitkeep = self.root / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")
        return self.root

    def workspace_path(self, merge_candidate_id: str) -> Path:
        if not merge_candidate_id or "/" in merge_candidate_id or "\\" in merge_candidate_id:
            raise GitAdapterError("invalid merge_candidate_id for worktree path")
        if ".." in merge_candidate_id:
            raise GitAdapterError("path escape in merge_candidate_id")
        return (self.root / merge_candidate_id).resolve()

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.is_file():
            return {"workspaces": {}}
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"workspaces": {}}

    def _save_index(self, data: dict[str, Any]) -> None:
        self.ensure_root()
        self._index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, merge_candidate_id: str) -> dict[str, Any] | None:
        return dict(self._load_index().get("workspaces") or {}).get(merge_candidate_id)

    def list_all(self) -> list[dict[str, Any]]:
        items = list((self._load_index().get("workspaces") or {}).values())
        items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
        return items

    def create(
        self,
        *,
        merge_candidate_id: str,
        source_commit: str,
        patch_id: str,
        target_branch: str,
    ) -> dict[str, Any]:
        """Create a fresh detached worktree; refuse reuse of an existing path."""
        self.ensure_root()
        path = self.workspace_path(merge_candidate_id)
        if path.exists() or self.get(merge_candidate_id) is not None:
            raise GitAdapterError(
                f"worktree already registered for {merge_candidate_id}; "
                "refuse reuse (create a new MergeCandidate)"
            )
        # Capture main porcelain before/after to assert main was not dirtied by us.
        before = self.git.status_porcelain()
        self.git.worktree_add(path, source_commit)
        after = self.git.status_porcelain()
        if before != after:
            # Best-effort cleanup
            try:
                self.git.worktree_remove(path, force=True)
            except GitAdapterError:
                pass
            raise GitAdapterError(
                "main workspace status changed during worktree add; aborting"
            )
        record = {
            "merge_candidate_id": merge_candidate_id,
            "path": str(path),
            "source_commit": source_commit,
            "target_branch": target_branch,
            "patch_id": patch_id,
            "created_at": utc_now_iso(),
            "main_workspace_modified": False,
        }
        index = self._load_index()
        workspaces = dict(index.get("workspaces") or {})
        workspaces[merge_candidate_id] = record
        index["workspaces"] = workspaces
        self._save_index(index)
        return record

    def remove(self, merge_candidate_id: str, *, force: bool = True) -> None:
        record = self.get(merge_candidate_id)
        path = self.workspace_path(merge_candidate_id)
        if path.exists():
            self.git.worktree_remove(path, force=force)
        if record is not None:
            index = self._load_index()
            workspaces = dict(index.get("workspaces") or {})
            workspaces.pop(merge_candidate_id, None)
            index["workspaces"] = workspaces
            self._save_index(index)
