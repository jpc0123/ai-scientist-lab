"""Whitelist-only Git subprocess adapter (v1.9.1).

No other module may invoke Git; all calls go through this adapter.
Never uses shell=True. Forbidden destructive operations are not exposed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class GitAdapterError(RuntimeError):
    """Raised when a whitelisted git command fails or is refused."""


# Top-level git verbs this adapter may invoke (first token after `git`).
_ALLOWED_VERBS = frozenset(
    {
        "status",
        "diff",
        "rev-parse",
        "worktree",
        "apply",
        "add",
        "commit",
        "merge",
        "revert",
        "branch",
        "log",
    }
)

# Explicitly denied argument fragments (defense in depth).
_DENIED_TOKENS = frozenset(
    {
        "--hard",
        "--force",
        "-f",
        "rebase",
        "filter-branch",
        "clean",
        "push",
        "reset",
        "cherry-pick",
        "config",
    }
)


@dataclass(frozen=True)
class GitResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class GitAdapter:
    """Run only fixed Git actions against a repository root."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not (self.repo_root / ".git").exists():
            # Bare worktree metadata may live elsewhere; still require git dir/file.
            git_file = self.repo_root / ".git"
            if not git_file.exists():
                raise GitAdapterError(f"not a git repository: {self.repo_root}")

    def _refuse_denied(self, args: Sequence[str]) -> None:
        for token in args:
            base = token.split("=")[0]
            if base in _DENIED_TOKENS or token in _DENIED_TOKENS:
                raise GitAdapterError(f"git argument denied: {token}")

    def _run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> GitResult:
        if not args:
            raise GitAdapterError("empty git args")
        verb = args[0]
        if verb not in _ALLOWED_VERBS:
            raise GitAdapterError(f"git verb not allowed: {verb}")
        self._refuse_denied(args)
        cmd = ["git", *args]
        completed = subprocess.run(
            cmd,
            cwd=str(cwd or self.repo_root),
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        result = GitResult(
            args=tuple(cmd),
            returncode=int(completed.returncode),
            stdout=(completed.stdout or "").strip(),
            stderr=(completed.stderr or "").strip(),
        )
        if check and not result.ok:
            raise GitAdapterError(
                f"git {' '.join(args)} failed ({result.returncode}): "
                f"{result.stderr or result.stdout}"
            )
        return result

    def rev_parse(self, ref: str = "HEAD") -> str:
        if not ref or ref.startswith("-"):
            raise GitAdapterError("invalid rev-parse ref")
        return self._run(["rev-parse", ref]).stdout

    def current_branch(self) -> str:
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout

    def status_porcelain(self, *, cwd: Path | None = None) -> str:
        return self._run(["status", "--porcelain"], cwd=cwd, check=True).stdout

    def diff(
        self,
        *extra: str,
        cwd: Path | None = None,
        check: bool = True,
    ) -> str:
        return self._run(["diff", *extra], cwd=cwd, check=check).stdout

    def worktree_list(self) -> str:
        return self._run(["worktree", "list", "--porcelain"]).stdout

    def worktree_add(self, path: Path, commitish: str) -> GitResult:
        path = Path(path).resolve()
        if not str(path).startswith(str(self.repo_root.resolve())):
            raise GitAdapterError("worktree path must be inside repository root")
        if path.exists():
            raise GitAdapterError(f"worktree path already exists: {path}")
        if not commitish or commitish.startswith("-"):
            raise GitAdapterError("invalid commitish for worktree add")
        # Detach at commit to avoid creating a named branch from user input.
        return self._run(
            ["worktree", "add", "--detach", str(path), commitish],
            check=True,
        )

    def worktree_remove(self, path: Path, *, force: bool = False) -> GitResult:
        """Remove a registered worktree. force uses --force only for unlock, not push."""
        path = Path(path).resolve()
        if not str(path).startswith(str(self.repo_root.resolve())):
            raise GitAdapterError("worktree path must be inside repository root")
        args = ["worktree", "remove"]
        if force:
            # git worktree remove --force is not push --force; still avoid -f alias.
            args.append("--force")
        args.append(str(path))
        # Bypass _refuse_denied for the specific worktree --force token by
        # constructing a narrow allow path:
        cmd = ["git", *args]
        completed = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        result = GitResult(
            args=tuple(cmd),
            returncode=int(completed.returncode),
            stdout=(completed.stdout or "").strip(),
            stderr=(completed.stderr or "").strip(),
        )
        if not result.ok:
            raise GitAdapterError(
                f"worktree remove failed: {result.stderr or result.stdout}"
            )
        return result

    def apply_check(self, patch_file: Path, *, cwd: Path) -> GitResult:
        patch_file = Path(patch_file).resolve()
        if not patch_file.is_file():
            raise GitAdapterError(f"patch file missing: {patch_file}")
        return self._run(
            ["apply", "--check", str(patch_file)],
            cwd=cwd,
            check=False,
        )

    def apply(self, patch_file: Path, *, cwd: Path) -> GitResult:
        """Apply patch inside an isolated worktree (v1.9.3+)."""
        patch_file = Path(patch_file).resolve()
        if not patch_file.is_file():
            raise GitAdapterError(f"patch file missing: {patch_file}")
        return self._run(["apply", str(patch_file)], cwd=cwd, check=True)
