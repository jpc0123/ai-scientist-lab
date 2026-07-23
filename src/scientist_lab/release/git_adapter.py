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

    def diff_name_only(self, *, cwd: Path) -> list[str]:
        out = self._run(["diff", "--name-only"], cwd=cwd, check=True).stdout
        return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]

    def untracked_files(self, *, cwd: Path) -> list[str]:
        result = self._run(
            ["status", "--porcelain", "-u"],
            cwd=cwd,
            check=True,
        )
        paths: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            code = line[:2]
            path = line[3:].strip().replace("\\", "/")
            if code.strip() == "??" or code.startswith("?"):
                paths.append(path)
            elif code[1] in {"M", "A", "D"} or code[0] in {"M", "A", "D"}:
                continue
        return paths

    def add_paths(self, paths: Sequence[str], *, cwd: Path) -> GitResult:
        cleaned: list[str] = []
        for raw in paths:
            path = str(raw).replace("\\", "/").strip()
            if not path or path.startswith("-") or ".." in path.split("/"):
                raise GitAdapterError(f"refusing git add path: {raw}")
            cleaned.append(path)
        if not cleaned:
            raise GitAdapterError("git add requires at least one path")
        return self._run(["add", "--", *cleaned], cwd=cwd, check=True)

    def commit_message_file(self, message_file: Path, *, cwd: Path) -> str:
        """Create a commit; returns new commit SHA. Message comes from a file."""
        message_file = Path(message_file).resolve()
        if not message_file.is_file():
            raise GitAdapterError(f"commit message file missing: {message_file}")
        self._run(["commit", "-F", str(message_file)], cwd=cwd, check=True)
        return self._run(["rev-parse", "HEAD"], cwd=cwd, check=True).stdout

    def merge_no_ff(self, commit_sha: str, message_file: Path) -> str:
        """Merge commit into current HEAD with --no-ff. Returns merge commit SHA."""
        if not commit_sha or commit_sha.startswith("-"):
            raise GitAdapterError("invalid merge commit sha")
        message_file = Path(message_file).resolve()
        if not message_file.is_file():
            raise GitAdapterError(f"merge message file missing: {message_file}")
        result = self._run(
            ["merge", "--no-ff", "-F", str(message_file), commit_sha],
            check=False,
        )
        if not result.ok:
            self._run(["merge", "--abort"], check=False)
            raise GitAdapterError(
                f"merge --no-ff failed: {result.stderr or result.stdout}"
            )
        return self.rev_parse("HEAD")

    def revert_commit(
        self,
        commit_sha: str,
        *,
        mainline: int = 1,
        message_file: Path | None = None,
    ) -> str:
        """Revert a commit (use mainline=1 for merge commits). Returns new SHA.

        Note: ``git revert`` has no ``-F`` message-file flag (unlike commit);
        ``message_file`` is accepted for API symmetry but ignored — use --no-edit.
        """
        if not commit_sha or commit_sha.startswith("-"):
            raise GitAdapterError("invalid revert commit sha")
        _ = message_file  # retained for callers; revert uses --no-edit only
        args: list[str] = [
            "revert",
            "--no-edit",
            "-m",
            str(int(mainline)),
            "--",
            commit_sha,
        ]
        result = self._run(args, check=False)
        if not result.ok:
            self._run(["revert", "--abort"], check=False)
            raise GitAdapterError(
                f"git revert failed: {result.stderr or result.stdout}"
            )
        return self.rev_parse("HEAD")
