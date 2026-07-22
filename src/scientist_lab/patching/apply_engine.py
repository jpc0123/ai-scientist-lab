"""Pure-Python unified-diff apply engine for sandbox workspaces (v1.6.4)."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.patching.diff_parser import DiffParseError, parse_unified_diff
from scientist_lab.patching.models import DiffFile, ParsedDiff
from scientist_lab.patching.path_policy import PathPolicy


class PatchApplyError(ValueError):
    """Raised when a patch cannot be applied safely inside a sandbox."""


def _safe_resolve(root: Path, relative: str, policy: PathPolicy) -> Path:
    ok, reason = policy.is_allowed(relative)
    if not ok:
        raise PatchApplyError(f"path policy blocked during apply: {relative} ({reason})")
    target = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise PatchApplyError(
            f"path escapes sandbox: {relative}"
        ) from exc
    return target


def _apply_hunks_to_lines(original: list[str], diff_file: DiffFile) -> list[str]:
    """Apply hunks in reverse order so line offsets remain valid."""
    lines = list(original)
    for hunk in reversed(diff_file.hunks):
        # Collect new segment from hunk lines.
        new_segment: list[str] = []
        old_segment: list[str] = []
        for raw in hunk.lines:
            if raw.startswith("\\"):
                continue
            if raw.startswith("+"):
                new_segment.append(raw[1:] + ("\n" if not raw[1:].endswith("\n") else ""))
            elif raw.startswith("-"):
                old_segment.append(raw[1:])
            elif raw.startswith(" "):
                text = raw[1:]
                new_segment.append(text + ("\n" if not text.endswith("\n") else ""))
                old_segment.append(text)
            else:
                # Treat unmarked as context
                new_segment.append(raw + ("\n" if not raw.endswith("\n") else ""))
                old_segment.append(raw)

        # Normalize: working lines without trailing newline markers for matching
        def strip_nl(seq: list[str]) -> list[str]:
            return [s[:-1] if s.endswith("\n") else s for s in seq]

        old_plain = strip_nl(old_segment)
        start = max(0, hunk.old_start - 1)
        end = start + len(old_plain)

        if diff_file.is_new_file and not original:
            # Entire file is new content from + lines only.
            plus_only = [
                (raw[1:] if raw.startswith("+") else raw[1:] if raw.startswith(" ") else "")
                for raw in hunk.lines
                if raw.startswith("+") or raw.startswith(" ")
            ]
            return [line if line.endswith("\n") else line + "\n" for line in plus_only if line is not None]

        actual = lines[start:end]
        actual_plain = [a[:-1] if a.endswith("\n") else a for a in actual]
        if actual_plain != old_plain:
            # Fuzzy: search nearby for old_plain
            found = None
            window = 20
            lo = max(0, start - window)
            hi = min(len(lines), start + window)
            for idx in range(lo, hi + 1):
                chunk = lines[idx : idx + len(old_plain)]
                chunk_plain = [c[:-1] if c.endswith("\n") else c for c in chunk]
                if chunk_plain == old_plain:
                    found = idx
                    break
            if found is None and old_plain:
                raise PatchApplyError(
                    f"hunk context mismatch for {diff_file.path} at line {hunk.old_start}"
                )
            start = found if found is not None else start
            end = start + len(old_plain)

        new_lines = [
            (s if s.endswith("\n") else s + "\n") for s in strip_nl(new_segment)
        ]
        lines[start:end] = new_lines
    return lines


def apply_new_file(diff_file: DiffFile) -> str:
    content_lines: list[str] = []
    for hunk in diff_file.hunks:
        for raw in hunk.lines:
            if raw.startswith("+") and not raw.startswith("+++"):
                content_lines.append(raw[1:] + "\n")
            elif raw.startswith(" ") and not raw.startswith("+++"):
                content_lines.append(raw[1:] + "\n")
    return "".join(content_lines)


def apply_parsed_diff_to_root(
    parsed: ParsedDiff,
    *,
    root: Path,
    policy: PathPolicy | None = None,
) -> list[str]:
    """Apply a parsed diff under ``root``. Returns list of relative paths written."""
    policy = policy or PathPolicy()
    touched: list[str] = []
    for diff_file in parsed.files:
        if diff_file.is_binary:
            raise PatchApplyError(f"refusing binary file: {diff_file.path}")
        rel = policy.normalize(diff_file.path)
        target = _safe_resolve(root, rel, policy)
        if diff_file.is_deleted_file:
            if target.exists():
                target.unlink()
            touched.append(rel)
            continue
        if diff_file.is_new_file or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(apply_new_file(diff_file), encoding="utf-8")
            touched.append(rel)
            continue
        original_text = target.read_text(encoding="utf-8")
        # Preserve whether file ended with newline
        original_lines = original_text.splitlines(keepends=True)
        if original_text and not original_lines:
            original_lines = [original_text]
        updated = _apply_hunks_to_lines(original_lines, diff_file)
        target.write_text("".join(updated), encoding="utf-8")
        touched.append(rel)
    return touched


def apply_unified_diff_to_root(
    unified_diff: str,
    *,
    root: Path,
    policy: PathPolicy | None = None,
) -> list[str]:
    try:
        parsed = parse_unified_diff(unified_diff)
    except DiffParseError as exc:
        raise PatchApplyError(str(exc)) from exc
    return apply_parsed_diff_to_root(parsed, root=root, policy=policy)
