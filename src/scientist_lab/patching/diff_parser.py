"""Unified Diff parser for PatchProposal (v1.6.2)."""

from __future__ import annotations

import re

from scientist_lab.patching.models import DiffFile, DiffHunk, ParsedDiff


_HUNK_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
)


class DiffParseError(ValueError):
    """Raised when a unified diff cannot be parsed safely."""


def _strip_prefix(path: str) -> str:
    text = (path or "").strip()
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    return text.replace("\\", "/")


def parse_unified_diff(text: str) -> ParsedDiff:
    """Parse a Unified Diff into structured files/hunks.

    Rejects empty diffs. Binary markers are recorded as ``is_binary=True``.
    """
    raw = text or ""
    if not raw.strip():
        raise DiffParseError("empty unified diff")

    lines = raw.splitlines()
    files: list[DiffFile] = []
    current: DiffFile | None = None
    hunk: DiffHunk | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            if current is not None:
                files.append(current)
            current = DiffFile()
            hunk = None
            parts = line.split()
            # diff --git a/foo b/foo
            if len(parts) >= 4:
                current.old_path = _strip_prefix(parts[2])
                current.new_path = _strip_prefix(parts[3])
            i += 1
            continue
        if line.startswith("--- "):
            if current is None:
                current = DiffFile()
            path = line[4:].strip()
            if path != "/dev/null":
                current.old_path = _strip_prefix(path)
            else:
                current.is_new_file = True
            i += 1
            continue
        if line.startswith("+++ "):
            if current is None:
                current = DiffFile()
            path = line[4:].strip()
            if path != "/dev/null":
                current.new_path = _strip_prefix(path)
            else:
                current.is_deleted_file = True
            i += 1
            continue
        if line.startswith("new file mode"):
            if current is not None:
                current.is_new_file = True
            i += 1
            continue
        if line.startswith("deleted file mode"):
            if current is not None:
                current.is_deleted_file = True
            i += 1
            continue
        if "Binary files" in line or line.startswith("GIT binary patch"):
            if current is None:
                current = DiffFile()
            current.is_binary = True
            i += 1
            continue
        match = _HUNK_RE.match(line)
        if match:
            if current is None:
                current = DiffFile()
            hunk = DiffHunk(
                header=line,
                old_start=int(match.group(1)),
                old_count=int(match.group(2) or "1"),
                new_start=int(match.group(3)),
                new_count=int(match.group(4) or "1"),
                lines=[],
            )
            current.hunks.append(hunk)
            i += 1
            continue
        if hunk is not None and (
            line.startswith("+")
            or line.startswith("-")
            or line.startswith(" ")
            or line == "\\ No newline at end of file"
        ):
            hunk.lines.append(line)
            i += 1
            continue
        # Ignore other metadata (index, similarity, etc.)
        i += 1

    if current is not None:
        files.append(current)

    if not files:
        raise DiffParseError("no file sections found in unified diff")

    for item in files:
        if not item.path and not item.is_binary:
            raise DiffParseError("diff file missing path")

    return ParsedDiff(files=files, raw=raw)
