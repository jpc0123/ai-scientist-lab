"""Patch fingerprinting and duplicate detection (v1.6.2)."""

from __future__ import annotations

import hashlib
import re


_INDEX_RE = re.compile(r"(?m)^index [0-9a-f]+\.\.[0-9a-f]+(?: \d+)?\n?")


def canonicalize_diff(unified_diff: str) -> str:
    """Normalize diff text for stable fingerprinting."""
    text = (unified_diff or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _INDEX_RE.sub("", text)
    # Drop trailing whitespace on each line but keep hunk markers.
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def fingerprint_diff(unified_diff: str) -> str:
    canonical = canonicalize_diff(unified_diff)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
