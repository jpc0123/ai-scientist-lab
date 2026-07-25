"""Diff content safety scanners for restricted patches (v2.2.3).

Checks secrets, prompt-injection markers, dangerous APIs, line budgets,
critical deletions, and new executables. Path allow/deny remains PathPolicy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scientist_lab.patching.models import DiffFile, ParsedDiff, VerificationIssue


# --- Limits -----------------------------------------------------------------

@dataclass(frozen=True)
class DiffSafetyLimits:
    max_added_lines: int = 200
    max_removed_lines: int = 200
    max_files: int = 10
    max_hunks_per_file: int = 40


DEFAULT_DIFF_SAFETY_LIMITS = DiffSafetyLimits()


# --- Patterns ---------------------------------------------------------------

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    (
        "openai_sk_key",
        re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    ),
    (
        "generic_api_key_assign",
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key|password)\b\s*[=:]\s*['\"][^'\"]{8,}"
        ),
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*"),
    ),
)

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions"),
    ),
    (
        "system_prompt_override",
        re.compile(r"(?i)(system\s*prompt|developer\s*message)\s*[:=]"),
    ),
    (
        "jailbreak_marker",
        re.compile(r"(?i)\b(dan\s*mode|jailbreak|bypass\s+safety)\b"),
    ),
)

_DANGEROUS_API_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("os_system", re.compile(r"\bos\.system\s*\(")),
    ("subprocess_call", re.compile(r"\bsubprocess\.(?:call|run|Popen|check_output)\s*\(")),
    ("eval_call", re.compile(r"(?<![\w.])eval\s*\(")),
    ("exec_call", re.compile(r"(?<![\w.])exec\s*\(")),
    ("compile_exec", re.compile(r"\bcompile\s*\([^)]*\)\s*;?\s*exec")),
    ("import_os_dunder", re.compile(r"__import__\s*\(\s*['\"]os['\"]\s*\)")),
    ("ctypes_windll", re.compile(r"\bctypes\.(?:windll|cdll)\b")),
    ("socket_connect", re.compile(r"\bsocket\.socket\s*\(")),
    ("requests_post_exfil", re.compile(r"(?i)\brequests\.(?:post|put|patch)\s*\(")),
    ("powershell", re.compile(r"(?i)\bpowershell\b|\bbash\s+-c\b|/bin/sh\b")),
)

_EXECUTABLE_SUFFIXES = (
    ".exe",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    ".bash",
    ".msi",
    ".com",
    ".dll",
    ".so",
    ".dylib",
)

_CRITICAL_DELETE_NAMES = frozenset(
    {
        "pyproject.toml",
        "requirements.txt",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env",
        "scientist_lab.db",
    }
)

_CRITICAL_DELETE_PREFIXES = (
    ".github/",
    ".git/",
    "src/scientist_lab/llm/",
)


@dataclass
class DiffLineStats:
    added_lines: int = 0
    removed_lines: int = 0
    files: int = 0
    hunks: int = 0
    new_file_modes: list[str] = field(default_factory=list)


def iter_added_line_bodies(diff_file: DiffFile) -> list[str]:
    bodies: list[str] = []
    for hunk in diff_file.hunks:
        for line in hunk.lines:
            if line.startswith("+") and not line.startswith("+++"):
                bodies.append(line[1:])
    return bodies


def collect_line_stats(parsed: ParsedDiff) -> DiffLineStats:
    stats = DiffLineStats(files=len(parsed.files))
    for diff_file in parsed.files:
        stats.hunks += len(diff_file.hunks)
        for hunk in diff_file.hunks:
            for line in hunk.lines:
                if line.startswith("+") and not line.startswith("+++"):
                    stats.added_lines += 1
                elif line.startswith("-") and not line.startswith("---"):
                    stats.removed_lines += 1
    return stats


def scan_text_for_secrets(text: str, *, path: str | None = None) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    for code, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            issues.append(
                VerificationIssue(
                    code=f"secret_{code}",
                    message=f"possible secret detected ({code})",
                    path=path,
                    blocking=True,
                )
            )
    return issues


def scan_text_for_injection(text: str, *, path: str | None = None) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    for code, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            issues.append(
                VerificationIssue(
                    code=f"prompt_injection_{code}",
                    message=f"prompt-injection marker detected ({code})",
                    path=path,
                    blocking=True,
                )
            )
    return issues


def scan_text_for_dangerous_apis(
    text: str, *, path: str | None = None
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    for code, pattern in _DANGEROUS_API_PATTERNS:
        if pattern.search(text):
            issues.append(
                VerificationIssue(
                    code=f"dangerous_api_{code}",
                    message=f"forbidden API/pattern in added line: {code}",
                    path=path,
                    blocking=True,
                )
            )
    return issues


def _is_critical_delete(path: str) -> bool:
    low = path.replace("\\", "/").lower()
    name = low.rsplit("/", 1)[-1]
    if name in _CRITICAL_DELETE_NAMES:
        return True
    return any(low.startswith(prefix) for prefix in _CRITICAL_DELETE_PREFIXES)


def _is_executable_path(path: str) -> bool:
    low = path.replace("\\", "/").lower()
    return any(low.endswith(suffix) for suffix in _EXECUTABLE_SUFFIXES)


def scan_parsed_diff(
    parsed: ParsedDiff,
    *,
    limits: DiffSafetyLimits | None = None,
    raw_diff: str = "",
) -> list[VerificationIssue]:
    """Return blocking VerificationIssue list for content-level risks."""
    limits = limits or DEFAULT_DIFF_SAFETY_LIMITS
    issues: list[VerificationIssue] = []
    stats = collect_line_stats(parsed)

    if stats.files > limits.max_files:
        issues.append(
            VerificationIssue(
                code="too_many_files",
                message=f"files={stats.files} exceeds max_files={limits.max_files}",
                blocking=True,
            )
        )
    if stats.added_lines > limits.max_added_lines:
        issues.append(
            VerificationIssue(
                code="too_many_added_lines",
                message=(
                    f"added_lines={stats.added_lines} exceeds "
                    f"max_added_lines={limits.max_added_lines}"
                ),
                blocking=True,
            )
        )
    if stats.removed_lines > limits.max_removed_lines:
        issues.append(
            VerificationIssue(
                code="too_many_removed_lines",
                message=(
                    f"removed_lines={stats.removed_lines} exceeds "
                    f"max_removed_lines={limits.max_removed_lines}"
                ),
                blocking=True,
            )
        )

    # Executable mode bits in raw diff header.
    for match in re.finditer(
        r"^new file mode (100[75]55)$", raw_diff or "", flags=re.MULTILINE
    ):
        issues.append(
            VerificationIssue(
                code="new_executable_mode",
                message=f"new executable file mode forbidden: {match.group(1)}",
                blocking=True,
            )
        )

    for diff_file in parsed.files:
        path = diff_file.path
        if len(diff_file.hunks) > limits.max_hunks_per_file:
            issues.append(
                VerificationIssue(
                    code="too_many_hunks",
                    message=(
                        f"hunks={len(diff_file.hunks)} exceeds "
                        f"max_hunks_per_file={limits.max_hunks_per_file}"
                    ),
                    path=path or None,
                    blocking=True,
                )
            )

        if diff_file.is_deleted_file and path and _is_critical_delete(path):
            issues.append(
                VerificationIssue(
                    code="critical_file_delete",
                    message=f"deleting critical path is forbidden: {path}",
                    path=path,
                    blocking=True,
                )
            )

        if diff_file.is_new_file and path and _is_executable_path(path):
            issues.append(
                VerificationIssue(
                    code="new_executable_file",
                    message=f"new executable/script path forbidden: {path}",
                    path=path,
                    blocking=True,
                )
            )

        for body in iter_added_line_bodies(diff_file):
            issues.extend(scan_text_for_secrets(body, path=path or None))
            issues.extend(scan_text_for_injection(body, path=path or None))
            issues.extend(scan_text_for_dangerous_apis(body, path=path or None))

    return issues
