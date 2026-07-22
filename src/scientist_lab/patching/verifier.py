"""Static PatchVerifier — path policy + diff safety (v1.6.3)."""

from __future__ import annotations

from datetime import datetime, timezone

from scientist_lab.patching.diff_parser import DiffParseError, parse_unified_diff
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import PatchVerification, VerificationIssue
from scientist_lab.patching.path_policy import PathPolicy


_SHELLISH = (
    "os.system(",
    "subprocess.",
    "bash -c",
    "powershell",
    "/bin/sh",
    "eval(",
    "__import__('os')",
)


class PatchVerifier:
    def __init__(self, policy: PathPolicy | None = None) -> None:
        self.policy = policy or PathPolicy()

    def verify(
        self,
        unified_diff: str,
        *,
        duplicate_of: str | None = None,
    ) -> PatchVerification:
        issues: list[VerificationIssue] = []
        files_touched: list[str] = []
        fp = fingerprint_diff(unified_diff)

        try:
            parsed = parse_unified_diff(unified_diff)
        except DiffParseError as exc:
            issues.append(
                VerificationIssue(
                    code="diff_parse_error",
                    message=str(exc),
                    blocking=True,
                )
            )
            return PatchVerification(
                ok=False,
                issues=issues,
                fingerprint_sha256=fp,
                duplicate_of=duplicate_of,
                checked_at=datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
            )

        for diff_file in parsed.files:
            path = diff_file.path
            if path:
                files_touched.append(path)
            if diff_file.is_binary:
                issues.append(
                    VerificationIssue(
                        code="binary_file",
                        message="binary patch is forbidden",
                        path=path or None,
                        blocking=True,
                    )
                )
                continue
            if not path:
                issues.append(
                    VerificationIssue(
                        code="missing_path",
                        message="diff file missing path",
                        blocking=True,
                    )
                )
                continue
            ok, reason = self.policy.is_allowed(path)
            if not ok:
                issues.append(
                    VerificationIssue(
                        code="path_policy_violation",
                        message=reason,
                        path=path,
                        blocking=True,
                    )
                )

            # Scan added lines for obvious shell/exec payloads.
            for hunk in diff_file.hunks:
                for line in hunk.lines:
                    if not line.startswith("+") or line.startswith("+++"):
                        continue
                    body = line[1:]
                    low = body.lower()
                    for token in _SHELLISH:
                        if token.lower() in low:
                            issues.append(
                                VerificationIssue(
                                    code="dangerous_content",
                                    message=f"forbidden pattern in added line: {token}",
                                    path=path,
                                    blocking=True,
                                )
                            )
                            break

        if duplicate_of:
            issues.append(
                VerificationIssue(
                    code="duplicate_patch",
                    message=f"duplicate of existing patch {duplicate_of}",
                    blocking=True,
                )
            )

        # Deduplicate issue messages
        unique: list[VerificationIssue] = []
        seen: set[str] = set()
        for issue in issues:
            key = f"{issue.code}:{issue.path}:{issue.message}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(issue)

        blocking = [i for i in unique if i.blocking]
        return PatchVerification(
            ok=not blocking,
            issues=unique,
            files_touched=sorted(set(files_touched)),
            fingerprint_sha256=fp,
            duplicate_of=duplicate_of,
            checked_at=datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        )
