"""Static PatchVerifier — path policy + diff safety (v1.6.3 / v2.2.3)."""

from __future__ import annotations

from datetime import datetime, timezone

from scientist_lab.patching.diff_parser import DiffParseError, parse_unified_diff
from scientist_lab.patching.diff_safety import (
    DEFAULT_DIFF_SAFETY_LIMITS,
    DiffSafetyLimits,
    scan_parsed_diff,
)
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import PatchVerification, VerificationIssue
from scientist_lab.patching.path_policy import PathPolicy


class PatchVerifier:
    def __init__(
        self,
        policy: PathPolicy | None = None,
        *,
        limits: DiffSafetyLimits | None = None,
    ) -> None:
        self.policy = policy or PathPolicy()
        self.limits = limits or DEFAULT_DIFF_SAFETY_LIMITS

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

        # v2.2.3 content / budget / secret / injection / dangerous API scans.
        issues.extend(
            scan_parsed_diff(
                parsed,
                limits=self.limits,
                raw_diff=unified_diff,
            )
        )

        if duplicate_of:
            issues.append(
                VerificationIssue(
                    code="duplicate_patch",
                    message=f"duplicate of existing patch {duplicate_of}",
                    blocking=True,
                )
            )

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
