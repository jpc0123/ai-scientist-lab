"""EvidenceValidator: VALID / INVALID / INCOMPLETE / NOT_APPLICABLE.

Does not KEEP/DISCARD. Reviewer may run only after VALID.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from scientist_lab.core.state_machine import EvidenceStatus


@dataclass(frozen=True)
class EvidenceVerdict:
    evidence_status: str
    reasons: tuple[str, ...] = ()
    review_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_status": self.evidence_status,
            "reasons": list(self.reasons),
            "review_allowed": self.review_allowed,
        }


def _finite(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


class EvidenceValidator:
    def validate(
        self,
        result: Mapping[str, Any],
        *,
        protocol: Mapping[str, Any] | None = None,
        expected_artifacts: list[str] | None = None,
        fingerprint_comparable: bool | None = None,
        dry_run: bool = False,
        handle_status: str | None = None,
    ) -> EvidenceVerdict:
        reasons: list[str] = []
        execution = dict(result.get("execution") or {})
        exec_status = str(execution.get("status") or "failed")
        handle_status = str(handle_status or exec_status)
        metrics = dict(result.get("metrics") or {})
        artifacts = dict(result.get("artifacts") or {})
        missing = list(artifacts.get("missing_expected") or [])
        primary = ((protocol or {}).get("objective") or {}).get("primary") or {}
        primary_metric = primary.get("metric")
        primary_value = metrics.get(primary_metric) if primary_metric else None

        if handle_status in {"failed", "timeout", "timed_out", "cancelled"} or exec_status in {
            "failed",
            "timeout",
            "cancelled",
        }:
            if execution.get("error_type"):
                reasons.append(f"execution {exec_status}: {execution.get('error_type')}")
            else:
                reasons.append(f"execution {handle_status or exec_status}; no scientific evidence")
            return EvidenceVerdict(EvidenceStatus.NOT_APPLICABLE.value, tuple(reasons), False)

        if dry_run and handle_status == "dry_run":
            reasons.append("dry_run produced no recovered artifacts")
            return EvidenceVerdict(EvidenceStatus.NOT_APPLICABLE.value, tuple(reasons), False)
        # dry_run + recovered artifacts arrive as handle_status=completed; continue to VALID checks.

        if fingerprint_comparable is False:
            reasons.append("fingerprint not comparable; numbers are not a valid comparison")
            return EvidenceVerdict(EvidenceStatus.INVALID.value, tuple(reasons), False)

        if primary_metric and primary_value is not None and not _finite(primary_value):
            reasons.append(f"primary metric {primary_metric} is not a finite number")
            return EvidenceVerdict(EvidenceStatus.INVALID.value, tuple(reasons), False)

        if expected_artifacts:
            paths = " ".join(str(p) for p in (artifacts.get("paths") or []))
            still_missing = [
                name for name in expected_artifacts if name not in paths and name not in missing
            ]
            missing = list(dict.fromkeys([*missing, *still_missing]))

        if missing:
            reasons.append(f"missing expected artifacts: {missing}")
            return EvidenceVerdict(EvidenceStatus.INCOMPLETE.value, tuple(reasons), False)

        if primary_metric and primary_value is None:
            reasons.append(f"primary metric {primary_metric} absent")
            return EvidenceVerdict(EvidenceStatus.INCOMPLETE.value, tuple(reasons), False)

        return EvidenceVerdict(EvidenceStatus.VALID.value, tuple(reasons), True)
