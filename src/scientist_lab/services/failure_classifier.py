"""Failure classifier for experiment runs.

Separates engineering faults (safe to auto-recover) from scientific / policy
failures (report only; wait for explicit user confirmation).

Never silently re-runs expensive GPU training for scientific outcomes.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

FailureKind = Literal[
    "engineering",
    "scientific",
    "user_pause",
    "infra",
    "unknown",
]

Action = Literal[
    "auto_recover_wait_worker",  # CLI died; worker may still be running
    "auto_retry_once",  # known pre-train / config engineering fault
    "wait_user",  # scientific or ambiguous — do not auto-rerun
    "abort",  # hard stop
]


@dataclass
class FailureClassification:
    kind: FailureKind
    action: Action
    reason_code: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    auto_rerun_allowed: bool = False
    notes: str = ""
    classified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Ordered rules: first match wins.
_RULES: list[tuple[str, FailureKind, Action, str, float, bool, re.Pattern[str]]] = [
    (
        "user_cancelled",
        "user_pause",
        "wait_user",
        "Job/exec cancelled by user or explicit pause.",
        0.95,
        False,
        re.compile(
            r"(paused_by_user|cancel_requested|status.: .cancelled.|user_requested_pause)",
            re.I,
        ),
    ),
    (
        "binding_json_corrupt",
        "engineering",
        "auto_recover_wait_worker",
        "RemoteJobBinding JSON empty/corrupt during CLI poll (non-atomic write race).",
        0.98,
        False,
        re.compile(
            r"(RemoteJobBinding|remote binding).*(Invalid JSON|EOF while parsing|empty)|"
            r"Invalid JSON: EOF while parsing a value",
            re.I,
        ),
    ),
    (
        "yamlconfig_no_setter",
        "engineering",
        "auto_retry_once",
        "YAMLConfig.train_dataloader has no setter; use _train_dataloader assignment.",
        0.95,
        True,
        re.compile(
            r"property 'train_dataloader' of 'YAMLConfig' object has no setter",
            re.I,
        ),
    ),
    (
        "deterministic_op_missing",
        "engineering",
        "auto_retry_once",
        "torch.use_deterministic_algorithms strict mode hits unsupported CUDA op "
        "(e.g. grid_sampler); use warn_only for DFINE diagnosis.",
        0.95,
        True,
        re.compile(
            r"does not have a deterministic implementation|"
            r"grid_sampler_2d_backward_cuda|"
            r"use_deterministic_algorithms\(True\)",
            re.I,
        ),
    ),
    (
        "module_import_path",
        "engineering",
        "auto_retry_once",
        "Host CLI missing PYTHONPATH / scientist_lab import.",
        0.9,
        True,
        re.compile(
            r"ModuleNotFoundError: No module named ['\"]scientist_lab['\"]|"
            r"can't find '__main__' module in",
            re.I,
        ),
    ),
    (
        "worker_unreachable",
        "infra",
        "wait_user",
        "Scientist Worker HTTP unreachable; start worker then resume.",
        0.9,
        False,
        re.compile(
            r"(Connection refused|Failed to establish a new connection|"
            r"RemoteWorkerUnavailable|worker unhealthy|"
            r"actively refused|无法连接到远程服务器)",
            re.I,
        ),
    ),
    (
        "docker_daemon_down",
        "infra",
        "wait_user",
        "Docker engine not available.",
        0.92,
        False,
        re.compile(
            r"dockerDesktopLinuxEngine|Cannot connect to the Docker daemon|"
            r"error during connect|Is the docker daemon running",
            re.I,
        ),
    ),
    (
        "oom_killed",
        "infra",
        "wait_user",
        "Process/container OOM-killed; needs budget/batch change (user confirm).",
        0.85,
        False,
        re.compile(r"(OOMKilled|CUDA out of memory|torch\.cuda\.OutOfMemoryError)", re.I),
    ),
    (
        "disk_gate",
        "infra",
        "wait_user",
        "Disk gate blocked formal train; free space then resume.",
        0.9,
        False,
        re.compile(r"(disk_gate|enforce_disk_gate|No space left on device)", re.I),
    ),
    (
        "protocol_validation",
        "engineering",
        "wait_user",
        "Protocol/contract validation failed; fix fixed_parameters then re-run.",
        0.85,
        False,
        re.compile(
            r"(ProtocolViolation|protocol validation|fixed_parameters|"
            r"missing execution_id|create-protocol)",
            re.I,
        ),
    ),
    (
        "nan_or_collapse",
        "scientific",
        "wait_user",
        "Training numeric collapse / NaN — scientific review required.",
        0.8,
        False,
        re.compile(
            r"(\bNaN\b|loss.*nan|metric.*nan|collapsed|early peak/collapse)",
            re.I,
        ),
    ),
    (
        "claim_or_gate_block",
        "scientific",
        "wait_user",
        "Scientific claim/gate blocked; do not auto-rerun.",
        0.85,
        False,
        re.compile(
            r"(claim_gate|allow_scientific_claims|formal_performance|"
            r"diagnose_seed_instability|inconclusive_need)",
            re.I,
        ),
    ),
    (
        "container_exit_generic",
        "unknown",
        "wait_user",
        "Container exited non-zero; inspect log before any retry.",
        0.55,
        False,
        re.compile(r"(container exit code [1-9]|\[worker\] failed)", re.I),
    ),
]


def _clip(text: str, n: int = 12_000) -> str:
    if len(text) <= n:
        return text
    return text[-n:]


def classify_failure(
    *,
    log_text: str = "",
    error_message: str = "",
    cli_text: str = "",
    worker_status: str | None = None,
    metrics_present: bool = False,
    exec_status: str | None = None,
) -> FailureClassification:
    """Classify a failed / incomplete experiment attempt."""
    blob = _clip("\n".join([error_message or "", cli_text or "", log_text or ""]))
    evidence: list[str] = []

    ws = (worker_status or "").lower()
    es = (exec_status or "").lower()
    if metrics_present:
        return FailureClassification(
            kind="engineering",
            action="auto_recover_wait_worker",
            reason_code="metrics_already_present",
            summary="Metrics already on disk; treat as recovered success side-path.",
            evidence=["metrics.json present"],
            confidence=0.99,
            auto_rerun_allowed=False,
            notes="Do not relaunch; reuse execution.",
        )
    if ws in {"running", "queued", "preparing", "collecting", "received", "validating"}:
        return FailureClassification(
            kind="engineering",
            action="auto_recover_wait_worker",
            reason_code="worker_still_active",
            summary="Host CLI failed but worker job is still active — wait & recover.",
            evidence=[f"worker_status={ws}"],
            confidence=0.93,
            auto_rerun_allowed=False,
        )
    if ws == "completed" and not metrics_present:
        return FailureClassification(
            kind="engineering",
            action="auto_recover_wait_worker",
            reason_code="worker_completed_host_missing_metrics",
            summary="Worker completed; host missing metrics — pull/recover artifacts.",
            evidence=[f"worker_status={ws}"],
            confidence=0.92,
            auto_rerun_allowed=False,
        )
    if es in {"cancelled"} or ws in {"cancelled", "cancel_requested"}:
        return FailureClassification(
            kind="user_pause",
            action="wait_user",
            reason_code="user_cancelled",
            summary="Execution cancelled; resume only after explicit user request.",
            evidence=[f"exec_status={es}", f"worker_status={ws}"],
            confidence=0.95,
            auto_rerun_allowed=False,
        )

    for code, kind, action, summary, conf, rerun, pat in _RULES:
        m = pat.search(blob)
        if not m:
            continue
        evidence.append(m.group(0)[:240])
        return FailureClassification(
            kind=kind,
            action=action,
            reason_code=code,
            summary=summary,
            evidence=evidence,
            confidence=conf,
            auto_rerun_allowed=bool(rerun and action == "auto_retry_once"),
            notes=(
                "Auto-retry limited to one attempt for known engineering faults "
                "that fail before meaningful training progress."
                if rerun
                else "Requires user confirmation before any expensive re-run."
            ),
        )

    return FailureClassification(
        kind="unknown",
        action="wait_user",
        reason_code="unclassified",
        summary="No rule matched; default to wait_user (no auto GPU re-run).",
        evidence=[blob[-200:]] if blob else [],
        confidence=0.3,
        auto_rerun_allowed=False,
    )


def write_classification(
    out_dir: Path,
    classification: FailureClassification,
    *,
    context: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "FAILURE_CLASSIFICATION.json"
    payload = {
        "classification": classification.to_dict(),
        "context": context or {},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if classification.action == "wait_user":
        wait_path = out_dir / "WAIT_USER.json"
        wait_path.write_text(
            json.dumps(
                {
                    "status": "waiting_for_user",
                    "reason_code": classification.reason_code,
                    "kind": classification.kind,
                    "summary": classification.summary,
                    "created_at": classification.classified_at,
                    "resume_hint": (
                        "Create RESUME_APPROVED.json with "
                        '{"approve": true, "action": "retry"|"continue"} '
                        "or relaunch autochain explicitly."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return path


def user_approved_resume(out_dir: Path) -> dict[str, Any] | None:
    path = Path(out_dir) / "RESUME_APPROVED.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("approve") is True:
        return data
    return None
