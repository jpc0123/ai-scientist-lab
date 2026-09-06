"""Replay Bundle export / load / redaction for real research loops (v2.1.7)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.llm.config import redact_secrets
from scientist_lab.llm.context_sanitizer import _scrub_obj
from scientist_lab.research_loop.errors import RealLoopValidationError


SECRET_LEAK_PATTERNS = (
    re.compile(r"(?i)\bsk-[a-z0-9]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9\-._~+/]+=*"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*(?!\[REDACTED\])\S+"),
    re.compile(r"(?i)authorization\s*[:=]\s*(?!\[REDACTED\])\S+"),
    re.compile(r"(?i)(password|worker[_-]?token)\s*[:=]\s*(?!\[REDACTED\])\S+"),
)

# Absolute host paths only — avoid false positives on JSON-escaped relative \\.
ABS_PATH_LEAK = re.compile(
    r"(?i)"
    r"("
    r"[a-z]:\\\\[^\"'\s]{3,}"  # JSON-escaped Windows drive path
    r"|[a-z]:\\[^\\\"'\s]{3,}"  # raw Windows drive path
    r"|/Users/[^\"'\s]{3,}"
    r"|/home/[^\"'\s]{3,}"
    r"|\\\\\\\\[a-z0-9._$-]+\\\\"  # JSON-escaped UNC
    r")"
)


def scrub_for_replay(value: Any) -> Any:
    """Deep scrub secrets and host paths for bundle export."""
    cleaned = _scrub_obj(value)
    if isinstance(cleaned, str):
        return redact_secrets(cleaned)
    if isinstance(cleaned, dict):
        return {k: scrub_for_replay(v) for k, v in cleaned.items()}
    if isinstance(cleaned, list):
        return [scrub_for_replay(v) for v in cleaned]
    return cleaned


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_bundle_for_secrets(root: Path) -> list[str]:
    """Return human-readable leak findings (empty means clean)."""
    findings: list[str] = []
    for path in sorted(root.rglob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            findings.append(f"{path.name}: unreadable ({exc})")
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        for pattern in SECRET_LEAK_PATTERNS:
            if pattern.search(text):
                findings.append(f"{rel}: secret-like pattern {pattern.pattern}")
                break
        # Absolute host paths are allowed only inside [REDACTED_PATH] markers.
        for match in ABS_PATH_LEAK.finditer(text):
            snippet = match.group(0)
            if "[REDACTED_PATH]" in snippet:
                continue
            findings.append(f"{rel}: absolute path leak near {snippet[:48]}")
            break
    return findings


def assert_bundle_redacted(root: Path) -> dict[str, Any]:
    findings = scan_bundle_for_secrets(root)
    if findings:
        raise RealLoopValidationError(
            "replay bundle failed secret/path redaction scan: "
            + "; ".join(findings[:5])
        )
    return {"ok": True, "findings": [], "secrets_redacted": True}


def load_replay_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    root = Path(bundle_dir)
    if not root.is_dir():
        raise RealLoopValidationError(f"replay bundle not found: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RealLoopValidationError(f"manifest.json missing in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session = {}
    session_path = root / "session.json"
    if session_path.is_file():
        session = json.loads(session_path.read_text(encoding="utf-8"))
    rounds: dict[str, dict[str, Any]] = {}
    for round_dir in sorted(root.glob("round_*")):
        if not round_dir.is_dir():
            continue
        payload: dict[str, Any] = {"dir": round_dir.name}
        for name in (
            "planning_context.json",
            "planner_response.json",
            "critic_response.json",
            "approved_candidate.json",
            "execution_summary.json",
            "evidence_summary.json",
            "feedback_summary.json",
            "feedback_verification.json",
            "provider_audit.json",
        ):
            path = round_dir / name
            if path.is_file():
                key = name.replace(".json", "")
                payload[key] = json.loads(path.read_text(encoding="utf-8"))
        rounds[round_dir.name] = payload
    llm_replays = []
    replay_dir = root / "llm_replays"
    if replay_dir.is_dir():
        for path in sorted(replay_dir.glob("*.json")):
            llm_replays.append(json.loads(path.read_text(encoding="utf-8")))
    report = {}
    report_path = root / "real_llm_closed_loop_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "bundle_dir": str(root),
        "manifest": manifest,
        "session": session,
        "rounds": rounds,
        "llm_replays": llm_replays,
        "report": report,
    }


def build_replay_bundle(
    *,
    session: Mapping[str, Any],
    rounds: list[Mapping[str, Any]],
    output_dir: Path | str,
    project_id: str,
    feedback_usage: list[Mapping[str, Any]] | None = None,
    llm_calls: list[Mapping[str, Any]] | None = None,
    plans: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_records: list[Mapping[str, Any]] | None = None,
    claim_matrix: Mapping[str, Any] | None = None,
    extra_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a redacted Replay Bundle directory and return export summary."""
    root = Path(output_dir)
    if root.exists():
        # Overwrite cleanly for deterministic CI exports.
        for child in root.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil

                shutil.rmtree(child)
    root.mkdir(parents=True, exist_ok=True)

    session_clean = scrub_for_replay(dict(session))
    file_hashes: dict[str, str] = {}
    file_hashes["session.json"] = _write_json(root / "session.json", session_clean)

    usage_by_round = {
        int(item.get("target_round_number") or 0): scrub_for_replay(dict(item))
        for item in (feedback_usage or [])
        if isinstance(item, Mapping)
    }
    plans = {str(k): scrub_for_replay(dict(v)) for k, v in (plans or {}).items()}
    evidence_by_id = {
        str(e.get("evidence_id")): scrub_for_replay(dict(e))
        for e in (evidence_records or [])
        if isinstance(e, Mapping) and e.get("evidence_id")
    }

    round_index: list[dict[str, Any]] = []
    for rnd in rounds:
        number = int(rnd.get("round_number") or 0)
        if number < 1:
            continue
        round_dir = root / f"round_{number}"
        round_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        context = scrub_for_replay(dict(rnd.get("planning_context_json") or {}))
        if context:
            rel = f"round_{number}/planning_context.json"
            file_hashes[rel] = _write_json(round_dir / "planning_context.json", context)
            written.append(rel)

        feedback = scrub_for_replay(dict(rnd.get("feedback_summary_json") or {}))
        if feedback:
            rel = f"round_{number}/feedback_summary.json"
            file_hashes[rel] = _write_json(round_dir / "feedback_summary.json", feedback)
            written.append(rel)

        audit = scrub_for_replay(dict(rnd.get("provider_audit_json") or {}))
        if audit:
            rel = f"round_{number}/provider_audit.json"
            file_hashes[rel] = _write_json(round_dir / "provider_audit.json", audit)
            written.append(rel)

        plan_id = rnd.get("plan_id")
        plan = plans.get(str(plan_id)) if plan_id else None
        if plan:
            rel = f"round_{number}/planner_response.json"
            file_hashes[rel] = _write_json(round_dir / "planner_response.json", plan)
            written.append(rel)
            approved_id = rnd.get("approved_candidate_id")
            candidates = list(plan.get("candidates") or [])
            approved = next(
                (
                    c
                    for c in candidates
                    if isinstance(c, dict) and c.get("candidate_id") == approved_id
                ),
                None,
            )
            if approved is None and candidates:
                approved = candidates[0] if isinstance(candidates[0], dict) else None
            if approved:
                rel = f"round_{number}/approved_candidate.json"
                file_hashes[rel] = _write_json(
                    round_dir / "approved_candidate.json", scrub_for_replay(approved)
                )
                written.append(rel)
            # Critic slice if present on plan payload.
            critic = plan.get("critic") or plan.get("review") or audit.get("critic")
            if critic:
                rel = f"round_{number}/critic_response.json"
                file_hashes[rel] = _write_json(
                    round_dir / "critic_response.json", scrub_for_replay(critic)
                )
                written.append(rel)

        exec_summary = scrub_for_replay(
            {
                "execution_node_id": rnd.get("execution_node_id"),
                "iteration_id": rnd.get("iteration_id"),
                "contract_id": rnd.get("contract_id"),
                "status": rnd.get("status"),
            }
        )
        rel = f"round_{number}/execution_summary.json"
        file_hashes[rel] = _write_json(round_dir / "execution_summary.json", exec_summary)
        written.append(rel)

        evidence_ids = [str(x) for x in (rnd.get("evidence_ids") or [])]
        evidence_payload = {
            "evidence_ids": evidence_ids,
            "claim_ids": list(rnd.get("claim_ids") or []),
            "records": [
                evidence_by_id[eid] for eid in evidence_ids if eid in evidence_by_id
            ],
        }
        if claim_matrix and number == 1:
            evidence_payload["claim_matrix"] = scrub_for_replay(dict(claim_matrix))
        rel = f"round_{number}/evidence_summary.json"
        file_hashes[rel] = _write_json(
            round_dir / "evidence_summary.json", scrub_for_replay(evidence_payload)
        )
        written.append(rel)

        if number in usage_by_round:
            rel = f"round_{number}/feedback_verification.json"
            file_hashes[rel] = _write_json(
                round_dir / "feedback_verification.json", usage_by_round[number]
            )
            written.append(rel)

        round_index.append(
            {
                "round_number": number,
                "round_id": rnd.get("round_id"),
                "status": rnd.get("status"),
                "plan_id": plan_id,
                "files": written,
            }
        )

    llm_index: list[dict[str, Any]] = []
    replay_dir = root / "llm_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for idx, call in enumerate(llm_calls or [], start=1):
        if not isinstance(call, Mapping):
            continue
        cleaned = scrub_for_replay(dict(call))
        call_id = str(cleaned.get("call_id") or f"replay_call_{idx:03d}")
        fingerprint = str(
            cleaned.get("request_fingerprint")
            or cleaned.get("fingerprint")
            or hashlib.sha256(
                json.dumps(cleaned, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
        )
        fname = f"{call_id}.json"
        rel = f"llm_replays/{fname}"
        file_hashes[rel] = _write_json(replay_dir / fname, cleaned)
        llm_index.append(
            {
                "call_id": call_id,
                "fingerprint": fingerprint,
                "path": rel,
                "provider": cleaned.get("provider") or cleaned.get("model_provider"),
            }
        )

    report = scrub_for_replay(
        {
            "schema_version": "v2.1.7",
            "session_id": session_clean.get("session_id"),
            "project_id": project_id,
            "status": session_clean.get("status"),
            "required_rounds": session_clean.get("required_rounds"),
            "real_only": session_clean.get("real_only"),
            "fallback_allowed": session_clean.get("fallback_allowed"),
            "fallback_used": session_clean.get("fallback_used"),
            "round_count": len(round_index),
            "feedback_usage_count": len(usage_by_round),
            "llm_replay_count": len(llm_index),
            "secrets_redacted": True,
            "exported_at": utc_now_iso(),
            **dict(extra_report or {}),
        }
    )
    file_hashes["real_llm_closed_loop_report.json"] = _write_json(
        root / "real_llm_closed_loop_report.json", report
    )

    manifest = {
        "schema_version": "replay_bundle_v1",
        "session_id": session_clean.get("session_id"),
        "project_id": project_id,
        "created_at": utc_now_iso(),
        "secrets_redacted": True,
        "files": file_hashes,
        "rounds": round_index,
        "llm_replays": llm_index,
        "feedback_usage_rounds": sorted(usage_by_round.keys()),
    }
    _write_json(root / "manifest.json", manifest)
    # Recompute manifest hash after write for self-description.
    manifest["manifest_sha256"] = _sha256_file(root / "manifest.json")
    _write_json(root / "manifest.json", manifest)

    assert_bundle_redacted(root)
    return {
        "ok": True,
        "bundle_dir": str(root),
        "session_id": session_clean.get("session_id"),
        "project_id": project_id,
        "manifest": manifest,
        "report": report,
        "file_count": len(file_hashes) + 1,
        "secrets_redacted": True,
    }
