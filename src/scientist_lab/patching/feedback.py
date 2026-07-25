"""PatchEvidence → lab feedback bridges (v2.2.7).

Feeds sandbox patch outcomes into Evidence, Claim drafts, Planner context,
and experiment-tree notes — without modifying the main workspace.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from scientist_lab.domain.models import new_id
from scientist_lab.evidence.models import EvidenceRecord, ScientificClaim
from scientist_lab.patching.evidence import PatchEvidence
from scientist_lab.patching.models import PatchProposal
from scientist_lab.storage.artifact_store import write_json


PatchVerdict = Literal["effective", "ineffective", "inconclusive"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def derive_patch_verdict(evidence: PatchEvidence) -> PatchVerdict:
    """Heuristic: tests ok → effective (sandbox-only); failed → ineffective; else inconclusive."""
    if evidence.sandbox_tests_ok is True and evidence.evidence_strength in {
        "moderate",
        "strong",
    }:
        return "effective"
    if evidence.sandbox_tests_ok is False:
        return "ineffective"
    return "inconclusive"


def patch_evidence_to_lab_record(
    evidence: PatchEvidence,
    *,
    proposal: PatchProposal | None = None,
) -> EvidenceRecord:
    """Map PatchEvidence into a durable EvidenceRecord (failure_analysis)."""
    meta = dict((proposal.metadata if proposal else None) or {})
    verdict = derive_patch_verdict(evidence)
    metric_summary = {
        "kind": "sandbox_patch",
        "patch_id": evidence.patch_id,
        "fingerprint_sha256": evidence.fingerprint_sha256,
        "sandbox_tests_ok": evidence.sandbox_tests_ok,
        "evidence_strength": evidence.evidence_strength,
        "verdict": verdict,
        "files_touched": list(evidence.files_touched),
        "provider": evidence.provider,
        "context_sha256": meta.get("context_sha256"),
        "source_commit": meta.get("source_commit"),
        "bundle_id": meta.get("bundle_id"),
        "main_workspace_modified": False,
    }
    limitations = list(evidence.limitations or [])
    limitations.append(
        "Code-patch evidence is sandbox-scoped; does not prove main-tree improvement."
    )
    return EvidenceRecord(
        evidence_id=evidence.evidence_id
        if evidence.evidence_id.startswith("ev_")
        else evidence.evidence_id,
        project_id=evidence.project_id,
        evidence_type="failure_analysis",
        source_node_ids=[],
        source_execution_ids=[],
        source_artifact_ids=[evidence.evidence_id],
        protocol_id=None,
        metric_summary=metric_summary,
        evidence_strength=evidence.evidence_strength,
        engineering_evidence_level=evidence.evidence_strength,
        scientific_evidence_level="weak",
        limitations=limitations,
        valid=True,
        claim_level="sandbox_code_patch",
        comparison_path=evidence.artifact_path,
        created_at=datetime.now(timezone.utc),
    )


def patch_evidence_to_claim_draft(
    evidence: PatchEvidence,
    *,
    verdict: PatchVerdict | None = None,
) -> ScientificClaim:
    resolved = verdict or derive_patch_verdict(evidence)
    if resolved == "effective":
        status = "partially_supported"
        text = (
            f"Sandbox patch {evidence.patch_id} passed registered tests "
            f"({evidence.evidence_strength}); keep as merge-candidate intent only."
        )
    elif resolved == "ineffective":
        status = "unsupported"
        text = (
            f"Sandbox patch {evidence.patch_id} failed registered tests; "
            "do not treat as an improvement."
        )
    else:
        status = "unsupported"
        text = (
            f"Sandbox patch {evidence.patch_id} has inconclusive evidence "
            "(tests missing or weak); gather more sandbox validation."
        )
    return ScientificClaim(
        claim_id=new_id("claim"),
        project_id=evidence.project_id,
        claim_text=text,
        claim_type="code_patch_effectiveness",
        required_evidence_types=["failure_analysis"],
        support_status=status,  # type: ignore[arg-type]
        supporting_evidence_ids=[evidence.evidence_id],
        limitations=list(evidence.limitations or []),
        reason=f"verdict={resolved}",
    )


def patch_evidence_to_planner_fragment(
    evidence: PatchEvidence,
    *,
    verdict: PatchVerdict | None = None,
    proposal: PatchProposal | None = None,
) -> dict[str, Any]:
    """Compact record for PlanningContext.patch_feedback_records."""
    resolved = verdict or derive_patch_verdict(evidence)
    meta = dict((proposal.metadata if proposal else None) or {})
    gaps = list((proposal.evidence_gap_ids if proposal else None) or [])
    return {
        "patch_id": evidence.patch_id,
        "evidence_id": evidence.evidence_id,
        "verdict": resolved,
        "evidence_strength": evidence.evidence_strength,
        "sandbox_tests_ok": evidence.sandbox_tests_ok,
        "fingerprint_sha256": evidence.fingerprint_sha256,
        "files_touched": list(evidence.files_touched),
        "evidence_gap_ids": gaps,
        "context_sha256": meta.get("context_sha256"),
        "source_commit": meta.get("source_commit"),
        "should_retain": resolved == "effective",
        "should_revise": resolved in {"ineffective", "inconclusive"},
        "main_workspace_modified": False,
        "recorded_at": evidence.recorded_at or _now(),
    }


def patch_evidence_to_tree_note(
    evidence: PatchEvidence,
    *,
    verdict: PatchVerdict | None = None,
) -> dict[str, Any]:
    resolved = verdict or derive_patch_verdict(evidence)
    return {
        "kind": "patch_feedback",
        "patch_id": evidence.patch_id,
        "evidence_id": evidence.evidence_id,
        "verdict": resolved,
        "title": evidence.title,
        "sandbox_tests_ok": evidence.sandbox_tests_ok,
        "main_workspace_modified": False,
        "note": (
            "Code-patch sandbox outcome recorded; tree is informational only "
            "(no automatic node creation from patches in v2.2.7)."
        ),
        "recorded_at": evidence.recorded_at or _now(),
    }


def build_patch_feedback_package(
    evidence: PatchEvidence,
    proposal: PatchProposal,
    *,
    outputs_root: Path,
) -> dict[str, Any]:
    """Build + persist the cross-module feedback package for one PatchEvidence."""
    verdict = derive_patch_verdict(evidence)
    lab_record = patch_evidence_to_lab_record(evidence, proposal=proposal)
    claim = patch_evidence_to_claim_draft(evidence, verdict=verdict)
    planner = patch_evidence_to_planner_fragment(
        evidence, verdict=verdict, proposal=proposal
    )
    tree_note = patch_evidence_to_tree_note(evidence, verdict=verdict)

    package = {
        "schema_version": "v2.2.7",
        "project_id": evidence.project_id,
        "patch_id": evidence.patch_id,
        "patch_evidence_id": evidence.evidence_id,
        "verdict": verdict,
        "lab_evidence": lab_record.model_dump(mode="json"),
        "claim_draft": claim.model_dump(mode="json"),
        "planner_fragment": planner,
        "tree_note": tree_note,
        "created_at": _now(),
    }

    out_dir = Path(outputs_root) / evidence.project_id / "patch_feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{evidence.evidence_id}_feedback.json"
    write_json(path, package)
    package["artifact_path"] = str(path)

    # Append planner fragment to a rolling index for context builders.
    index_path = out_dir / "planner_feedback_index.json"
    index: list[dict[str, Any]] = []
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                index = raw
        except (OSError, json.JSONDecodeError):
            index = []
    index = [item for item in index if item.get("patch_id") != evidence.patch_id]
    index.append(planner)
    write_json(index_path, index)
    package["planner_index_path"] = str(index_path)
    return package


def load_planner_patch_feedback(
    outputs_root: Path, project_id: str, *, limit: int = 20
) -> list[dict[str, Any]]:
    path = Path(outputs_root) / project_id / "patch_feedback" / "planner_feedback_index.json"
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return list(raw)[-max(1, int(limit)) :]
