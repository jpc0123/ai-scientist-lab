"""Offline service for real research loop sessions (v2.1.1+).

v2.1.1: create / show / check sessions without network.
v2.1.2: record round feedback and build PlanningContext with feedback gate.
"""

from __future__ import annotations

from typing import Any

from scientist_lab.agents.context_builder import build_planning_context
from scientist_lab.agents.models import RoundFeedbackSummary
from scientist_lab.domain.models import new_id
from scientist_lab.research_loop.errors import (
    RealLoopNotFoundError,
    RealLoopValidationError,
)
from scientist_lab.research_loop.feedback import (
    build_round_feedback_summary,
    normalize_comparison_for_feedback,
    require_round_feedback_for_planning,
)
from scientist_lab.research_loop.models import RealResearchLoopSession, ResearchLoopRound
from scientist_lab.research_loop.repository import RealResearchLoopRepository
from scientist_lab.research_loop.state_machine import (
    NEXT_ACTIONS,
    require_status,
    transition,
)


class RealResearchLoopService:
    def __init__(self, experiment_service: Any) -> None:
        self.experiments = experiment_service
        self.repo = RealResearchLoopRepository(experiment_service.session_factory)

    def create(
        self,
        project_id: str,
        *,
        profile_id: str,
        protocol_id: str,
        rounds: int = 2,
        baseline_node_ids: list[str] | None = None,
        tree_id: str | None = None,
        fallback_allowed: bool = False,
    ) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        profile_id = (profile_id or "").strip()
        protocol_id = (protocol_id or "").strip()
        if not project_id:
            raise RealLoopValidationError("project_id is required")
        if not profile_id:
            raise RealLoopValidationError("profile_id is required")
        if not protocol_id:
            raise RealLoopValidationError("protocol_id is required")
        if int(rounds) < 2:
            raise RealLoopValidationError("required_rounds must be >= 2 for v2.1")

        project = self.experiments.repo.get_project(project_id)
        if project is None:
            raise RealLoopValidationError(f"project not found: {project_id}")

        profile = self.experiments.llm_evals.get_profile(profile_id)
        if profile is None:
            raise RealLoopValidationError(f"llm profile not found: {profile_id}")
        if not profile.enabled:
            raise RealLoopValidationError(f"llm profile disabled: {profile_id}")

        protocol = self.experiments.protocols.get(protocol_id)
        if protocol is None:
            raise RealLoopValidationError(f"protocol not found: {protocol_id}")

        baselines = list(baseline_node_ids or [])
        for node_id in baselines:
            node = self.experiments.repo.get_node(node_id)
            if node is None:
                raise RealLoopValidationError(f"baseline node not found: {node_id}")
            if node.project_id != project_id:
                raise RealLoopValidationError(
                    f"baseline node {node_id} does not belong to project {project_id}"
                )

        if fallback_allowed:
            raise RealLoopValidationError(
                "fallback_allowed must be false for real research loops (v2.1)"
            )

        session = RealResearchLoopSession.create(
            project_id=project_id,
            provider_profile_id=profile_id,
            protocol_id=protocol_id,
            required_rounds=int(rounds),
            baseline_node_ids=baselines,
            tree_id=tree_id,
            fallback_allowed=False,
        )
        self.repo.upsert_session(session)

        rounds_created: list[ResearchLoopRound] = []
        for n in range(1, session.required_rounds + 1):
            round_obj = ResearchLoopRound.create(
                session_id=session.session_id, round_number=n
            )
            self.repo.upsert_round(round_obj)
            rounds_created.append(round_obj)

        return self._session_view(session, rounds=rounds_created)

    def show(self, session_id: str) -> dict[str, Any]:
        session = self._require(session_id)
        rounds = self.repo.list_rounds(session.session_id)
        return self._session_view(session, rounds=rounds)

    def check(self, session_id: str) -> dict[str, Any]:
        """Offline integrity check — no network."""
        session = self._require(session_id)
        rounds = self.repo.list_rounds(session.session_id)
        issues: list[str] = []
        checks: list[dict[str, Any]] = []

        def _add(name: str, ok: bool, detail: str) -> None:
            checks.append({"name": name, "ok": ok, "detail": detail})
            if not ok:
                issues.append(f"{name}: {detail}")

        project = self.experiments.repo.get_project(session.project_id)
        _add(
            "project_bound",
            project is not None,
            session.project_id if project else "missing",
        )

        profile = self.experiments.llm_evals.get_profile(session.provider_profile_id)
        _add(
            "profile_bound",
            profile is not None and bool(profile.enabled),
            session.provider_profile_id
            if profile
            else f"missing:{session.provider_profile_id}",
        )

        protocol = self.experiments.protocols.get(session.protocol_id)
        _add(
            "protocol_bound",
            protocol is not None,
            session.protocol_id if protocol else "missing",
        )

        _add(
            "fallback_disabled",
            (not session.fallback_allowed) and (not session.fallback_used),
            f"allowed={session.fallback_allowed} used={session.fallback_used}",
        )
        _add(
            "real_only",
            bool(session.real_only),
            f"real_only={session.real_only}",
        )
        _add(
            "required_rounds",
            session.required_rounds >= 2,
            f"required_rounds={session.required_rounds}",
        )
        _add(
            "round_shells",
            len(rounds) >= session.required_rounds,
            f"rounds={len(rounds)} expected>={session.required_rounds}",
        )
        _add(
            "persisted",
            self.repo.get_session(session.session_id) is not None,
            session.session_id,
        )
        _add("no_network_required", True, "check is offline")

        for node_id in session.baseline_node_ids:
            node = self.experiments.repo.get_node(node_id)
            _add(
                f"baseline:{node_id}",
                node is not None and node.project_id == session.project_id,
                "ok" if node else "missing",
            )

        round1 = next((r for r in rounds if r.round_number == 1), None)
        if session.status in {
            "round_1_feedback_ready",
            "round_2_planning",
            "round_2_reviewing",
            "completed",
        }:
            has_fb = bool(round1 and round1.feedback_summary_json)
            _add(
                "round_1_feedback_summary",
                has_fb,
                "present" if has_fb else "missing",
            )

        overall = "ok" if not issues else "failed"
        return {
            "session_id": session.session_id,
            "status": session.status,
            "overall": overall,
            "network_used": False,
            "checks": checks,
            "issues": issues,
            "next_action": NEXT_ACTIONS.get(session.status, ""),
            "session": session.to_view(),
        }

    def mark_baseline_ready(self, session_id: str) -> dict[str, Any]:
        """Advance created → baseline_ready (still offline)."""
        session = self._require(session_id)
        require_status(session, "created")
        if not session.baseline_node_ids:
            raise RealLoopValidationError(
                "baseline_node_ids required before baseline_ready"
            )
        transition(session, "baseline_ready")
        self.repo.upsert_session(session)
        return self.show(session.session_id)

    def record_feedback(
        self,
        session_id: str,
        *,
        source_round: int = 1,
        parent_node_id: str,
        executed_node_id: str,
        comparison: dict[str, Any] | None = None,
        evidence_ids: list[str] | None = None,
        claims_changed: list[str] | None = None,
        comparison_ids: list[str] | None = None,
        success_criteria_met: list[str] | None = None,
        failure_criteria_met: list[str] | None = None,
        unresolved_gaps: list[str] | None = None,
        limitations: list[str] | None = None,
        previous_hypothesis: str | None = None,
        executed_parameters: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
        advance_status: bool = True,
    ) -> dict[str, Any]:
        """Persist RoundFeedbackSummary for a completed round (offline)."""
        session = self._require(session_id)
        source_round = int(source_round)
        if source_round < 1:
            raise RealLoopValidationError("source_round must be >= 1")

        parent = self.experiments.repo.get_node(parent_node_id)
        executed = self.experiments.repo.get_node(executed_node_id)
        if parent is None:
            raise RealLoopValidationError(f"parent node not found: {parent_node_id}")
        if executed is None:
            raise RealLoopValidationError(
                f"executed node not found: {executed_node_id}"
            )
        if (
            parent.project_id != session.project_id
            or executed.project_id != session.project_id
        ):
            raise RealLoopValidationError(
                "feedback nodes must belong to the session project"
            )

        summary = build_round_feedback_summary(
            source_round=source_round,
            parent_node_id=parent_node_id,
            executed_node_id=executed_node_id,
            comparison=comparison,
            evidence_ids=evidence_ids,
            claims_changed=claims_changed,
            comparison_ids=comparison_ids,
            success_criteria_met=success_criteria_met,
            failure_criteria_met=failure_criteria_met,
            unresolved_gaps=unresolved_gaps,
            limitations=limitations,
            previous_hypothesis=previous_hypothesis,
            executed_parameters=executed_parameters
            or dict((executed.contract_json or {}).get("parameters") or {}),
            failure=failure,
        )

        round_obj = self._require_round(session.session_id, source_round)
        round_obj.execution_node_id = executed_node_id
        round_obj.evidence_ids = list(summary.evidence_added)
        round_obj.claim_ids = list(summary.claims_changed)
        round_obj.feedback_summary_json = summary.model_dump(mode="json")
        round_obj.status = "feedback_ready"
        self.repo.upsert_round(round_obj)

        if executed_node_id not in session.execution_node_ids:
            session.execution_node_ids = [*session.execution_node_ids, executed_node_id]

        if advance_status and source_round == 1 and session.status == "round_1_executing":
            transition(session, "round_1_feedback_ready")
        self.repo.upsert_session(session)

        return {
            "session_id": session.session_id,
            "source_round": source_round,
            "status": session.status,
            "feedback_summary": summary.model_dump(mode="json"),
            "round": round_obj.model_dump(mode="json"),
        }

    def record_execution_feedback(
        self,
        session_id: str,
        *,
        source_round: int = 1,
        parent_node_id: str | None = None,
        executed_node_id: str | None = None,
        advance_status: bool = True,
        include_round2_context: bool = True,
    ) -> dict[str, Any]:
        """Build Evidence/Claim from executed Digits node and persist RoundFeedbackSummary.

        v2.1.5: bridges round_1_executing → round_1_feedback_ready with real evidence ids.
        """
        session = self._require(session_id)
        source_round = int(source_round)
        if source_round != 1:
            raise RealLoopValidationError(
                f"v2.1.5 record-execution-feedback supports round 1 only (got {source_round})"
            )
        require_status(session, "round_1_executing")

        round_obj = self._require_round(session.session_id, source_round)
        parent_id = str(
            parent_node_id
            or (session.baseline_node_ids[0] if session.baseline_node_ids else "")
            or ""
        ).strip()
        executed_id = str(
            executed_node_id or round_obj.execution_node_id or ""
        ).strip()
        if not parent_id:
            raise RealLoopValidationError(
                "parent_node_id required (or session.baseline_node_ids)"
            )
        if not executed_id:
            raise RealLoopValidationError(
                "executed_node_id required (run real-loop-execute first)"
            )

        parent = self.experiments.repo.get_node(parent_id)
        executed = self.experiments.repo.get_node(executed_id)
        if parent is None:
            raise RealLoopValidationError(f"parent node not found: {parent_id}")
        if executed is None:
            raise RealLoopValidationError(f"executed node not found: {executed_id}")

        # Prefer iteration comparison artifact when present; else rebuild.
        comparison: dict[str, Any] = {}
        comparison_ids: list[str] = []
        if round_obj.iteration_id:
            try:
                from scientist_lab.iteration.service import IterationService
                from pathlib import Path
                import json as _json

                iter_status = IterationService(self.experiments).get_status(
                    round_obj.iteration_id
                )
                for path in list(iter_status.get("comparison_paths") or []):
                    p = Path(str(path))
                    # comparison_paths may be dir names or full paths
                    candidates = [p]
                    if not p.is_file():
                        candidates.extend(
                            [
                                p / "group_comparison.json",
                                self.experiments.settings.outputs_dir
                                / session.project_id
                                / "comparisons"
                                / p.name
                                / "group_comparison.json",
                            ]
                        )
                    for cand_path in candidates:
                        if cand_path.is_file():
                            comparison = _json.loads(
                                cand_path.read_text(encoding="utf-8")
                            )
                            comparison_ids.append(str(cand_path))
                            break
                    if comparison:
                        break
            except Exception:  # noqa: BLE001
                comparison = {}

        if not comparison:
            comparison = self.experiments.compare_node_groups(parent_id, executed_id)

        comparison = normalize_comparison_for_feedback(comparison)

        evidence_payload = self.experiments.build_evidence(parent_id, executed_id)
        evidence_ids = [
            str(x)
            for x in (evidence_payload.get("evidence_ids") or [])
            if str(x).strip()
        ]
        if evidence_payload.get("comparison_path"):
            comparison_ids.append(str(evidence_payload["comparison_path"]))

        claim_matrix = self.experiments.build_claim_matrix(
            session.project_id, protocol_id=session.protocol_id
        )
        claims = list(claim_matrix.get("claims") or [])
        claims_changed = [
            str(c.get("claim_id"))
            for c in claims
            if isinstance(c, dict) and c.get("claim_id")
        ]

        recorded = self.record_feedback(
            session.session_id,
            source_round=source_round,
            parent_node_id=parent_id,
            executed_node_id=executed_id,
            comparison=comparison,
            evidence_ids=evidence_ids,
            claims_changed=claims_changed,
            comparison_ids=comparison_ids,
            previous_hypothesis=executed.hypothesis or parent.hypothesis,
            executed_parameters=dict((executed.contract_json or {}).get("parameters") or {}),
            advance_status=advance_status,
        )

        round2_context = None
        if include_round2_context and recorded.get("status") == "round_1_feedback_ready":
            round2_context = self.build_planning_context_for_round(
                session.session_id,
                round_number=2,
                enforce_feedback_gate=True,
            )

        return {
            "session_id": session.session_id,
            "source_round": source_round,
            "status": recorded.get("status"),
            "ok": True,
            "parent_node_id": parent_id,
            "executed_node_id": executed_id,
            "evidence_ids": evidence_ids,
            "claims_changed": claims_changed,
            "comparison_ids": comparison_ids,
            "feedback_summary": recorded.get("feedback_summary"),
            "evidence": evidence_payload,
            "claim_matrix": {
                "claim_count": len(claims_changed),
                "project_id": session.project_id,
            },
            "round2_context": round2_context,
            "next_action": NEXT_ACTIONS.get(str(recorded.get("status") or ""), ""),
        }

    def prepare_next_round(self, session_id: str) -> dict[str, Any]:
        """Advance round_1_feedback_ready → round_2_planning and refresh Round-2 context."""
        session = self._require(session_id)
        require_status(session, "round_1_feedback_ready")
        transition(session, "round_2_planning")
        session.current_round = 2
        self.repo.upsert_session(session)
        context = self.build_planning_context_for_round(
            session.session_id, round_number=2, enforce_feedback_gate=True
        )
        return {
            "session_id": session.session_id,
            "status": session.status,
            "ok": True,
            "current_round": session.current_round,
            "context": context,
            "next_action": NEXT_ACTIONS.get(session.status, ""),
        }

    def verify_feedback_use(
        self,
        session_id: str,
        *,
        round_number: int = 2,
        plan_id: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Prove Round N planning used Round N-1 feedback (deterministic; v2.1.6)."""
        from scientist_lab.research_loop.feedback_use_verifier import (
            verify_feedback_use as _verify,
        )
        from scientist_lab.research_loop.models import FeedbackUsageRecord

        session = self._require(session_id)
        round_number = int(round_number)
        if round_number < 2:
            raise RealLoopValidationError(
                "verify-feedback requires round_number >= 2"
            )

        source = self._require_round(session.session_id, round_number - 1)
        target = self._require_round(session.session_id, round_number)
        if not source.feedback_summary_json:
            raise RealLoopValidationError(
                f"round {round_number - 1} feedback_summary missing"
            )

        context = dict(target.planning_context_json or {})
        if not context:
            # Rebuild so verification can still run after next-round.
            built = self.build_planning_context_for_round(
                session.session_id,
                round_number=round_number,
                enforce_feedback_gate=True,
            )
            context = dict(built.get("context") or {})
            target = self._require_round(session.session_id, round_number)

        resolved_plan_id = str(plan_id or target.plan_id or "").strip() or None
        candidates: list[dict[str, Any]] = []
        if resolved_plan_id:
            try:
                plan = self.experiments.show_plan(resolved_plan_id)
                candidates = [
                    dict(c)
                    for c in (plan.get("candidates") or [])
                    if isinstance(c, dict)
                ]
            except Exception as exc:  # noqa: BLE001
                raise RealLoopValidationError(
                    f"cannot load plan {resolved_plan_id}: {exc}"
                ) from exc
        else:
            raise RealLoopValidationError(
                "plan_id missing on target round; run real-loop-plan for round 2 first"
            )

        prev_candidates: list[dict[str, Any]] = []
        if source.plan_id:
            try:
                prev_plan = self.experiments.show_plan(source.plan_id)
                prev_candidates = [
                    dict(c)
                    for c in (prev_plan.get("candidates") or [])
                    if isinstance(c, dict)
                ]
            except Exception:  # noqa: BLE001
                prev_candidates = []

        feedback = dict(source.feedback_summary_json or {})
        verification = _verify(
            feedback_summary=feedback,
            planning_context=context,
            candidates=candidates,
            previous_executed_parameters=dict(feedback.get("executed_parameters") or {}),
            previous_candidate_parameters=prev_candidates,
            expected_context_sha256=str(context.get("context_sha256") or "") or None,
        )

        record = FeedbackUsageRecord.create(
            session_id=session.session_id,
            source_round_id=source.round_id,
            target_round_id=target.round_id,
            source_round_number=source.round_number,
            target_round_number=target.round_number,
            context_sha256=str(context.get("context_sha256") or ""),
            plan_id=resolved_plan_id,
            verification=verification,
        )
        if persist:
            self.repo.upsert_feedback_usage(record)

        return {
            "session_id": session.session_id,
            "ok": bool(verification.pass_status),
            "pass_status": bool(verification.pass_status),
            "round_number": round_number,
            "plan_id": resolved_plan_id,
            "checks": {
                "context_contains_round_1_metrics": verification.context_contains_round_1_metrics,
                "context_contains_round_1_evidence": verification.context_contains_round_1_evidence,
                "context_contains_round_1_decision": verification.context_contains_round_1_decision,
                "output_references_new_evidence": verification.output_references_new_evidence,
                "output_changes_experiment_plan": verification.output_changes_experiment_plan,
                "output_avoids_duplicate_candidate": verification.output_avoids_duplicate_candidate,
            },
            "issues": list(verification.issues),
            "details": dict(verification.details),
            "feedback_usage": record.model_dump(mode="json"),
            "context_sha256": record.context_sha256,
            "next_action": (
                "Feedback use verified; continue round-2 review/approval."
                if verification.pass_status
                else "Fix Round-2 plan: must use Round-1 feedback and change parameters."
            ),
        }

    def export_replay(
        self,
        session_id: str,
        *,
        output_dir: str | None = None,
        allow_incomplete: bool = False,
    ) -> dict[str, Any]:
        """Export a redacted Replay Bundle for offline CI (v2.1.7)."""
        from pathlib import Path

        from scientist_lab.research_loop.replay_bundle import build_replay_bundle

        session = self._require(session_id)
        rounds = self.repo.list_rounds(session.session_id)
        if not allow_incomplete:
            ready_statuses = {
                "round_1_feedback_ready",
                "round_2_planning",
                "round_2_reviewing",
                "completed",
            }
            has_r1_feedback = any(
                r.round_number == 1 and bool(r.feedback_summary_json) for r in rounds
            )
            if session.status not in ready_statuses and not has_r1_feedback:
                raise RealLoopValidationError(
                    "export requires at least round-1 feedback "
                    f"(status={session.status}); pass allow_incomplete=true to force"
                )

        out = Path(
            output_dir
            or (
                Path(self.experiments.settings.outputs_dir)
                / session.project_id
                / "real_loop"
                / session.session_id
            )
        )

        plans: dict[str, dict[str, Any]] = {}
        for rnd in rounds:
            if not rnd.plan_id:
                continue
            try:
                plans[str(rnd.plan_id)] = self.experiments.show_plan(str(rnd.plan_id))
            except Exception:  # noqa: BLE001
                plans[str(rnd.plan_id)] = {
                    "plan_id": rnd.plan_id,
                    "candidates": [],
                    "status": "unavailable",
                }

        evidence_records: list[dict[str, Any]] = []
        claim_matrix: dict[str, Any] = {}
        try:
            evidence_records = self.experiments.list_evidence(
                project_id=session.project_id
            )
        except Exception:  # noqa: BLE001
            evidence_records = []
        try:
            claim_matrix = self.experiments.show_claim_matrix(session.project_id)
        except Exception:  # noqa: BLE001
            try:
                claim_matrix = self.experiments.build_claim_matrix(
                    session.project_id, protocol_id=session.protocol_id
                )
            except Exception:  # noqa: BLE001
                claim_matrix = {}

        llm_calls: list[dict[str, Any]] = []
        try:
            from scientist_lab.llm.repository import LLMCallRepository

            audit_root = (
                Path(self.experiments.settings.outputs_dir) / session.project_id / "llm"
            )
            if audit_root.exists():
                llm_calls = [
                    c.model_dump(mode="json")
                    for c in LLMCallRepository(audit_root).list_calls(limit=200)
                ]
        except Exception:  # noqa: BLE001
            llm_calls = []

        usage = [
            u.model_dump(mode="json")
            for u in self.repo.list_feedback_usage(session.session_id, limit=20)
        ]

        result = build_replay_bundle(
            session=session.to_view(),
            rounds=[r.model_dump(mode="json") for r in rounds],
            output_dir=out,
            project_id=session.project_id,
            feedback_usage=usage,
            llm_calls=llm_calls,
            plans=plans,
            evidence_records=evidence_records,
            claim_matrix=claim_matrix,
            extra_report={
                "provider_profile_id": session.provider_profile_id,
                "protocol_id": session.protocol_id,
                "execution_node_ids": list(session.execution_node_ids),
                "baseline_node_ids": list(session.baseline_node_ids),
            },
        )
        result["next_action"] = (
            "Replay bundle ready for offline accept_v21 / ReplayProvider."
        )
        return result

    def build_planning_context_for_round(
        self,
        session_id: str,
        *,
        round_number: int,
        enforce_feedback_gate: bool = True,
    ) -> dict[str, Any]:
        """Build PlanningContext for the given round (offline; no LLM call)."""
        session = self._require(session_id)
        round_number = int(round_number)
        if round_number < 1:
            raise RealLoopValidationError("round_number must be >= 1")

        project = self.experiments.repo.get_project(session.project_id)
        if project is None:
            raise RealLoopValidationError(f"project not found: {session.project_id}")
        protocol = self.experiments.protocols.get(session.protocol_id)
        if protocol is None:
            raise RealLoopValidationError(f"protocol not found: {session.protocol_id}")

        nodes = self.experiments.repo.list_nodes(project_id=session.project_id)
        protocol_dict = (
            protocol.model_dump(mode="json")
            if hasattr(protocol, "model_dump")
            else dict(protocol)
        )

        feedback: RoundFeedbackSummary | None = None
        previous_hypothesis = None
        previous_params: dict[str, Any] = {}
        recent_exec: dict[str, Any] = {}
        evidence_records: list[dict[str, Any]] = []
        claim_support_matrix: dict[str, Any] = {}
        comparisons: list[dict[str, Any]] = []
        if round_number >= 2:
            prev = self._require_round(session.session_id, round_number - 1)
            if not prev.feedback_summary_json:
                raise RealLoopValidationError(
                    f"round {round_number - 1} feedback_summary missing; "
                    "run real-loop-record-execution-feedback first"
                )
            feedback = RoundFeedbackSummary.model_validate(prev.feedback_summary_json)
            previous_hypothesis = feedback.previous_hypothesis
            previous_params = dict(feedback.executed_parameters or {})
            recent_exec = {
                "executed_node_id": feedback.executed_node_id,
                "parent_node_id": feedback.parent_node_id,
                "outcome_label": feedback.outcome_label,
                "metric_deltas": feedback.metric_deltas,
                "evidence_added": feedback.evidence_added,
                "claims_changed": feedback.claims_changed,
            }

            # Inject project Evidence / Claim matrix into Round-2 context (v2.1.5).
            try:
                all_evidence = self.experiments.list_evidence(
                    project_id=session.project_id
                )
            except Exception:  # noqa: BLE001
                all_evidence = []
            wanted = set(feedback.evidence_added or [])
            if wanted:
                evidence_records = [
                    e
                    for e in all_evidence
                    if isinstance(e, dict) and e.get("evidence_id") in wanted
                ]
                if not evidence_records:
                    evidence_records = list(all_evidence)
            else:
                evidence_records = list(all_evidence)

            try:
                claim_support_matrix = self.experiments.show_claim_matrix(
                    session.project_id
                )
            except Exception:  # noqa: BLE001
                try:
                    claim_support_matrix = self.experiments.build_claim_matrix(
                        session.project_id, protocol_id=session.protocol_id
                    )
                except Exception:  # noqa: BLE001
                    claim_support_matrix = {}

            for cid in feedback.comparison_ids or []:
                from pathlib import Path
                import json as _json

                path = Path(str(cid))
                if path.is_file():
                    try:
                        comparisons.append(
                            _json.loads(path.read_text(encoding="utf-8"))
                        )
                    except Exception:  # noqa: BLE001
                        pass
            if not comparisons and feedback.parent_node_id and feedback.executed_node_id:
                try:
                    comparisons.append(
                        normalize_comparison_for_feedback(
                            self.experiments.compare_node_groups(
                                feedback.parent_node_id, feedback.executed_node_id
                            )
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass

        context = build_planning_context(
            project_id=session.project_id,
            research_goal=getattr(project, "research_goal", None)
            or getattr(project, "title", "")
            or "Digits real research loop",
            protocol=protocol_dict,
            nodes=nodes,
            loop_session_id=session.session_id,
            loop_round_number=round_number,
            round_feedback_summary=feedback,
            recent_execution_summary=recent_exec,
            previous_planner_hypothesis=previous_hypothesis,
            previous_parameter_changes=previous_params,
            evidence_records=evidence_records,
            claim_support_matrix=claim_support_matrix,
            comparisons=comparisons,
        )

        if enforce_feedback_gate:
            require_round_feedback_for_planning(context, round_number=round_number)

        round_obj = self._require_round(session.session_id, round_number)
        context_id = round_obj.planning_context_id or new_id("rlctx")
        round_obj.planning_context_id = context_id
        round_obj.planning_context_json = context.model_dump(mode="json")
        if round_number >= 2:
            round_obj.status = "planning"
        self.repo.upsert_round(round_obj)

        return {
            "session_id": session.session_id,
            "round_number": round_number,
            "planning_context_id": context_id,
            "context_sha256": context.context_sha256,
            "has_round_feedback": context.round_feedback_summary is not None,
            "evidence_count": len(evidence_records),
            "claim_count": len(
                (claim_support_matrix or {}).get("claims") or []
            ),
            "comparison_count": len(comparisons),
            "context": context.model_dump(mode="json"),
        }

    def list_sessions(
        self, *, project_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return [
            self._session_view(s, rounds=self.repo.list_rounds(s.session_id))
            for s in self.repo.list_sessions(project_id=project_id, limit=limit)
        ]

    def check_profile(
        self,
        profile_id: str | None = None,
        *,
        session_id: str | None = None,
        suite_version: str | None = "eval_suite_v1",
        require_quality_gate: bool = True,
    ) -> dict[str, Any]:
        """Offline Model Profile qualification for a real loop (no network)."""
        from scientist_lab.research_loop.provider_gate import check_profile_for_real_loop

        resolved_profile_id = (profile_id or "").strip() or None
        if session_id:
            session = self._require(session_id)
            resolved_profile_id = resolved_profile_id or session.provider_profile_id
        if not resolved_profile_id:
            raise RealLoopValidationError("profile_id or session_id is required")

        profile = self.experiments.llm_evals.get_profile(resolved_profile_id)
        evaluation = None
        if profile is not None:
            evaluation = self.experiments.llm_evals.latest_evaluation_for_profile(
                resolved_profile_id, suite_version=suite_version
            )
        budget_configured = True
        try:
            if session_id:
                session = self._require(session_id)
                rem = self.experiments.budget.remaining(session.project_id)
                budget_configured = bool(rem)
            elif profile is not None:
                # Profile-only check: treat experiment budget as optional.
                budget_configured = True
        except Exception:  # noqa: BLE001
            budget_configured = False

        report = check_profile_for_real_loop(
            profile=profile,
            evaluation=evaluation,
            suite_version=suite_version,
            require_quality_gate=require_quality_gate,
            budget_configured=budget_configured,
            network_enabled=False,
        )
        report["session_id"] = session_id
        return report

    def plan_round(
        self,
        session_id: str,
        *,
        round_number: int | None = None,
        allow_network: bool = False,
        transport: Any = None,
        openai_config: Any = None,
        suite_version: str | None = "eval_suite_v1",
        provider: str = "openai-compatible",
    ) -> dict[str, Any]:
        """Run real-only Planner for a loop round.

        Never falls back to mock/fake/replay. Failures mark the session and
        return without substitute candidates.
        """
        from scientist_lab.research_loop.provider_gate import (
            assert_no_fallback,
            assert_real_only_provider,
            provider_audit,
            require_profile_qualified_for_real_loop,
        )

        session = self._require(session_id)
        if session.fallback_allowed or not session.real_only:
            raise RealLoopValidationError(
                "session must be real_only with fallback_allowed=false"
            )

        round_number = int(
            round_number
            if round_number is not None
            else (1 if session.current_round < 1 else session.current_round)
        )
        if round_number < 1:
            raise RealLoopValidationError("round_number must be >= 1")

        try:
            requested = assert_real_only_provider(provider, real_only=True)
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="provider_failed",
                error=exc,
                requested_provider=provider,
                actual_provider=None,
            )

        # Advance into planning state when legal.
        try:
            if round_number == 1 and session.status == "baseline_ready":
                transition(session, "round_1_planning")
            elif round_number == 2 and session.status == "round_1_feedback_ready":
                transition(session, "round_2_planning")
            self.repo.upsert_session(session)
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="planner_failed",
                error=exc,
                requested_provider=requested,
                actual_provider=None,
            )

        profile = self.experiments.llm_evals.get_profile(session.provider_profile_id)
        evaluation = self.experiments.llm_evals.latest_evaluation_for_profile(
            session.provider_profile_id, suite_version=suite_version
        )
        try:
            rem = self.experiments.budget.remaining(session.project_id)
            quality_report = require_profile_qualified_for_real_loop(
                profile=profile,
                evaluation=evaluation,
                suite_version=suite_version,
                budget_configured=bool(rem),
            )
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="quality_gate_blocked",
                error=exc,
                requested_provider=requested,
                actual_provider=None,
            )

        # Build context with feedback gate (round >= 2).
        try:
            ctx_bundle = self.build_planning_context_for_round(
                session.session_id,
                round_number=round_number,
                enforce_feedback_gate=True,
            )
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="feedback_incomplete",
                error=exc,
                requested_provider=requested,
                actual_provider=None,
            )

        # Real planner call — never silent fallback.
        plan_result = self.experiments.plan_next(
            session.project_id,
            protocol_id=session.protocol_id,
            provider=requested,
            allow_network=allow_network,
            transport=transport,
            openai_config=openai_config,
            model_profile=session.provider_profile_id,
            require_quality_gate=True,
            allow_unqualified_profile=False,
            suite_version=suite_version,
        )

        requested_out = str(plan_result.get("requested_provider") or requested)
        actual_out = plan_result.get("actual_provider") or plan_result.get(
            "model_provider"
        )
        fallback_used = bool(plan_result.get("fallback_used", False))
        audit = provider_audit(
            requested_provider=requested_out,
            actual_provider=actual_out,
            fallback_allowed=False,
            fallback_used=fallback_used,
            extra={"quality_gate": plan_result.get("quality_gate") or quality_report},
        )

        status = str(plan_result.get("status") or "")
        if status in {"profile_not_qualified", "real_provider_failed"}:
            fail_status = (
                "quality_gate_blocked"
                if status == "profile_not_qualified"
                else "provider_failed"
            )
            return self._fail_round(
                session,
                round_number=round_number,
                status=fail_status,
                error=plan_result.get("error") or status,
                requested_provider=requested_out,
                actual_provider=actual_out,
                fallback_used=fallback_used,
                plan_result=plan_result,
                audit=audit,
            )
        if status == "planner_failed":
            return self._fail_round(
                session,
                round_number=round_number,
                status="planner_failed",
                error=plan_result.get("error") or "planner_failed: no valid candidates",
                requested_provider=requested_out,
                actual_provider=actual_out,
                fallback_used=fallback_used,
                plan_result=plan_result,
                audit=audit,
            )

        try:
            assert_no_fallback(
                requested_provider=requested_out,
                actual_provider=str(actual_out) if actual_out else None,
                fallback_allowed=False,
                fallback_used=fallback_used,
            )
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="provider_failed",
                error=exc,
                requested_provider=requested_out,
                actual_provider=actual_out,
                fallback_used=True,
                plan_result=plan_result,
                audit=audit,
            )

        # Persist successful plan onto the round — no substitute candidates on failure path.
        round_obj = self._require_round(session.session_id, round_number)
        candidates = list(plan_result.get("candidates") or [])
        candidate_ids = [
            str(c.get("candidate_id"))
            for c in candidates
            if isinstance(c, dict) and c.get("candidate_id")
        ]
        round_obj.candidate_ids = candidate_ids
        round_obj.plan_id = plan_result.get("plan_id")
        round_obj.provider_audit_json = audit
        round_obj.planning_context_json = ctx_bundle.get("context") or {}
        round_obj.planning_context_id = ctx_bundle.get("planning_context_id") or ""
        round_obj.status = "reviewing"
        self.repo.upsert_round(round_obj)

        session.current_round = round_number
        session.generated_candidate_ids = [
            *session.generated_candidate_ids,
            *[c for c in candidate_ids if c not in session.generated_candidate_ids],
        ]
        session.error_type = None
        session.error_message = None
        try:
            if round_number == 1 and session.status == "round_1_planning":
                transition(session, "round_1_reviewing")
            elif round_number == 2 and session.status == "round_2_planning":
                transition(session, "round_2_reviewing")
        except Exception:
            pass
        self.repo.upsert_session(session)

        payload = {
            "session_id": session.session_id,
            "round_number": round_number,
            "status": session.status,
            "plan_id": plan_result.get("plan_id"),
            "candidates": candidates,
            "candidate_ids": candidate_ids,
            "provider_audit": audit,
            "context_sha256": ctx_bundle.get("context_sha256"),
            "fallback_used": False,
            "plan": plan_result,
        }
        if round_number >= 2 and plan_result.get("plan_id"):
            try:
                payload["feedback_use"] = self.verify_feedback_use(
                    session.session_id,
                    round_number=round_number,
                    plan_id=str(plan_result.get("plan_id")),
                    persist=True,
                )
                payload["feedback_use_ok"] = bool(
                    payload["feedback_use"].get("pass_status")
                )
            except Exception as exc:  # noqa: BLE001
                payload["feedback_use"] = {
                    "ok": False,
                    "pass_status": False,
                    "error": str(exc),
                }
                payload["feedback_use_ok"] = False
        return payload

    def review_round(
        self,
        session_id: str,
        *,
        round_number: int | None = None,
        allow_network: bool = False,
        transport: Any = None,
        openai_config: Any = None,
        provider: str = "openai-compatible",
    ) -> dict[str, Any]:
        """Run real-only Critic review for candidates of a loop round."""
        from scientist_lab.research_loop.provider_gate import (
            assert_no_fallback,
            assert_real_only_provider,
            provider_audit,
        )

        session = self._require(session_id)
        if session.fallback_allowed or not session.real_only:
            raise RealLoopValidationError(
                "session must be real_only with fallback_allowed=false"
            )
        round_number = int(
            round_number
            if round_number is not None
            else (session.current_round or 1)
        )
        try:
            requested = assert_real_only_provider(provider, real_only=True)
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="provider_failed",
                error=exc,
                requested_provider=provider,
                actual_provider=None,
            )

        round_obj = self._require_round(session.session_id, round_number)
        plan_id = round_obj.plan_id
        if not plan_id:
            return self._fail_round(
                session,
                round_number=round_number,
                status="critic_failed",
                error="plan_id missing; run real-loop-plan first",
                requested_provider=requested,
                actual_provider=None,
            )

        try:
            review = self.experiments.review_plan(
                plan_id,
                provider=requested,
                allow_network=allow_network,
                transport=transport,
                openai_config=openai_config,
            )
        except TypeError:
            # Older review_plan signature without provider kwargs.
            review = self.experiments.review_plan(plan_id)
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="critic_failed",
                error=exc,
                requested_provider=requested,
                actual_provider=None,
            )

        if isinstance(review, dict) and review.get("status") == "real_provider_failed":
            return self._fail_round(
                session,
                round_number=round_number,
                status="provider_failed",
                error=review.get("error") or "real_provider_failed",
                requested_provider=str(review.get("requested_provider") or requested),
                actual_provider=review.get("actual_provider"),
                fallback_used=bool(review.get("fallback_used", False)),
                plan_result=review,
            )

        requested_out = requested
        actual_out = None
        if isinstance(review, dict):
            actual_out = review.get("actual_provider") or review.get("model_provider")
            fallback_used = bool(review.get("fallback_used", False))
        else:
            fallback_used = False
        audit = provider_audit(
            requested_provider=requested_out,
            actual_provider=actual_out,
            fallback_allowed=False,
            fallback_used=fallback_used,
        )
        try:
            assert_no_fallback(
                requested_provider=requested_out,
                actual_provider=str(actual_out) if actual_out else requested_out,
                fallback_allowed=False,
                fallback_used=fallback_used,
            )
        except Exception as exc:
            return self._fail_round(
                session,
                round_number=round_number,
                status="provider_failed",
                error=exc,
                requested_provider=requested_out,
                actual_provider=actual_out,
                fallback_used=True,
                audit=audit,
            )

        round_obj.provider_audit_json = {
            **dict(round_obj.provider_audit_json or {}),
            "critic": audit,
        }
        round_obj.status = "waiting_approval"
        self.repo.upsert_round(round_obj)

        try:
            if round_number == 1 and session.status == "round_1_reviewing":
                transition(session, "round_1_waiting_approval")
            elif round_number == 2 and session.status == "round_2_reviewing":
                # Round 2 reviewing completes the loop for v2.1 planning proof;
                # waiting approval for round-2 execute is future work.
                transition(session, "completed")
        except Exception:
            pass
        self.repo.upsert_session(session)

        return {
            "session_id": session.session_id,
            "round_number": round_number,
            "status": session.status,
            "plan_id": plan_id,
            "provider_audit": audit,
            "fallback_used": False,
            "review": review,
        }

    def approve_round(
        self,
        session_id: str,
        *,
        candidate_id: str,
        round_number: int | None = None,
        seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Human-approve one candidate and materialize a Digits iteration (no run yet)."""
        from scientist_lab.iteration.service import IterationService
        from scientist_lab.research_loop.execution_gate import (
            assert_real_digits_contract,
            enrich_parent_contract_for_digits,
        )

        session = self._require(session_id)
        if session.fallback_allowed or not session.real_only:
            raise RealLoopValidationError(
                "session must be real_only with fallback_allowed=false"
            )
        round_number = int(
            round_number
            if round_number is not None
            else (session.current_round or 1)
        )
        if round_number == 1:
            require_status(session, "round_1_waiting_approval")
        else:
            raise RealLoopValidationError(
                f"v2.1.4 approve supports round 1 only (got round={round_number})"
            )

        round_obj = self._require_round(session.session_id, round_number)
        plan_id = round_obj.plan_id
        if not plan_id:
            raise RealLoopValidationError("plan_id missing; run real-loop-plan first")
        cand = str(candidate_id or "").strip()
        if not cand:
            raise RealLoopValidationError("candidate_id is required")
        if round_obj.candidate_ids and cand not in round_obj.candidate_ids:
            raise RealLoopValidationError(
                f"candidate {cand} not in round candidate_ids={round_obj.candidate_ids}"
            )

        # Enrich parent baseline to Digits real contract surface before generation.
        plan = self.experiments.show_plan(plan_id)
        cand_rows = {
            item.get("candidate_id"): item for item in plan.get("candidates") or []
        }
        cand_row = cand_rows.get(cand)
        if cand_row is None:
            raise RealLoopValidationError(f"candidate not found in plan: {cand}")
        parent_id = str(cand_row.get("parent_node_id") or "")
        parent = self.experiments.repo.get_node(parent_id)
        if parent is None:
            raise RealLoopValidationError(f"parent node not found: {parent_id}")

        protocol = self.experiments.protocols.get(session.protocol_id)
        protocol_dict = (
            protocol.model_dump(mode="json")
            if protocol is not None and hasattr(protocol, "model_dump")
            else (dict(protocol) if protocol else {})
        )
        project = self.experiments.repo.get_project(session.project_id)
        enriched = enrich_parent_contract_for_digits(
            parent.contract_json,
            protocol=protocol_dict,
            project_id=session.project_id,
            node_id=parent.node_id,
            title=getattr(project, "title", None) if project else None,
            research_goal=getattr(project, "research_goal", None) if project else None,
            hypothesis=parent.hypothesis,
        )
        assert_real_digits_contract(enriched)
        parent.contract_json = enriched
        parent.updated_at = parent.updated_at  # keep; touch via upsert
        from scientist_lab.domain.models import utc_now_iso

        parent.updated_at = utc_now_iso()
        self.experiments.repo.upsert_node(parent)

        self.experiments.approve_candidate(plan_id, cand)
        iteration = IterationService(self.experiments)
        started = iteration.start_from_plan(plan_id, cand, seeds=seeds)

        # Validate generated proposal is still Digits-real (never mock).
        contract = started.get("contract") or {}
        if not contract and started.get("proposal_path"):
            from pathlib import Path
            import json as _json

            contract = _json.loads(Path(started["proposal_path"]).read_text(encoding="utf-8"))
        gate = assert_real_digits_contract(contract)

        round_obj.approved_candidate_id = cand
        round_obj.contract_id = str(
            contract.get("node_id") or started.get("proposed_node_id") or ""
        ) or None
        round_obj.iteration_id = started.get("iteration_id")
        round_obj.execution_node_id = started.get("proposed_node_id") or contract.get(
            "node_id"
        )
        round_obj.status = "waiting_approval"
        self.repo.upsert_round(round_obj)
        self.repo.upsert_session(session)

        return {
            "session_id": session.session_id,
            "round_number": round_number,
            "status": session.status,
            "ok": True,
            "plan_id": plan_id,
            "candidate_id": cand,
            "iteration_id": round_obj.iteration_id,
            "proposed_node_id": round_obj.execution_node_id,
            "contract_id": round_obj.contract_id,
            "execution_gate": gate,
            "mock_execution": False,
            "next_action": NEXT_ACTIONS.get(session.status, ""),
            "iteration": started,
        }

    def reject_round(
        self,
        session_id: str,
        *,
        candidate_id: str | None = None,
        round_number: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Human-reject the round candidate without executing."""
        session = self._require(session_id)
        round_number = int(
            round_number
            if round_number is not None
            else (session.current_round or 1)
        )
        require_status(session, "round_1_waiting_approval")
        round_obj = self._require_round(session.session_id, round_number)
        plan_id = round_obj.plan_id
        cand = str(candidate_id or round_obj.approved_candidate_id or "").strip()
        if plan_id and cand:
            try:
                self.experiments.reject_candidate(plan_id, cand, reason=reason)
            except Exception:  # noqa: BLE001
                pass
        round_obj.status = "cancelled"
        self.repo.upsert_round(round_obj)
        transition(session, "candidate_rejected")
        session.error_type = "CandidateRejected"
        session.error_message = reason or f"candidate rejected: {cand or 'unknown'}"
        self.repo.upsert_session(session)
        return {
            "session_id": session.session_id,
            "round_number": round_number,
            "status": session.status,
            "ok": True,
            "candidate_id": cand or None,
            "reason": reason,
            "next_action": NEXT_ACTIONS.get(session.status, ""),
        }

    def execute_round(
        self,
        session_id: str,
        *,
        round_number: int | None = None,
        seeds: list[int] | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Run the approved Digits iteration (hard-ban mock execution)."""
        from pathlib import Path
        import json as _json

        from scientist_lab.iteration.service import IterationService
        from scientist_lab.research_loop.execution_gate import assert_real_digits_contract

        session = self._require(session_id)
        if session.fallback_allowed or not session.real_only:
            raise RealLoopValidationError(
                "session must be real_only with fallback_allowed=false"
            )
        round_number = int(
            round_number
            if round_number is not None
            else (session.current_round or 1)
        )
        if round_number != 1:
            raise RealLoopValidationError(
                f"v2.1.4 execute supports round 1 only (got round={round_number})"
            )
        require_status(session, "round_1_waiting_approval")

        round_obj = self._require_round(session.session_id, round_number)
        iteration_id = round_obj.iteration_id
        if not iteration_id:
            raise RealLoopValidationError(
                "iteration_id missing; run real-loop-approve first"
            )
        if not round_obj.approved_candidate_id:
            raise RealLoopValidationError(
                "approved_candidate_id missing; run real-loop-approve first"
            )

        iteration = IterationService(self.experiments)
        status_payload = iteration.get_status(iteration_id)
        proposal_path = status_payload.get("proposal_path")
        if not proposal_path:
            raise RealLoopValidationError(
                f"iteration {iteration_id} has no proposal_path"
            )
        contract = _json.loads(Path(proposal_path).read_text(encoding="utf-8"))
        gate = assert_real_digits_contract(contract)

        transition(session, "round_1_executing")
        round_obj.status = "executing"
        session.error_type = None
        session.error_message = None
        self.repo.upsert_round(round_obj)
        self.repo.upsert_session(session)

        try:
            run_payload = iteration.approve_and_run(
                iteration_id, seeds=seeds, wait=wait
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail_round(
                session,
                round_number=round_number,
                status="execution_failed",
                error=exc,
                requested_provider="digits-real",
                actual_provider="digits-real",
                fallback_used=False,
                plan_result={"execution_gate": gate, "mock_execution": False},
            )

        node_id = (
            run_payload.get("proposed_node_id")
            or (run_payload.get("run_result") or {}).get("node_id")
            or round_obj.execution_node_id
            or contract.get("node_id")
        )
        if node_id:
            round_obj.execution_node_id = str(node_id)
            if str(node_id) not in session.execution_node_ids:
                session.execution_node_ids = [
                    *session.execution_node_ids,
                    str(node_id),
                ]
        round_obj.status = "executing"
        self.repo.upsert_round(round_obj)
        self.repo.upsert_session(session)

        run_result = run_payload.get("run_result") or {}
        results = list(run_result.get("results") or run_payload.get("results") or [])
        failed = [
            r
            for r in results
            if isinstance(r, dict) and str(r.get("status") or "").lower() == "failed"
        ]
        if wait and failed and not any(
            isinstance(r, dict) and str(r.get("status") or "").lower() == "completed"
            for r in results
        ):
            return self._fail_round(
                session,
                round_number=round_number,
                status="execution_failed",
                error=f"all seed runs failed ({len(failed)})",
                requested_provider="digits-real",
                actual_provider="digits-real",
                fallback_used=False,
                plan_result={
                    "execution_gate": gate,
                    "run_payload": run_payload,
                    "mock_execution": False,
                },
            )

        return {
            "session_id": session.session_id,
            "round_number": round_number,
            "status": session.status,
            "ok": True,
            "iteration_id": iteration_id,
            "execution_node_id": round_obj.execution_node_id,
            "approved_candidate_id": round_obj.approved_candidate_id,
            "execution_gate": gate,
            "mock_execution": False,
            "wait": wait,
            "next_action": NEXT_ACTIONS.get(session.status, ""),
            "iteration": run_payload,
        }

    def _fail_round(
        self,
        session: RealResearchLoopSession,
        *,
        round_number: int,
        status: str,
        error: Any,
        requested_provider: str,
        actual_provider: str | None,
        fallback_used: bool = False,
        plan_result: dict[str, Any] | None = None,
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.provider_gate import provider_audit

        message = str(error)
        session.error_type = type(error).__name__ if not isinstance(error, str) else "Error"
        session.error_message = message
        try:
            if session.status not in {
                "completed",
                "cancelled",
                "provider_failed",
                "planner_failed",
                "critic_failed",
                "quality_gate_blocked",
                "feedback_incomplete",
                "execution_failed",
                "candidate_rejected",
            }:
                transition(session, status)
        except Exception:
            session.status = status  # type: ignore[assignment]
            session.touch()
        self.repo.upsert_session(session)

        audit_payload = audit or provider_audit(
            requested_provider=requested_provider,
            actual_provider=actual_provider,
            fallback_allowed=False,
            fallback_used=fallback_used,
        )
        try:
            round_obj = self._require_round(session.session_id, round_number)
            round_obj.provider_audit_json = audit_payload
            round_obj.status = "failed"
            # Explicitly do NOT invent substitute candidates.
            self.repo.upsert_round(round_obj)
        except Exception:
            pass

        return {
            "session_id": session.session_id,
            "round_number": round_number,
            "status": session.status,
            "ok": False,
            "error_type": session.error_type,
            "error": message,
            "provider_audit": audit_payload,
            "fallback_used": fallback_used,
            "candidates": [],
            "plan": plan_result,
        }

    def _require(self, session_id: str) -> RealResearchLoopSession:
        try:
            return self.repo.require_session(session_id)
        except KeyError as exc:
            raise RealLoopNotFoundError(str(exc)) from exc

    def _require_round(self, session_id: str, round_number: int) -> ResearchLoopRound:
        rounds = self.repo.list_rounds(session_id)
        for item in rounds:
            if int(item.round_number) == int(round_number):
                return item
        raise RealLoopNotFoundError(
            f"real-loop round {round_number} not found for session {session_id}"
        )

    def _session_view(
        self,
        session: RealResearchLoopSession,
        *,
        rounds: list[ResearchLoopRound] | None = None,
    ) -> dict[str, Any]:
        from scientist_lab.research_loop.display_mode import enrich_session_view

        data = session.to_view()
        data["next_action"] = NEXT_ACTIONS.get(session.status, "")
        data["rounds"] = [r.model_dump(mode="json") for r in (rounds or [])]
        return enrich_session_view(data)
