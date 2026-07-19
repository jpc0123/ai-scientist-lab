from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.domain.models import new_id, utc_now_iso
from scientist_lab.iteration.models import ApprovalRecord, IterationSession
from scientist_lab.iteration.repository import IterationRepository
from scientist_lab.iteration.workflow import require_status, transition
from scientist_lab.services.experiment_service import ExperimentService, load_contract
from scientist_lab.storage.artifact_store import sha256_file


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
MINIMUM_SUCCESSFUL_SEEDS = 3

NEXT_ACTIONS: dict[str, str] = {
    "created": "Iteration created; waiting for feedback generation.",
    "feedback_ready": "Feedback ready; waiting for proposal.",
    "proposal_ready": "Proposal ready; moving to approval.",
    "waiting_approval": "Review the proposal and run iterate-approve or iterate-reject.",
    "approved": "Approved; starting seed runs.",
    "running": "Seed runs in progress.",
    "comparing": "Comparing new node against source nodes.",
    "waiting_decision": "Run iterate-finalize to record the selected node.",
    "completed": "Iteration completed.",
    "rejected": "Proposal rejected. Start a new iteration if needed.",
    "failed": "Iteration failed. Inspect error_type / error_message.",
    "cancelled": "Iteration cancelled.",
    "stopped_no_recommendation": "No actionable recommendation; iteration stopped.",
}


class InvalidSelectedNode(ValueError):
    """Raised when finalize selects a node outside the current iteration set."""


class IterationService:
    def __init__(
        self,
        experiment_service: ExperimentService,
        *,
        minimum_successful_seeds: int = MINIMUM_SUCCESSFUL_SEEDS,
    ) -> None:
        self.experiments = experiment_service
        self.repo = IterationRepository(experiment_service.session_factory)
        self.minimum_successful_seeds = minimum_successful_seeds

    def start_iteration(
        self,
        baseline_node_id: str,
        candidate_node_id: str,
        seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        baseline = self.experiments.repo.get_node(baseline_node_id)
        candidate = self.experiments.repo.get_node(candidate_node_id)
        if baseline is None:
            raise KeyError(f"未找到 node: {baseline_node_id}")
        if candidate is None:
            raise KeyError(f"未找到 node: {candidate_node_id}")
        if baseline.project_id != candidate.project_id:
            raise ValueError("baseline 与 candidate 必须属于同一 project")

        seed_list = list(seeds or DEFAULT_SEEDS)
        if not seed_list:
            raise ValueError("seeds 不能为空")

        now = utc_now_iso()
        session = IterationSession(
            iteration_id=new_id("iter"),
            project_id=candidate.project_id,
            source_baseline_node_id=baseline_node_id,
            source_candidate_node_id=candidate_node_id,
            status="created",
            seeds=seed_list,
            created_at=now,
            updated_at=now,
        )
        self.repo.save_session(session)

        try:
            feedback = self.experiments.analyze_feedback(
                baseline_node_id, candidate_node_id
            )
            session.feedback_path = feedback.get("feedback_path")
            transition(session, "feedback_ready")
            self.repo.save_session(session)

            try:
                proposal = self.experiments.propose_next(
                    baseline_node_id, candidate_node_id
                )
            except ValueError:
                transition(session, "stopped_no_recommendation")
                self.repo.save_session(session)
                return self._status_payload(session)

            contract = proposal.get("contract") or {}
            session.proposed_node_id = contract.get("node_id")
            session.proposal_path = proposal.get("contract_path")
            if session.proposal_path:
                session.proposal_sha256 = sha256_file(Path(session.proposal_path))
            transition(session, "proposal_ready")
            transition(session, "waiting_approval")
            self.repo.save_session(session)
            return self._status_payload(session)
        except Exception as exc:  # noqa: BLE001
            session.error_type = type(exc).__name__
            session.error_message = str(exc)
            try:
                transition(session, "failed")
            except Exception:  # noqa: BLE001
                session.status = "failed"  # type: ignore[assignment]
                session.updated_at = utc_now_iso()
            self.repo.save_session(session)
            raise

    def approve_and_run(
        self,
        iteration_id: str,
        seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        session = self._get(iteration_id)
        require_status(session, "waiting_approval")
        if not session.proposal_path:
            raise ValueError(f"Iteration {iteration_id} 缺少 proposal_path")

        proposal_path = Path(session.proposal_path)
        if not proposal_path.exists():
            raise FileNotFoundError(f"提案契约不存在: {proposal_path}")

        approved_hash = sha256_file(proposal_path)
        contract_modified = (
            session.proposal_sha256 is not None
            and approved_hash != session.proposal_sha256
        )

        approval = ApprovalRecord(
            approval_id=new_id("approval"),
            iteration_id=iteration_id,
            decision="approved",
            reason=None,
            contract_path=str(proposal_path),
            contract_sha256=approved_hash,
            created_at=utc_now_iso(),
        )
        self.repo.save_approval(approval)

        session.approved_sha256 = approved_hash
        transition(session, "approved")
        transition(session, "running")
        self.repo.save_session(session)

        run_seeds = list(seeds) if seeds is not None else list(session.seeds)
        if seeds is not None:
            session.seeds = run_seeds

        warnings: list[str] = []
        if contract_modified:
            warnings.append(
                "contract_modified_before_approval: proposal_sha256 != approved_sha256"
            )

        try:
            contract = load_contract(proposal_path)
            run_result = self.experiments.run_seeds(
                contract, run_seeds, auto_aggregate=True
            )
        except Exception as exc:  # noqa: BLE001
            session.error_type = type(exc).__name__
            session.error_message = str(exc)
            transition(session, "failed")
            self.repo.save_session(session)
            raise

        results = list(run_result.get("results") or [])
        session.execution_ids = [
            str(item.get("execution_id"))
            for item in results
            if item.get("execution_id")
        ]
        successful = [item for item in results if item.get("status") == "completed"]
        if len(successful) < len(results):
            warnings.append(
                f"{len(results) - len(successful)} of {len(results)} seed runs failed."
            )

        if len(successful) < self.minimum_successful_seeds:
            session.error_type = "insufficient_successful_seeds"
            session.error_message = (
                f"Only {len(successful)} of {len(results)} seed runs completed "
                f"(minimum={self.minimum_successful_seeds})."
            )
            transition(session, "failed")
            self.repo.save_session(session)
            payload = self._status_payload(session)
            payload["warnings"] = warnings
            payload["contract_modified_before_approval"] = contract_modified
            payload["run_result"] = run_result
            return payload

        transition(session, "comparing")
        self.repo.save_session(session)

        new_node_id = session.proposed_node_id
        if not new_node_id:
            new_node_id = contract.node_id
            session.proposed_node_id = new_node_id

        try:
            comparison_paths: list[str] = []
            for source_id in (
                session.source_baseline_node_id,
                session.source_candidate_node_id,
            ):
                feedback = self.experiments.analyze_feedback(source_id, new_node_id)
                path = feedback.get("group_comparison_path") or feedback.get(
                    "feedback_path"
                )
                if path:
                    comparison_paths.append(str(path))
            session.comparison_paths = comparison_paths
            transition(session, "waiting_decision")
            self.repo.save_session(session)
        except Exception as exc:  # noqa: BLE001
            session.error_type = type(exc).__name__
            session.error_message = str(exc)
            transition(session, "failed")
            self.repo.save_session(session)
            raise

        payload = self._status_payload(session)
        payload["warnings"] = warnings
        payload["contract_modified_before_approval"] = contract_modified
        payload["proposal_sha256"] = session.proposal_sha256
        payload["approved_sha256"] = session.approved_sha256
        return payload

    def reject(
        self,
        iteration_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        session = self._get(iteration_id)
        require_status(session, "waiting_approval")

        contract_path = session.proposal_path or ""
        contract_hash = ""
        if session.proposal_path and Path(session.proposal_path).exists():
            contract_hash = sha256_file(Path(session.proposal_path))
        elif session.proposal_sha256:
            contract_hash = session.proposal_sha256

        approval = ApprovalRecord(
            approval_id=new_id("approval"),
            iteration_id=iteration_id,
            decision="rejected",
            reason=reason,
            contract_path=contract_path,
            contract_sha256=contract_hash or "n/a",
            created_at=utc_now_iso(),
        )
        self.repo.save_approval(approval)
        transition(session, "rejected")
        self.repo.save_session(session)
        return self._status_payload(session)

    def finalize(
        self,
        iteration_id: str,
        *,
        selected_node_id: str,
        decision_type: str,
        reason: str,
        evidence_strength: str = "moderate",
    ) -> dict[str, Any]:
        session = self._get(iteration_id)
        require_status(session, "waiting_decision")

        valid_nodes = {
            session.source_baseline_node_id,
            session.source_candidate_node_id,
            session.proposed_node_id,
        }
        valid_nodes.discard(None)
        if selected_node_id not in valid_nodes:
            raise InvalidSelectedNode(
                f"selected_node_id '{selected_node_id}' is not in "
                f"{sorted(str(v) for v in valid_nodes)}"
            )

        alternatives = [
            node_id
            for node_id in (
                session.source_baseline_node_id,
                session.source_candidate_node_id,
                session.proposed_node_id,
            )
            if node_id and node_id != selected_node_id
        ]
        decision = self.experiments.record_decision(
            selected_node_id=selected_node_id,
            alternatives=alternatives,
            decision_type=decision_type,
            reason=reason,
            evidence_strength=evidence_strength,
            baseline_node_id=session.source_baseline_node_id,
            candidate_node_id=session.source_candidate_node_id,
        )
        session.selected_node_id = selected_node_id
        session.decision_id = decision.get("decision_id")
        transition(session, "completed")
        self.repo.save_session(session)

        payload = self._status_payload(session)
        payload["decision"] = decision
        return payload

    def get_status(self, iteration_id: str) -> dict[str, Any]:
        return self._status_payload(self._get(iteration_id))

    def list_iterations(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sessions = self.repo.list_sessions(project_id=project_id, limit=limit)
        return [
            {
                "iteration_id": item.iteration_id,
                "project_id": item.project_id,
                "status": item.status,
                "source_baseline_node_id": item.source_baseline_node_id,
                "source_candidate_node_id": item.source_candidate_node_id,
                "proposed_node_id": item.proposed_node_id,
                "selected_node_id": item.selected_node_id,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in sessions
        ]

    def _get(self, iteration_id: str) -> IterationSession:
        session = self.repo.get_session(iteration_id)
        if session is None:
            raise KeyError(f"未找到 iteration: {iteration_id}")
        return session

    def _status_payload(self, session: IterationSession) -> dict[str, Any]:
        return {
            "iteration_id": session.iteration_id,
            "project_id": session.project_id,
            "status": session.status,
            "source_nodes": {
                "baseline": session.source_baseline_node_id,
                "candidate": session.source_candidate_node_id,
            },
            "source_baseline_node_id": session.source_baseline_node_id,
            "source_candidate_node_id": session.source_candidate_node_id,
            "proposed_node_id": session.proposed_node_id,
            "proposal_path": session.proposal_path,
            "feedback_path": session.feedback_path,
            "proposal_sha256": session.proposal_sha256,
            "approved_sha256": session.approved_sha256,
            "seeds": session.seeds,
            "execution_ids": session.execution_ids,
            "comparisons": [
                Path(path).parent.name if path else path
                for path in session.comparison_paths
            ],
            "comparison_paths": session.comparison_paths,
            "selected_node_id": session.selected_node_id,
            "decision_id": session.decision_id,
            "error_type": session.error_type,
            "error_message": session.error_message,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "next_action": NEXT_ACTIONS.get(session.status, ""),
        }
