from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.agents.context_builder import build_planning_context
from scientist_lab.agents.critic import Critic, CriticReview, MockCritic
from scientist_lab.agents.models import (
    CandidateExperiment,
    ExperimentCandidateRecord,
    ExperimentPlanRecord,
    PlanningContext,
    PlannerOutput,
)
from scientist_lab.agents.legacy_planner import MockPlanner, Planner
from scientist_lab.agents.ranker import rank_candidates
from scientist_lab.agents.repository import AgentPlanRepository
from scientist_lab.domain.models import new_id
from scientist_lab.planning.candidate_verifier import (
    CandidateVerifier,
    output_sha256,
)
from scientist_lab.planning.contract_generator import generate_contract_from_candidate

if TYPE_CHECKING:
    from scientist_lab.llm.limits import ProviderLimits


class AgentPlanningService:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        planner: Planner | None = None,
        critic: Critic | None = None,
        outputs_root: Path | None = None,
        provider_mode: str = "mock",
    ) -> None:
        self._repo = AgentPlanRepository(session_factory)
        self.outputs_root = Path(outputs_root) if outputs_root else None
        self.provider_mode = (provider_mode or "mock").strip().lower()
        if planner is not None or critic is not None:
            self.planner = planner or MockPlanner()
            self.critic = critic or MockCritic()
        else:
            self.configure_provider(self.provider_mode)
        self.verifier = CandidateVerifier()

    def configure_provider(
        self,
        mode: str = "mock",
        *,
        project_id: str | None = None,
        audit_root: Path | str | None = None,
        limits: ProviderLimits | None = None,
        allow_network: bool = False,
        transport: Any = None,
        openai_config: Any = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        """Switch planner/critic between mock / fake / replay / real providers."""
        from scientist_lab.agents.provider_bridge import (
            build_planner_critic,
            normalize_provider_mode,
        )

        resolved = normalize_provider_mode(mode)
        root = audit_root
        if root is None and self.outputs_root is not None:
            root = self.outputs_root / (project_id or "_llm") / "llm"
        self.planner, self.critic = build_planner_critic(
            resolved,
            audit_root=root,
            project_id=project_id,
            limits=limits,
            allow_network=allow_network,
            transport=transport,
            openai_config=openai_config,
            environ=environ,
        )
        self.provider_mode = resolved
        self.requested_provider = resolved
        self.allow_network = bool(allow_network)

    def plan_next(
        self,
        *,
        project_id: str,
        research_goal: str,
        protocol: dict[str, Any] | None,
        nodes: list[Any],
        evidence_records: list[dict[str, Any]] | None = None,
        claim_support_matrix: dict[str, Any] | None = None,
        comparisons: list[dict[str, Any]] | None = None,
        current_best_node_id: str | None = None,
        remaining_budget: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = build_planning_context(
            project_id=project_id,
            research_goal=research_goal,
            protocol=protocol,
            nodes=nodes,
            evidence_records=evidence_records,
            claim_support_matrix=claim_support_matrix,
            comparisons=comparisons,
            current_best_node_id=current_best_node_id,
            remaining_budget=remaining_budget,
        )
        output = self.planner.plan(context)
        return self._persist_plan(context, output)

    def _persist_plan(
        self,
        context: PlanningContext,
        output: PlannerOutput,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        plan_id = new_id("plan")
        output_payload = output.model_dump(mode="json")
        verifications = self.verifier.verify_all(output.candidates, context)

        valid_candidates = []
        candidate_records: list[ExperimentCandidateRecord] = []
        for candidate, verification in zip(output.candidates, verifications, strict=True):
            status = "verified" if verification.valid else "rejected"
            record = ExperimentCandidateRecord(
                candidate_id=candidate.candidate_id,
                plan_id=plan_id,
                candidate_json=candidate.model_dump(mode="json"),
                verification_json=verification.model_dump(mode="json"),
                status=status,  # type: ignore[arg-type]
                created_at=now,
            )
            candidate_records.append(record)
            if verification.valid:
                valid_candidates.append(candidate.candidate_id)

        plan_status = (
            "verified" if valid_candidates or output.stop_recommended else "planner_failed"
        )
        if output.candidates and not valid_candidates and not output.stop_recommended:
            plan_status = "planner_failed"

        model_provider = getattr(self.planner, "model_provider", None) or "mock"
        model_name = getattr(self.planner, "model_name", None) or "mock-planner-v1"
        prompt_version = (
            "provider_v1"
            if model_provider not in {"mock", "mock-planner"}
            else "mock_v1"
        )

        plan = ExperimentPlanRecord(
            plan_id=plan_id,
            project_id=context.project_id,
            status=plan_status,  # type: ignore[arg-type]
            context_json=context.model_dump(mode="json"),
            planner_output_json=output_payload,
            model_provider=str(model_provider),
            model_name=str(model_name),
            prompt_version=prompt_version,
            context_sha256=context.context_sha256,
            output_sha256=output_sha256(output_payload),
            created_at=now,
            updated_at=now,
        )
        self._repo.upsert_plan(plan)
        for record in candidate_records:
            self._repo.upsert_candidate(record)

        return {
            "plan_id": plan_id,
            "project_id": context.project_id,
            "status": plan.status,
            "model_provider": plan.model_provider,
            "model_name": plan.model_name,
            "prompt_version": plan.prompt_version,
            "requested_provider": getattr(self, "requested_provider", None)
            or self.provider_mode,
            "actual_provider": plan.model_provider,
            "fallback_used": False,
            "context_sha256": plan.context_sha256,
            "output_sha256": plan.output_sha256,
            "reasoning_summary": output.reasoning_summary,
            "stop_recommended": output.stop_recommended,
            "stop_reason": output.stop_reason,
            "candidates": [
                {
                    **item.candidate_json,
                    "verification": item.verification_json,
                    "status": item.status,
                }
                for item in candidate_records
            ],
            "valid_candidate_count": len(valid_candidates),
        }

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        plan = self._require_plan(plan_id)
        candidates = self._repo.list_candidates(plan_id)
        payload = plan.model_dump(mode="json")
        payload["candidates"] = [self._candidate_view(item) for item in candidates]
        return payload

    def list_plans(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        plans = self._repo.list_plans(project_id=project_id)
        return [
            {
                "plan_id": item.plan_id,
                "project_id": item.project_id,
                "status": item.status,
                "model_name": item.model_name,
                "prompt_version": item.prompt_version,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "stop_recommended": bool(
                    (item.planner_output_json or {}).get("stop_recommended")
                ),
                "candidate_count": len(
                    (item.planner_output_json or {}).get("candidates") or []
                ),
            }
            for item in plans
        ]

    def review_plan(self, plan_id: str) -> dict[str, Any]:
        plan = self._require_plan(plan_id)
        context = PlanningContext.model_validate(plan.context_json)
        reviews: list[dict[str, Any]] = []
        for record in self._repo.list_candidates(plan_id):
            if record.status == "rejected":
                continue
            if not (record.verification_json or {}).get("valid", False):
                continue
            candidate = CandidateExperiment.model_validate(record.candidate_json)
            review = self.critic.review(candidate, context)
            status = "reviewed"
            if review.recommendation == "reject":
                status = "rejected"
            updated = record.model_copy(
                update={
                    "critic_review_json": review.model_dump(mode="json"),
                    "status": status,
                }
            )
            self._repo.upsert_candidate(updated)
            reviews.append(review.model_dump(mode="json"))

        plan = plan.model_copy(
            update={
                "status": "reviewed",
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0),
            }
        )
        self._repo.upsert_plan(plan)
        return {"plan_id": plan_id, "status": plan.status, "reviews": reviews}

    def rank_plan_candidates(self, plan_id: str, *, max_keep: int = 3) -> dict[str, Any]:
        plan = self._require_plan(plan_id)
        pairs: list[tuple[CandidateExperiment, CriticReview | None]] = []
        records_by_id: dict[str, ExperimentCandidateRecord] = {}
        for record in self._repo.list_candidates(plan_id):
            records_by_id[record.candidate_id] = record
            if record.status in {"rejected"}:
                continue
            if not (record.verification_json or {}).get("valid", False):
                continue
            candidate = CandidateExperiment.model_validate(record.candidate_json)
            critic = None
            if record.critic_review_json:
                critic = CriticReview.model_validate(record.critic_review_json)
            pairs.append((candidate, critic))

        ranked = rank_candidates(pairs, max_keep=max_keep)
        kept_ids = {item["candidate_id"] for item in ranked}
        for record in records_by_id.values():
            if record.candidate_id in kept_ids:
                score_row = next(
                    item for item in ranked if item["candidate_id"] == record.candidate_id
                )
                updated = record.model_copy(
                    update={
                        "final_score": score_row["final_score"],
                        "rank": score_row["rank"],
                        "status": "ranked",
                    }
                )
                self._repo.upsert_candidate(updated)
            elif record.status not in {"rejected", "approved", "contract_generated"}:
                # Not in top-k: keep reviewed/verified but unranked
                if record.status in {"reviewed", "verified", "ranked"}:
                    updated = record.model_copy(
                        update={"rank": None, "final_score": None, "status": "reviewed"}
                    )
                    self._repo.upsert_candidate(updated)

        plan = plan.model_copy(
            update={
                "status": "ranked",
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0),
            }
        )
        self._repo.upsert_plan(plan)
        return {"plan_id": plan_id, "status": plan.status, "ranking": ranked}

    def approve_candidate(self, plan_id: str, candidate_id: str) -> dict[str, Any]:
        plan = self._require_plan(plan_id)
        record = self._require_candidate(plan_id, candidate_id)
        if record.status == "approved":
            raise ValueError(f"candidate already approved: {candidate_id}")
        if record.status == "rejected":
            raise ValueError(f"rejected candidate cannot be approved: {candidate_id}")
        if record.status not in {"verified", "reviewed", "ranked"}:
            raise ValueError(
                f"candidate status {record.status} cannot be approved"
            )
        if not (record.verification_json or {}).get("valid", False):
            raise ValueError("candidate failed verification")
        updated = record.model_copy(update={"status": "approved"})
        self._repo.upsert_candidate(updated)
        plan = plan.model_copy(
            update={
                "status": "approved",
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0),
            }
        )
        self._repo.upsert_plan(plan)
        return self._candidate_view(updated)

    def reject_candidate(
        self, plan_id: str, candidate_id: str, *, reason: str | None = None
    ) -> dict[str, Any]:
        self._require_plan(plan_id)
        record = self._require_candidate(plan_id, candidate_id)
        if record.status == "approved":
            raise ValueError("approved candidate cannot be rejected; generate contract or keep")
        if record.status == "rejected":
            raise ValueError(f"candidate already rejected: {candidate_id}")
        review = dict(record.critic_review_json or {})
        if reason:
            review.setdefault("human_reject_reason", reason)
        updated = record.model_copy(
            update={"status": "rejected", "critic_review_json": review or None}
        )
        self._repo.upsert_candidate(updated)
        return self._candidate_view(updated)

    def generate_contract(
        self,
        plan_id: str,
        candidate_id: str,
        *,
        parent_contract: dict[str, Any],
        existing_node_ids: set[str],
    ) -> dict[str, Any]:
        plan = self._require_plan(plan_id)
        record = self._require_candidate(plan_id, candidate_id)
        if record.status != "approved":
            raise ValueError(
                "only approved candidates can generate contracts "
                f"(status={record.status})"
            )
        candidate = CandidateExperiment.model_validate(record.candidate_json)
        out_dir = None
        if self.outputs_root is not None:
            out_dir = (
                self.outputs_root
                / plan.project_id
                / "plans"
                / plan_id
            )
        result = generate_contract_from_candidate(
            parent_contract=parent_contract,
            candidate=candidate,
            plan_id=plan_id,
            existing_node_ids=existing_node_ids,
            output_dir=out_dir,
        )
        updated = record.model_copy(update={"status": "contract_generated"})
        self._repo.upsert_candidate(updated)
        plan = plan.model_copy(
            update={
                "status": "contract_generated",
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0),
            }
        )
        self._repo.upsert_plan(plan)
        result["candidate_status"] = updated.status
        result["plan_status"] = plan.status
        return result

    def _require_plan(self, plan_id: str) -> ExperimentPlanRecord:
        plan = self._repo.get_plan(plan_id)
        if plan is None:
            raise KeyError(f"plan not found: {plan_id}")
        return plan

    def _require_candidate(
        self, plan_id: str, candidate_id: str
    ) -> ExperimentCandidateRecord:
        record = self._repo.get_candidate(candidate_id)
        if record is None or record.plan_id != plan_id:
            raise KeyError(f"candidate not found in plan: {candidate_id}")
        return record

    @staticmethod
    def _candidate_view(item: ExperimentCandidateRecord) -> dict[str, Any]:
        return {
            **item.candidate_json,
            "verification": item.verification_json,
            "critic_review": item.critic_review_json,
            "status": item.status,
            "rank": item.rank,
            "final_score": item.final_score,
        }
