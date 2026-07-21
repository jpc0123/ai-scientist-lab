"""Planner interface and rule-based MockPlanner (no LLM dependency)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from scientist_lab.agents.models import (
    CandidateExperiment,
    ExpectedOutcome,
    PlanningContext,
    PlannerOutput,
)
from scientist_lab.domain.models import new_id
from scientist_lab.planning.candidate_verifier import parameter_fingerprint


class Planner(ABC):
    @abstractmethod
    def plan(self, context: PlanningContext) -> PlannerOutput:
        raise NotImplementedError


class MockPlanner(Planner):
    """Deterministic planner for workflow debugging before real LLM."""

    def __init__(self, *, max_candidates: int = 3) -> None:
        self.max_candidates = max(1, min(3, int(max_candidates)))

    def plan(self, context: PlanningContext) -> PlannerOutput:
        budget = dict(context.remaining_budget or {})
        if int(budget.get("max_new_nodes", 0) or 0) <= 0:
            return PlannerOutput(
                project_id=context.project_id,
                reasoning_summary=(
                    "No remaining node budget; stop recommended instead of "
                    "proposing uninformative experiments."
                ),
                candidates=[],
                stop_recommended=True,
                stop_reason=(
                    "Remaining budget is insufficient for an informative "
                    "matched-seed experiment."
                ),
            )

        parent = context.current_best_node_id or self._first_node_id(context)
        if not parent:
            return PlannerOutput(
                project_id=context.project_id,
                reasoning_summary="No nodes available to plan from.",
                candidates=[],
                stop_recommended=True,
                stop_reason="Project has no experiment nodes.",
            )

        gaps = self._evidence_gaps(context)
        drafts: list[CandidateExperiment] = []

        # Scenario 1: ablation removing fusion.
        drafts.append(
            self._candidate(
                parent_node_id=parent,
                title="Ablate fusion → RGB-only control",
                hypothesis=(
                    "If early fusion drives AP_small gains, removing fusion "
                    "(RGB-only) under the same protocol should reduce AP_small."
                ),
                experiment_type="ablation",
                parameter_changes={"input_mode": "rgb", "fusion_method": "none"},
                gaps=gaps
                or ["Missing controlled ablation for fusion contribution."],
                priority=0.86,
                rationale=(
                    "Isolate fusion contribution with a matched-protocol RGB-only "
                    "control instead of enlarging the model."
                ),
            )
        )

        # Scenario 2: thermal-only control / robustness modality check.
        drafts.append(
            self._candidate(
                parent_node_id=parent,
                title="Thermal-only matched control",
                hypothesis=(
                    "Thermal-only under the same protocol clarifies whether fusion "
                    "gains require RGB, thermal, or both."
                ),
                experiment_type="ablation",
                parameter_changes={
                    "input_mode": "thermal",
                    "fusion_method": "none",
                },
                gaps=gaps
                or ["Missing single-modality thermal control under matched budget."],
                priority=0.78,
                rationale=(
                    "Add a thermal-only node to complete the modality ablation set."
                ),
            )
        )

        # Scenario 3: efficiency / keep fusion but only if we invent an allowed tweak.
        # With only input_mode/fusion_method allowed, propose early_concat explicit
        # replication-style only when fingerprint not tested; else skip.
        drafts.append(
            self._candidate(
                parent_node_id=parent,
                title="Confirm early-fusion configuration",
                hypothesis=(
                    "Reconfirming early_concat fusion under matched seeds improves "
                    "confidence before claiming modality benefits."
                ),
                experiment_type="replication",
                parameter_changes={
                    "input_mode": "rgbt",
                    "fusion_method": "early_concat",
                },
                gaps=gaps
                or ["Stability of fusion gains across matched seeds is unclear."],
                priority=0.55,
                rationale=(
                    "If fusion fingerprint is already tested, verifier will reject "
                    "this duplicate; otherwise it strengthens replication evidence."
                ),
                estimated_cost={"gpu_hours": 1.5, "seeds": 3},
            )
        )

        tested = set(context.tested_parameter_fingerprints or [])
        kept: list[CandidateExperiment] = []
        for item in drafts:
            fp = parameter_fingerprint(item.parameter_changes)
            if fp in tested:
                continue
            kept.append(item)
            if len(kept) >= self.max_candidates:
                break

        if not kept:
            return PlannerOutput(
                project_id=context.project_id,
                reasoning_summary=(
                    "All simple modality ablations appear already tested under the "
                    "current protocol fingerprints."
                ),
                candidates=[],
                stop_recommended=True,
                stop_reason=(
                    "No novel allowed parameter combinations remain within the "
                    "current protocol allow-list."
                ),
            )

        return PlannerOutput(
            project_id=context.project_id,
            reasoning_summary=(
                "Propose protocol-constrained modality ablations that address "
                "evidence gaps around fusion contribution, without changing code, "
                "dataset, or environment."
            ),
            candidates=kept[: self.max_candidates],
            stop_recommended=False,
            stop_reason=None,
        )

    @staticmethod
    def _first_node_id(context: PlanningContext) -> str | None:
        for item in context.nodes:
            node_id = item.get("node_id")
            if node_id:
                return str(node_id)
        return None

    @staticmethod
    def _evidence_gaps(context: PlanningContext) -> list[str]:
        gaps: list[str] = []
        matrix = dict(context.claim_support_matrix or {})
        for claim in matrix.get("claims") or []:
            status = str(claim.get("support_status") or "")
            text = str(claim.get("claim_text") or claim.get("reason") or "")
            if status in {"partially_supported", "unsupported", "blocked"}:
                gaps.append(
                    f"{status}: {text}".strip(": ")
                )
        for record in context.evidence_records or []:
            for limitation in record.get("limitations") or []:
                text = str(limitation)
                if "ablation" in text.lower() or "stand-in" in text.lower():
                    gaps.append(text)
        # Deduplicate preserve order
        seen: set[str] = set()
        unique: list[str] = []
        for item in gaps:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique[:6]

    def _candidate(
        self,
        *,
        parent_node_id: str,
        title: str,
        hypothesis: str,
        experiment_type: str,
        parameter_changes: dict[str, Any],
        gaps: list[str],
        priority: float,
        rationale: str,
        estimated_cost: dict[str, Any] | None = None,
    ) -> CandidateExperiment:
        return CandidateExperiment(
            candidate_id=new_id("candidate"),
            parent_node_id=parent_node_id,
            title=title,
            hypothesis=hypothesis,
            experiment_type=experiment_type,  # type: ignore[arg-type]
            parameter_changes=parameter_changes,
            expected_outcomes=[
                ExpectedOutcome(
                    metric="AP_small",
                    direction="decrease",
                    rationale="Ablating fusion should reduce small-object AP if fusion helps.",
                ),
                ExpectedOutcome(
                    metric="mAP50_95",
                    direction="decrease",
                    rationale="Overall detection should not improve when fusion is removed.",
                ),
            ]
            if experiment_type == "ablation"
            else [
                ExpectedOutcome(
                    metric="mAP50_95",
                    direction="maintain",
                    rationale="Replication should stay directionally consistent.",
                )
            ],
            evidence_gap_addressed=list(gaps[:3]),
            success_criteria={
                "matched_seeds": 3,
                "primary_metric": "mAP50_95",
                "note": "Interpret under exploratory_comparison only.",
            },
            failure_criteria={
                "if_no_metric_drop_after_ablation": (
                    "Fusion contribution claim remains unsupported."
                )
            },
            estimated_cost=estimated_cost
            or {"gpu_hours": 1.0, "seeds": 3, "nodes": 1},
            priority=priority,
            rationale=rationale,
            claim_limitations=[
                "Fast Eval / stand-in evidence remains weak.",
                "Do not claim full RGBT-Tiny or SOTA improvements.",
            ],
        )
