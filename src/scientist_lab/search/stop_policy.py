"""StopPolicy for finite experiment trees (v1.1.7)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from scientist_lab.search.models import ExperimentTree, TreeNode
from scientist_lab.search.selection_policy import rank_expandable_parents
from scientist_lab.search.state_machine import is_tree_terminal


class StopPolicy(BaseModel):
    max_depth: int = 3
    max_nodes: int = 10
    max_no_improvement_rounds: int = 2
    minimum_score_improvement: float = 0.01

    @field_validator("max_depth", "max_nodes", "max_no_improvement_rounds")
    @classmethod
    def _positive(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("must be >= 1")
        return int(value)

    @field_validator("minimum_score_improvement")
    @classmethod
    def _nonneg(cls, value: float) -> float:
        if float(value) < 0:
            raise ValueError("minimum_score_improvement must be >= 0")
        return float(value)

    @classmethod
    def from_tree(cls, tree: ExperimentTree) -> StopPolicy:
        return cls(
            max_depth=tree.max_depth,
            max_nodes=tree.max_nodes,
            max_no_improvement_rounds=int(
                getattr(tree, "max_no_improvement_rounds", 2) or 2
            ),
            minimum_score_improvement=float(
                getattr(tree, "minimum_score_improvement", 0.01) or 0.01
            ),
        )


class StopDecision(BaseModel):
    should_stop: bool = False
    status: str | None = None
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def evaluate_stop(
    tree: ExperimentTree,
    nodes: list[TreeNode],
    *,
    policy: StopPolicy | None = None,
    remaining_budget: dict[str, Any] | None = None,
    last_plan_filter: dict[str, Any] | None = None,
    planner_stop_recommended: bool = False,
    planner_stop_reason: str | None = None,
) -> StopDecision:
    """Highest-priority stop decision for the current tree state."""
    if is_tree_terminal(tree.status):
        return StopDecision(
            should_stop=True,
            status=tree.status,
            reason=getattr(tree, "stop_reason", None)
            or f"tree already terminal ({tree.status})",
            details={"already_terminal": True},
        )

    pol = policy or StopPolicy.from_tree(tree)
    rem = dict(remaining_budget or {})
    node_count = len(nodes)
    max_depth_present = max((n.depth for n in nodes), default=0)

    nodes_left = rem.get("max_new_nodes")
    gpu_left = rem.get("max_total_gpu_hours")
    if isinstance(nodes_left, (int, float)) and int(nodes_left) <= 0:
        return StopDecision(
            should_stop=True,
            status="budget_exhausted",
            reason="No remaining node budget.",
            details={"remaining": rem},
        )
    if isinstance(gpu_left, (int, float)) and float(gpu_left) <= 0:
        return StopDecision(
            should_stop=True,
            status="budget_exhausted",
            reason="No remaining GPU-hour budget.",
            details={"remaining": rem},
        )

    if node_count >= tree.max_nodes:
        return StopDecision(
            should_stop=True,
            status="no_valid_candidates",
            reason=f"Tree reached max_nodes={tree.max_nodes}.",
            details={"node_count": node_count, "max_nodes": tree.max_nodes},
        )

    selection = rank_expandable_parents(tree, nodes)
    if not selection.selected:
        if selection.stop_status == "max_depth_reached" or (
            max_depth_present >= pol.max_depth
            and all(
                n.depth >= pol.max_depth
                or n.status in {"pruned", "failed", "stopped"}
                for n in nodes
            )
        ):
            return StopDecision(
                should_stop=True,
                status="max_depth_reached",
                reason=f"All expandable parents reached max_depth={pol.max_depth}.",
                details={"max_depth_present": max_depth_present},
            )
        return StopDecision(
            should_stop=True,
            status=selection.stop_status or "no_valid_candidates",
            reason=selection.reason or "No expandable parent under current limits.",
            details={"blocked_count": len(selection.blocked)},
        )

    if planner_stop_recommended:
        return StopDecision(
            should_stop=True,
            status="no_valid_candidates",
            reason=planner_stop_reason or "Planner recommended stop.",
            details={"planner_stop": True},
        )

    if last_plan_filter is not None:
        accepted = int(last_plan_filter.get("accepted_count") or 0)
        rejected = list(last_plan_filter.get("rejected") or [])
        deferred = list(last_plan_filter.get("deferred") or [])
        if accepted <= 0:
            if rejected and all(
                str(r.get("tree_filter")) == "duplicate_configuration" for r in rejected
            ) and not deferred:
                return StopDecision(
                    should_stop=True,
                    status="no_valid_candidates",
                    reason="All candidates were duplicate configurations.",
                    details={"rejected_count": len(rejected)},
                )
            if rejected and all(
                str(r.get("tree_filter")) == "verifier_rejected" for r in rejected
            ) and not deferred:
                return StopDecision(
                    should_stop=True,
                    status="no_valid_candidates",
                    reason="All candidates were rejected by verifier/critic rules.",
                    details={"rejected_count": len(rejected)},
                )
            return StopDecision(
                should_stop=True,
                status="no_valid_candidates",
                reason="No accepted candidates after tree filtering.",
                details={
                    "rejected_count": len(rejected),
                    "deferred_count": len(deferred),
                },
            )

    rounds = int(getattr(tree, "no_improvement_rounds", 0) or 0)
    if rounds >= pol.max_no_improvement_rounds:
        return StopDecision(
            should_stop=True,
            status="completed",
            reason=(
                f"No score improvement for {rounds} round(s) "
                f"(threshold={pol.minimum_score_improvement})."
            ),
            details={
                "no_improvement_rounds": rounds,
                "best_score_seen": getattr(tree, "best_score_seen", None),
            },
        )

    return StopDecision(should_stop=False)


def update_improvement_counters(
    tree: ExperimentTree,
    *,
    new_scores: list[float],
    policy: StopPolicy | None = None,
) -> ExperimentTree:
    """Update best_score_seen / no_improvement_rounds from newly evaluated scores."""
    pol = policy or StopPolicy.from_tree(tree)
    best = getattr(tree, "best_score_seen", None)
    rounds = int(getattr(tree, "no_improvement_rounds", 0) or 0)
    improved = False
    current_best = float(best) if best is not None else None
    for score in new_scores:
        if current_best is None:
            current_best = float(score)
            improved = True
            continue
        if float(score) >= current_best + pol.minimum_score_improvement:
            current_best = float(score)
            improved = True
    if not new_scores:
        return tree
    if improved:
        rounds = 0
    else:
        rounds += 1
    return tree.model_copy(
        update={
            "best_score_seen": current_best,
            "no_improvement_rounds": rounds,
        }
    )
