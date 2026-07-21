"""Best-First parent selection for finite experiment trees (v1.1.3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from scientist_lab.search.models import ExperimentTree, TreeNode
from scientist_lab.search.state_machine import is_tree_terminal


# Nodes that may serve as expansion parents.
EXPANDABLE_STATUSES: frozenset[str] = frozenset({"evaluated", "selected"})
# Root may still be "created" after tree-create; it represents an existing experiment.
ROOT_EXPANDABLE_STATUSES: frozenset[str] = frozenset(
    {"created", "evaluated", "selected"}
)

BLOCKED_STATUSES: frozenset[str] = frozenset(
    {"pruned", "failed", "stopped", "proposed", "waiting_approval", "running"}
)


class ParentCandidate(BaseModel):
    tree_node_id: str
    experiment_node_id: str
    node_type: str
    status: str
    depth: int
    score: float | None = None
    expansion_priority: float
    child_count: int
    reasons_ok: list[str] = Field(default_factory=list)
    reasons_blocked: list[str] = Field(default_factory=list)
    expandable: bool = False


class ParentSelectionResult(BaseModel):
    tree_id: str
    selected: ParentCandidate | None = None
    candidates: list[ParentCandidate] = Field(default_factory=list)
    blocked: list[ParentCandidate] = Field(default_factory=list)
    reason: str | None = None
    stop_suggested: bool = False
    stop_status: str | None = None


def is_expandable_status(node: TreeNode) -> bool:
    if node.node_type == "root":
        return node.status in ROOT_EXPANDABLE_STATUSES
    return node.status in EXPANDABLE_STATUSES


def explain_expandability(
    tree: ExperimentTree,
    node: TreeNode,
    *,
    child_count: int,
    node_count: int,
) -> ParentCandidate:
    ok: list[str] = []
    blocked: list[str] = []

    if is_tree_terminal(tree.status):
        blocked.append(f"tree is terminal ({tree.status})")

    if node.status in BLOCKED_STATUSES:
        blocked.append(f"node status blocked ({node.status})")
    elif not is_expandable_status(node):
        blocked.append(
            f"node not yet expandable (status={node.status}, type={node.node_type})"
        )
    else:
        ok.append(f"status={node.status} is expandable")

    if node.depth >= tree.max_depth:
        blocked.append(f"max_depth reached (depth={node.depth}>={tree.max_depth})")
    else:
        ok.append(f"depth {node.depth} < max_depth {tree.max_depth}")

    if child_count >= tree.max_children_per_node:
        blocked.append(
            f"max_children reached ({child_count}>={tree.max_children_per_node})"
        )
    else:
        ok.append(
            f"children {child_count} < max_children {tree.max_children_per_node}"
        )

    if node_count >= tree.max_nodes:
        blocked.append(f"tree max_nodes reached ({node_count}>={tree.max_nodes})")
    else:
        ok.append(f"tree nodes {node_count} < max_nodes {tree.max_nodes}")

    # Debug nodes are expandable for debug/replication follow-ups, but noted.
    if node.node_type == "debug":
        ok.append("debug parent: prefer diagnostic follow-ups, not improve claims")

    priority = (
        float(node.expansion_priority)
        if node.expansion_priority is not None
        else (float(node.score) if node.score is not None else 0.0)
    )

    return ParentCandidate(
        tree_node_id=node.tree_node_id,
        experiment_node_id=node.experiment_node_id,
        node_type=node.node_type,
        status=node.status,
        depth=node.depth,
        score=node.score,
        expansion_priority=priority,
        child_count=child_count,
        reasons_ok=ok,
        reasons_blocked=blocked,
        expandable=len(blocked) == 0,
    )


def rank_expandable_parents(
    tree: ExperimentTree,
    nodes: list[TreeNode],
    *,
    children_by_parent: dict[str | None, list[TreeNode]] | None = None,
) -> ParentSelectionResult:
    """Best-First ranking: highest expansion_priority among expandable nodes."""
    by_parent: dict[str | None, list[TreeNode]] = {}
    if children_by_parent is None:
        for node in nodes:
            by_parent.setdefault(node.parent_tree_node_id, []).append(node)
    else:
        by_parent = children_by_parent

    node_count = len(nodes)
    evaluated: list[ParentCandidate] = []
    blocked: list[ParentCandidate] = []

    for node in nodes:
        child_count = len(by_parent.get(node.tree_node_id, []))
        candidate = explain_expandability(
            tree, node, child_count=child_count, node_count=node_count
        )
        if candidate.expandable:
            evaluated.append(candidate)
        else:
            blocked.append(candidate)

    evaluated.sort(
        key=lambda c: (
            -c.expansion_priority,
            -(c.score if c.score is not None else -1.0),
            c.depth,
            c.tree_node_id,
        )
    )

    if is_tree_terminal(tree.status):
        return ParentSelectionResult(
            tree_id=tree.tree_id,
            selected=None,
            candidates=[],
            blocked=blocked,
            reason=f"tree is terminal ({tree.status})",
            stop_suggested=True,
            stop_status=tree.status,
        )

    if node_count >= tree.max_nodes:
        return ParentSelectionResult(
            tree_id=tree.tree_id,
            selected=None,
            candidates=[],
            blocked=blocked,
            reason="max_nodes reached; no expandable parent",
            stop_suggested=True,
            stop_status="max_depth_reached"
            if all(n.depth >= tree.max_depth for n in nodes)
            else "no_valid_candidates",
        )

    if not evaluated:
        # Distinguish depth vs no candidates
        all_at_depth = bool(nodes) and all(n.depth >= tree.max_depth for n in nodes)
        stop_status = "max_depth_reached" if all_at_depth else "no_valid_candidates"
        return ParentSelectionResult(
            tree_id=tree.tree_id,
            selected=None,
            candidates=[],
            blocked=blocked,
            reason="no expandable parent under current limits",
            stop_suggested=True,
            stop_status=stop_status,
        )

    best = evaluated[0]
    return ParentSelectionResult(
        tree_id=tree.tree_id,
        selected=best,
        candidates=evaluated,
        blocked=blocked,
        reason=f"selected by best-first expansion_priority={best.expansion_priority:.4f}",
        stop_suggested=False,
        stop_status=None,
    )


def selection_payload(result: ParentSelectionResult) -> dict[str, Any]:
    return result.model_dump(mode="json")
