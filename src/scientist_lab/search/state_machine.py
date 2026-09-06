from __future__ import annotations

from scientist_lab.search.models import TREE_TERMINAL_STATUSES, TreeNodeStatus, TreeStatus


class InvalidTreeTransition(ValueError):
    """Raised when a tree or tree-node status jump is illegal."""


TREE_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset(
        {
            "active",
            "waiting_approval",
            "failed",
            "user_stopped",
        }
    ),
    "active": frozenset(
        {
            "waiting_approval",
            "running",
            "evaluating",
            "completed",
            "budget_exhausted",
            "max_depth_reached",
            "no_valid_candidates",
            "user_stopped",
            "failed",
        }
    ),
    "waiting_approval": frozenset(
        {
            "active",
            "running",
            "evaluating",
            "completed",
            "budget_exhausted",
            "max_depth_reached",
            "no_valid_candidates",
            "user_stopped",
            "failed",
        }
    ),
    "running": frozenset(
        {
            "evaluating",
            "waiting_approval",
            "active",
            "completed",
            "budget_exhausted",
            "max_depth_reached",
            "no_valid_candidates",
            "user_stopped",
            "failed",
        }
    ),
    "evaluating": frozenset(
        {
            "active",
            "waiting_approval",
            "completed",
            "budget_exhausted",
            "max_depth_reached",
            "no_valid_candidates",
            "user_stopped",
            "failed",
        }
    ),
    "completed": frozenset(),
    "budget_exhausted": frozenset(),
    "max_depth_reached": frozenset(),
    "no_valid_candidates": frozenset(),
    "user_stopped": frozenset(),
    "failed": frozenset(),
}

NODE_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"proposed", "waiting_approval", "evaluated", "pruned", "stopped", "failed"}),
    "proposed": frozenset({"waiting_approval", "pruned", "stopped", "failed"}),
    "waiting_approval": frozenset({"running", "pruned", "stopped", "failed", "proposed"}),
    "running": frozenset({"evaluated", "failed", "stopped"}),
    "evaluated": frozenset({"selected", "pruned", "stopped"}),
    "selected": frozenset({"pruned", "stopped"}),
    "pruned": frozenset(),
    "failed": frozenset({"pruned", "stopped"}),
    "stopped": frozenset(),
}


def assert_tree_transition(current: TreeStatus | str, new: TreeStatus | str) -> None:
    cur = str(current)
    nxt = str(new)
    if cur == nxt:
        return
    allowed = TREE_TRANSITIONS.get(cur)
    if allowed is None:
        raise InvalidTreeTransition(f"unknown tree status: {cur}")
    if nxt not in allowed:
        raise InvalidTreeTransition(f"illegal tree transition: {cur} -> {nxt}")


def assert_node_transition(
    current: TreeNodeStatus | str, new: TreeNodeStatus | str
) -> None:
    cur = str(current)
    nxt = str(new)
    if cur == nxt:
        return
    allowed = NODE_TRANSITIONS.get(cur)
    if allowed is None:
        raise InvalidTreeTransition(f"unknown tree node status: {cur}")
    if nxt not in allowed:
        raise InvalidTreeTransition(f"illegal tree node transition: {cur} -> {nxt}")


def is_tree_terminal(status: TreeStatus | str) -> bool:
    return str(status) in TREE_TERMINAL_STATUSES
