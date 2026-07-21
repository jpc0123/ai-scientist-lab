"""Export ExperimentTree as JSON or Mermaid (v1.1.9)."""

from __future__ import annotations

import re
from typing import Any

from scientist_lab.search.models import ExperimentTree, TreeNode


def _safe_mermaid_id(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(raw))
    if not text:
        text = "node"
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def _node_label(node: TreeNode) -> str:
    parts = [node.experiment_node_id, f"{node.node_type} | {node.status}"]
    extras: list[str] = []
    if node.score is not None:
        extras.append(f"score={node.score:.2f}")
    if node.expansion_priority is not None:
        extras.append(f"expand={node.expansion_priority:.2f}")
    if extras:
        parts.append(", ".join(extras))
    # Mermaid node text: escape quotes / newlines.
    return "<br/>".join(parts).replace('"', "'")


def render_mermaid(
    tree: ExperimentTree,
    nodes: list[TreeNode],
) -> str:
    """Render a flowchart TD diagram for the experiment tree."""
    lines = [
        "flowchart TD",
        f'  classDef terminal fill:#eee,stroke:#999,color:#333;',
    ]
    if not nodes:
        empty_id = _safe_mermaid_id(tree.tree_id)
        lines.append(f'  {empty_id}["(empty tree {tree.tree_id})"]')
        return "\n".join(lines)

    by_id = {n.tree_node_id: n for n in nodes}
    for node in nodes:
        mid = _safe_mermaid_id(node.tree_node_id)
        label = _node_label(node)
        if node.status in {"pruned", "failed", "stopped"}:
            lines.append(f'  {mid}["{label}"]:::terminal')
        else:
            lines.append(f'  {mid}["{label}"]')

    for node in nodes:
        if not node.parent_tree_node_id:
            continue
        if node.parent_tree_node_id not in by_id:
            continue
        parent = _safe_mermaid_id(node.parent_tree_node_id)
        child = _safe_mermaid_id(node.tree_node_id)
        lines.append(f"  {parent} --> {child}")

    lines.append(f'  %% tree_id={tree.tree_id} status={tree.status}')
    return "\n".join(lines)


def export_tree_payload(
    tree: ExperimentTree,
    nodes: list[TreeNode],
    *,
    ascii_tree: str,
    format: str = "json",
) -> dict[str, Any]:
    fmt = (format or "json").strip().lower()
    if fmt not in {"json", "mermaid"}:
        raise ValueError(f"unsupported export format: {format}")

    mermaid = render_mermaid(tree, nodes)
    payload: dict[str, Any] = {
        "tree_id": tree.tree_id,
        "format": fmt,
        "project_id": tree.project_id,
        "protocol_id": tree.protocol_id,
        "status": tree.status,
        "stop_reason": tree.stop_reason,
        "root_node_id": tree.root_node_id,
        "selected_node_id": tree.selected_node_id,
        "max_depth": tree.max_depth,
        "max_nodes": tree.max_nodes,
        "max_children_per_node": tree.max_children_per_node,
        "node_count": len(nodes),
        "ascii_tree": ascii_tree,
        "mermaid": mermaid,
        "nodes": [n.model_dump(mode="json") for n in nodes],
        "created_at": tree.created_at.isoformat()
        if hasattr(tree.created_at, "isoformat")
        else tree.created_at,
        "updated_at": tree.updated_at.isoformat()
        if hasattr(tree.updated_at, "isoformat")
        else tree.updated_at,
    }
    if fmt == "mermaid":
        return {
            "tree_id": tree.tree_id,
            "format": "mermaid",
            "status": tree.status,
            "mermaid": mermaid,
            "ascii_tree": ascii_tree,
        }
    return payload
