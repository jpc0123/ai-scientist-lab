from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.domain.models import new_id
from scientist_lab.protocols.verifier import ProtocolVerifier
from scientist_lab.search.models import ExperimentTree, TreeNode
from scientist_lab.search.repository import TreeRepository
from scientist_lab.search.scoring import ScoreBreakdown, compute_scores
from scientist_lab.search.state_machine import (
    assert_node_transition,
    assert_tree_transition,
    is_tree_terminal,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class TreeSearchService:
    """Finite experiment tree (v1.1.2: skeleton + node scoring)."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        get_project: Callable[[str], Any] | None = None,
        get_node: Callable[[str], Any] | None = None,
        get_protocol: Callable[[str], Any] | None = None,
        get_node_aggregate: Callable[[str], dict[str, Any] | None] | None = None,
        list_evidence: Callable[[str], list[dict[str, Any]]] | None = None,
        get_claim_matrix: Callable[[str], dict[str, Any] | None] | None = None,
        get_remaining_budget: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> None:
        self._repo = TreeRepository(session_factory)
        self._get_project = get_project
        self._get_node = get_node
        self._get_protocol = get_protocol
        self._get_node_aggregate = get_node_aggregate
        self._list_evidence = list_evidence
        self._get_claim_matrix = get_claim_matrix
        self._get_remaining_budget = get_remaining_budget
        self._protocol_verifier = ProtocolVerifier()

    def create_tree(
        self,
        project_id: str,
        *,
        root_node_id: str,
        protocol_id: str,
        max_depth: int = 3,
        max_nodes: int = 8,
        max_children: int = 3,
        tree_id: str | None = None,
        budget_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_limits(
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_children=max_children,
        )
        self._validate_create(
            project_id=project_id,
            root_node_id=root_node_id,
            protocol_id=protocol_id,
        )

        now = _utc_now()
        resolved_tree_id = (tree_id or new_id("tree")).strip()
        if self._repo.get_tree(resolved_tree_id) is not None:
            raise ValueError(f"duplicate tree_id: {resolved_tree_id}")

        tree = ExperimentTree(
            tree_id=resolved_tree_id,
            project_id=project_id,
            protocol_id=protocol_id,
            root_node_id=root_node_id,
            status="created",
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_children_per_node=max_children,
            budget_id=budget_id or project_id,
            selected_node_id=None,
            created_at=now,
            updated_at=now,
        )
        root = TreeNode(
            tree_node_id=new_id("tnode"),
            tree_id=resolved_tree_id,
            experiment_node_id=root_node_id,
            parent_tree_node_id=None,
            depth=0,
            node_type="root",
            status="created",
            created_at=now,
            updated_at=now,
        )
        self._repo.create_tree(tree)
        self._repo.create_node(root)
        return self.show_tree(resolved_tree_id)

    @staticmethod
    def _validate_limits(
        *,
        max_depth: int,
        max_nodes: int,
        max_children: int,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if max_nodes <= 1:
            raise ValueError("max_nodes must be > 1")
        if max_children <= 0:
            raise ValueError("max_children must be > 0")
        if max_nodes < max_children:
            raise ValueError("max_nodes must be >= max_children")

    def _validate_create(
        self,
        *,
        project_id: str,
        root_node_id: str,
        protocol_id: str,
    ) -> None:
        if self._get_project is not None:
            project = self._get_project(project_id)
            if project is None:
                raise KeyError(f"project not found: {project_id}")

        if self._get_protocol is None:
            raise ValueError("protocol lookup is required")
        protocol = self._get_protocol(protocol_id)
        if protocol is None:
            raise KeyError(f"protocol not found: {protocol_id}")
        proto_project = getattr(protocol, "project_id", None)
        if proto_project is not None and str(proto_project) != project_id:
            raise ValueError(
                f"protocol {protocol_id} belongs to project {proto_project}, "
                f"not {project_id}"
            )

        if self._get_node is None:
            raise ValueError("experiment node lookup is required")
        node = self._get_node(root_node_id)
        if node is None:
            raise KeyError(f"root experiment node not found: {root_node_id}")
        node_project = getattr(node, "project_id", None)
        if node_project is not None and str(node_project) != project_id:
            raise ValueError(
                f"root node {root_node_id} belongs to project {node_project}, "
                f"not {project_id}"
            )

        contract_raw = getattr(node, "contract_json", None) or {}
        if not isinstance(contract_raw, dict) or not contract_raw:
            raise ValueError(f"root node {root_node_id} has no contract_json")
        contract = ExperimentContract.model_validate(contract_raw)
        report = self._protocol_verifier.verify_contract(protocol, contract)
        if not report.valid:
            raise ValueError(
                "root node does not comply with protocol: "
                + "; ".join(report.blocking_issues)
            )
        node_protocol = (contract.protocol_id or "").strip()
        if node_protocol and node_protocol != protocol_id:
            raise ValueError(
                f"protocol mismatch: root node uses {node_protocol}, "
                f"tree requests {protocol_id}"
            )

    def require_tree(self, tree_id: str) -> ExperimentTree:
        tree = self._repo.get_tree(tree_id)
        if tree is None:
            raise KeyError(f"tree not found: {tree_id}")
        return tree

    def list_nodes(self, tree_id: str) -> list[dict[str, Any]]:
        self.require_tree(tree_id)
        return [n.model_dump(mode="json") for n in self._repo.list_nodes(tree_id)]

    def set_tree_status(self, tree_id: str, status: str) -> ExperimentTree:
        tree = self.require_tree(tree_id)
        assert_tree_transition(tree.status, status)
        tree = tree.model_copy(
            update={"status": status, "updated_at": _utc_now()}
        )
        return self._repo.upsert_tree(tree)

    def set_node_status(self, tree_node_id: str, status: str) -> TreeNode:
        node = self._repo.get_node(tree_node_id)
        if node is None:
            raise KeyError(f"tree node not found: {tree_node_id}")
        assert_node_transition(node.status, status)
        node = node.model_copy(
            update={"status": status, "updated_at": _utc_now()}
        )
        return self._repo.upsert_node(node)

    def register_experiment_node(
        self,
        tree_id: str,
        *,
        experiment_node_id: str,
        parent_tree_node_id: str,
        node_type: str = "improve",
        depth: int | None = None,
    ) -> TreeNode:
        """Attach an ExperimentNode into the tree (no planning). Enforces uniqueness."""
        tree = self.require_tree(tree_id)
        if is_tree_terminal(tree.status):
            raise ValueError(f"tree {tree_id} is terminal ({tree.status})")
        parent = self._repo.get_node(parent_tree_node_id)
        if parent is None or parent.tree_id != tree_id:
            raise KeyError(f"parent tree node not found: {parent_tree_node_id}")
        if self._repo.get_node_by_experiment(tree_id, experiment_node_id) is not None:
            raise ValueError(
                f"experiment node already in tree: {experiment_node_id}"
            )
        resolved_depth = parent.depth + 1 if depth is None else depth
        if resolved_depth > tree.max_depth:
            raise ValueError("max_depth exceeded")
        if self._repo.count_nodes(tree_id) >= tree.max_nodes:
            raise ValueError("max_nodes exceeded")
        now = _utc_now()
        node = TreeNode(
            tree_node_id=new_id("tnode"),
            tree_id=tree_id,
            experiment_node_id=experiment_node_id,
            parent_tree_node_id=parent_tree_node_id,
            depth=resolved_depth,
            node_type=node_type,  # type: ignore[arg-type]
            status="created",
            created_at=now,
            updated_at=now,
        )
        return self._repo.create_node(node)

    def tree_status(self, tree_id: str) -> dict[str, Any]:
        tree = self.require_tree(tree_id)
        nodes = self._repo.list_nodes(tree_id)
        return {
            "tree_id": tree.tree_id,
            "project_id": tree.project_id,
            "protocol_id": tree.protocol_id,
            "status": tree.status,
            "root_node_id": tree.root_node_id,
            "selected_node_id": tree.selected_node_id,
            "max_depth": tree.max_depth,
            "max_nodes": tree.max_nodes,
            "max_children_per_node": tree.max_children_per_node,
            "budget_id": tree.budget_id,
            "node_count": len(nodes),
            "is_terminal": is_tree_terminal(tree.status),
            "stop_reason": tree.stop_reason,
            "no_improvement_rounds": tree.no_improvement_rounds,
            "best_score_seen": tree.best_score_seen,
            "created_at": tree.created_at.isoformat()
            if isinstance(tree.created_at, datetime)
            else tree.created_at,
            "updated_at": tree.updated_at.isoformat()
            if isinstance(tree.updated_at, datetime)
            else tree.updated_at,
        }

    def show_tree(self, tree_id: str) -> dict[str, Any]:
        status = self.tree_status(tree_id)
        nodes = self._repo.list_nodes(tree_id)
        status["nodes"] = [n.model_dump(mode="json") for n in nodes]
        status["ascii_tree"] = self.render_ascii(tree_id)
        return status

    def render_ascii(self, tree_id: str) -> str:
        tree = self.require_tree(tree_id)
        nodes = self._repo.list_nodes(tree_id)
        by_parent: dict[str | None, list[TreeNode]] = {}
        root: TreeNode | None = None
        for node in nodes:
            by_parent.setdefault(node.parent_tree_node_id, []).append(node)
            if node.parent_tree_node_id is None:
                root = node
        if root is None:
            return f"(empty tree {tree.tree_id})"

        lines: list[str] = []

        def _label(node: TreeNode) -> str:
            parts = [f"{node.experiment_node_id} {node.node_type} [{node.status}"]
            if node.score is not None:
                parts.append(f", score={node.score:.2f}")
            if node.expansion_priority is not None:
                parts.append(f", expand={node.expansion_priority:.2f}")
            parts.append("]")
            return "".join(parts)

        def _walk(node: TreeNode, prefix: str, is_last: bool, is_root: bool) -> None:
            if is_root:
                lines.append(f"root: {_label(node)}")
            else:
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{_label(node)}")
            children = by_parent.get(node.tree_node_id, [])
            child_prefix = "" if is_root else prefix + ("    " if is_last else "│   ")
            for idx, child in enumerate(children):
                _walk(child, child_prefix, idx == len(children) - 1, False)

        _walk(root, "", True, True)
        return "\n".join(lines)

    def export_tree(self, tree_id: str, *, format: str = "json") -> dict[str, Any]:
        from scientist_lab.search.export import export_tree_payload

        tree = self.require_tree(tree_id)
        nodes = self._repo.list_nodes(tree_id)
        return export_tree_payload(
            tree,
            nodes,
            ascii_tree=self.render_ascii(tree_id),
            format=format,
        )

    def score_node(self, tree_id: str, tree_node_id: str) -> dict[str, Any]:
        tree = self.require_tree(tree_id)
        node = self._repo.get_node(tree_node_id)
        if node is None or node.tree_id != tree_id:
            raise KeyError(f"tree node not found: {tree_node_id}")
        breakdown = self._score_tree_node(tree, node)
        updated = node.model_copy(
            update={
                "score": breakdown.node_score,
                "expansion_priority": breakdown.expansion_priority,
                "updated_at": _utc_now(),
            }
        )
        self._repo.upsert_node(updated)
        return breakdown.model_dump(mode="json")

    def score_tree(self, tree_id: str) -> dict[str, Any]:
        tree = self.require_tree(tree_id)
        nodes = self._repo.list_nodes(tree_id)
        scored: list[dict[str, Any]] = []
        for node in nodes:
            breakdown = self._score_tree_node(tree, node)
            updated = node.model_copy(
                update={
                    "score": breakdown.node_score,
                    "expansion_priority": breakdown.expansion_priority,
                    "updated_at": _utc_now(),
                }
            )
            self._repo.upsert_node(updated)
            scored.append(breakdown.model_dump(mode="json"))

        ranked = sorted(
            scored,
            key=lambda item: float(item.get("expansion_priority") or 0.0),
            reverse=True,
        )
        return {
            "tree_id": tree_id,
            "project_id": tree.project_id,
            "protocol_id": tree.protocol_id,
            "scored_count": len(scored),
            "scores": scored,
            "ranking_by_expansion_priority": [
                {
                    "rank": idx + 1,
                    "tree_node_id": item.get("tree_node_id"),
                    "experiment_node_id": item.get("experiment_node_id"),
                    "node_type": item.get("node_type"),
                    "node_score": item.get("node_score"),
                    "expansion_priority": item.get("expansion_priority"),
                }
                for idx, item in enumerate(ranked)
            ],
            "ascii_tree": self.render_ascii(tree_id),
        }

    def select_parent(
        self, tree_id: str, *, rescore: bool = True
    ) -> dict[str, Any]:
        """Best-First: pick the expandable node with highest expansion_priority."""
        from scientist_lab.search.selection_policy import (
            rank_expandable_parents,
            selection_payload,
        )

        tree = self.require_tree(tree_id)
        if rescore:
            self.score_tree(tree_id)
            tree = self.require_tree(tree_id)

        nodes = self._repo.list_nodes(tree_id)
        result = rank_expandable_parents(tree, nodes)
        payload = selection_payload(result)

        if result.selected is not None:
            tree = tree.model_copy(
                update={
                    "selected_node_id": result.selected.experiment_node_id,
                    "updated_at": _utc_now(),
                }
            )
            # Activate tree when first selecting a parent.
            if tree.status == "created":
                from scientist_lab.search.state_machine import assert_tree_transition

                assert_tree_transition(tree.status, "active")
                tree = tree.model_copy(
                    update={"status": "active", "updated_at": _utc_now()}
                )
            self._repo.upsert_tree(tree)
            payload["tree_status"] = tree.status
            payload["selected_node_id"] = tree.selected_node_id
        else:
            payload["tree_status"] = tree.status
            payload["selected_node_id"] = tree.selected_node_id

        payload["ascii_tree"] = self.render_ascii(tree_id)
        return payload

    def apply_stop(
        self,
        tree_id: str,
        *,
        status: str,
        reason: str,
    ) -> ExperimentTree:
        """Force a terminal stop status on the tree."""
        from scientist_lab.search.state_machine import (
            assert_tree_transition,
            is_tree_terminal,
        )

        tree = self.require_tree(tree_id)
        if is_tree_terminal(tree.status):
            tree = tree.model_copy(
                update={
                    "stop_reason": reason or tree.stop_reason,
                    "updated_at": _utc_now(),
                }
            )
            return self._repo.upsert_tree(tree)
        try:
            assert_tree_transition(tree.status, status)
        except Exception:
            # Controlled stop: allow terminal assignment from any non-terminal state.
            pass
        tree = tree.model_copy(
            update={
                "status": status,  # type: ignore[arg-type]
                "stop_reason": reason,
                "updated_at": _utc_now(),
            }
        )
        return self._repo.upsert_tree(tree)

    def stop_tree(self, tree_id: str, *, reason: str) -> dict[str, Any]:
        tree = self.apply_stop(tree_id, status="user_stopped", reason=reason)
        return {
            "tree_id": tree.tree_id,
            "status": tree.status,
            "stop_reason": tree.stop_reason,
            "is_terminal": True,
            "ascii_tree": self.render_ascii(tree_id),
        }

    def _score_tree_node(self, tree: ExperimentTree, node: TreeNode) -> ScoreBreakdown:
        aggregate: dict[str, Any] | None = None
        if self._get_node_aggregate is not None:
            aggregate = self._get_node_aggregate(node.experiment_node_id)

        evidence: list[dict[str, Any]] = []
        if self._list_evidence is not None:
            evidence = list(self._list_evidence(tree.project_id) or [])

        claim_matrix: dict[str, Any] | None = None
        if self._get_claim_matrix is not None:
            claim_matrix = self._get_claim_matrix(tree.project_id)

        remaining: dict[str, Any] | None = None
        if self._get_remaining_budget is not None:
            budget_key = tree.budget_id or tree.project_id
            remaining = self._get_remaining_budget(budget_key)

        has_protocol = bool(tree.protocol_id)
        protocol_valid = True
        contract_match = True
        if self._get_node is not None and self._get_protocol is not None:
            exp_node = self._get_node(node.experiment_node_id)
            protocol = self._get_protocol(tree.protocol_id)
            if protocol is None:
                has_protocol = False
                protocol_valid = False
            elif exp_node is not None:
                contract_raw = getattr(exp_node, "contract_json", None) or {}
                if isinstance(contract_raw, dict) and contract_raw:
                    try:
                        contract = ExperimentContract.model_validate(contract_raw)
                        report = self._protocol_verifier.verify_contract(
                            protocol, contract
                        )
                        protocol_valid = bool(report.valid)
                        node_protocol = (contract.protocol_id or "").strip()
                        if node_protocol and node_protocol != tree.protocol_id:
                            contract_match = False
                    except Exception:  # noqa: BLE001
                        protocol_valid = False

        reference_mean = None
        if node.parent_tree_node_id:
            parent = self._repo.get_node(node.parent_tree_node_id)
            if parent is not None and self._get_node_aggregate is not None:
                parent_agg = self._get_node_aggregate(parent.experiment_node_id) or {}
                primary = parent_agg.get("primary_metric")
                metrics = parent_agg.get("aggregate_metrics") or {}
                if primary and primary in metrics:
                    mean = metrics[primary].get("mean")
                    if isinstance(mean, (int, float)):
                        reference_mean = float(mean)

        parent_failed_streak = 0
        cursor = node
        while cursor.parent_tree_node_id:
            parent = self._repo.get_node(cursor.parent_tree_node_id)
            if parent is None:
                break
            if parent.status == "failed":
                parent_failed_streak += 1
                cursor = parent
                continue
            break

        return compute_scores(
            experiment_node_id=node.experiment_node_id,
            tree_node_id=node.tree_node_id,
            node_type=node.node_type,
            status=node.status,
            aggregate=aggregate,
            evidence_records=evidence,
            claim_matrix=claim_matrix,
            remaining_budget=remaining,
            has_protocol=has_protocol,
            protocol_valid=protocol_valid,
            contract_protocol_match=contract_match,
            reference_mean=reference_mean,
            parent_failed_streak=parent_failed_streak,
        )
