from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.search.models import ExperimentTree, TreeNode
from scientist_lab.storage.database import Base


class ExperimentTreeRow(Base):
    __tablename__ = "experiment_trees"

    tree_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    protocol_id: Mapped[str] = mapped_column(String, nullable=False)
    root_node_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    max_depth: Mapped[int] = mapped_column(Integer, nullable=False)
    max_nodes: Mapped[int] = mapped_column(Integer, nullable=False)
    max_children_per_node: Mapped[int] = mapped_column(Integer, nullable=False)

    budget_id: Mapped[str | None] = mapped_column(String, nullable=True)
    selected_node_id: Mapped[str | None] = mapped_column(String, nullable=True)

    stop_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    no_improvement_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_score_seen: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_no_improvement_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    minimum_score_improvement: Mapped[float] = mapped_column(Float, nullable=False, default=0.01)

    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class ExperimentTreeNodeRow(Base):
    __tablename__ = "experiment_tree_nodes"
    __table_args__ = (
        UniqueConstraint(
            "tree_id", "experiment_node_id", name="idx_tree_experiment_node"
        ),
        Index("idx_tree_nodes_tree", "tree_id"),
        Index("idx_tree_nodes_parent", "parent_tree_node_id"),
    )

    tree_node_id: Mapped[str] = mapped_column(String, primary_key=True)
    tree_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiment_trees.tree_id"), nullable=False
    )
    experiment_node_id: Mapped[str] = mapped_column(String, nullable=False)

    parent_tree_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    node_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    expansion_priority: Mapped[float | None] = mapped_column(Float, nullable=True)

    plan_id: Mapped[str | None] = mapped_column(String, nullable=True)
    candidate_id: Mapped[str | None] = mapped_column(String, nullable=True)
    iteration_id: Mapped[str | None] = mapped_column(String, nullable=True)
    decision_id: Mapped[str | None] = mapped_column(String, nullable=True)

    evidence_ids_json: Mapped[str | None] = mapped_column(String, nullable=True)
    resolved_evidence_gaps_json: Mapped[str | None] = mapped_column(String, nullable=True)
    new_evidence_gaps_json: Mapped[str | None] = mapped_column(String, nullable=True)
    claim_matrix_path: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_search_tree_schema(engine) -> None:
    Base.metadata.create_all(
        engine,
        tables=[ExperimentTreeRow.__table__, ExperimentTreeNodeRow.__table__],
    )
    _migrate_legacy_search_tree_schema(engine)


def _migrate_legacy_search_tree_schema(engine) -> None:
    """Migrate draft schemas and add stop/evidence columns when missing."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "experiment_trees" not in inspector.get_table_names():
        return

    tree_cols = {c["name"] for c in inspector.get_columns("experiment_trees")}
    node_cols = (
        {c["name"] for c in inspector.get_columns("experiment_tree_nodes")}
        if "experiment_tree_nodes" in inspector.get_table_names()
        else set()
    )
    # Old draft column that is NOT part of the current model.
    if "prune_reason" in node_cols:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS experiment_tree_nodes"))
            conn.execute(text("DROP TABLE IF EXISTS experiment_trees"))
        ExperimentTreeRow.__table__.create(bind=engine, checkfirst=True)
        ExperimentTreeNodeRow.__table__.create(bind=engine, checkfirst=True)
        return

    alters: list[str] = []
    if "stop_reason" not in tree_cols:
        alters.append("ALTER TABLE experiment_trees ADD COLUMN stop_reason TEXT")
    if "no_improvement_rounds" not in tree_cols:
        alters.append(
            "ALTER TABLE experiment_trees ADD COLUMN no_improvement_rounds INTEGER NOT NULL DEFAULT 0"
        )
    if "best_score_seen" not in tree_cols:
        alters.append("ALTER TABLE experiment_trees ADD COLUMN best_score_seen REAL")
    if "max_no_improvement_rounds" not in tree_cols:
        alters.append(
            "ALTER TABLE experiment_trees ADD COLUMN max_no_improvement_rounds INTEGER NOT NULL DEFAULT 2"
        )
    if "minimum_score_improvement" not in tree_cols:
        alters.append(
            "ALTER TABLE experiment_trees ADD COLUMN minimum_score_improvement REAL NOT NULL DEFAULT 0.01"
        )
    if "evidence_ids_json" not in node_cols:
        alters.append(
            "ALTER TABLE experiment_tree_nodes ADD COLUMN evidence_ids_json TEXT"
        )
    if "resolved_evidence_gaps_json" not in node_cols:
        alters.append(
            "ALTER TABLE experiment_tree_nodes ADD COLUMN resolved_evidence_gaps_json TEXT"
        )
    if "new_evidence_gaps_json" not in node_cols:
        alters.append(
            "ALTER TABLE experiment_tree_nodes ADD COLUMN new_evidence_gaps_json TEXT"
        )
    if "claim_matrix_path" not in node_cols:
        alters.append(
            "ALTER TABLE experiment_tree_nodes ADD COLUMN claim_matrix_path TEXT"
        )
    if not alters:
        return
    with engine.begin() as conn:
        for stmt in alters:
            conn.execute(text(stmt))


def _dt_to_str(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    return str(value)


def _list_to_json(values: list[str] | None) -> str | None:
    items = [str(v) for v in (values or [])]
    if not items:
        return None
    return json.dumps(items, ensure_ascii=False)


def _json_to_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


class TreeRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert_tree(self, tree: ExperimentTree) -> ExperimentTree:
        with self._session_factory() as session:
            row = session.get(ExperimentTreeRow, tree.tree_id)
            payload = self._tree_to_row(tree)
            if row is None:
                session.add(ExperimentTreeRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return tree

    def create_tree(self, tree: ExperimentTree) -> ExperimentTree:
        with self._session_factory() as session:
            if session.get(ExperimentTreeRow, tree.tree_id) is not None:
                raise ValueError(f"duplicate tree_id: {tree.tree_id}")
            session.add(ExperimentTreeRow(**self._tree_to_row(tree)))
            session.commit()
        return tree

    def get_tree(self, tree_id: str) -> ExperimentTree | None:
        with self._session_factory() as session:
            row = session.get(ExperimentTreeRow, tree_id)
            return self._tree_from_row(row) if row else None

    def list_trees(self, *, project_id: str | None = None) -> list[ExperimentTree]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = select(ExperimentTreeRow)
            if project_id:
                stmt = stmt.where(ExperimentTreeRow.project_id == project_id)
            stmt = stmt.order_by(
                ExperimentTreeRow.created_at.desc(), ExperimentTreeRow.tree_id
            )
            return [self._tree_from_row(row) for row in session.scalars(stmt).all()]

    def upsert_node(self, node: TreeNode) -> TreeNode:
        with self._session_factory() as session:
            row = session.get(ExperimentTreeNodeRow, node.tree_node_id)
            payload = self._node_to_row(node)
            if row is None:
                session.add(ExperimentTreeNodeRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return node

    def create_node(self, node: TreeNode) -> TreeNode:
        existing = self.get_node_by_experiment(node.tree_id, node.experiment_node_id)
        if existing is not None:
            raise ValueError(
                f"experiment node already in tree: {node.experiment_node_id}"
            )
        with self._session_factory() as session:
            if session.get(ExperimentTreeNodeRow, node.tree_node_id) is not None:
                raise ValueError(f"duplicate tree_node_id: {node.tree_node_id}")
            session.add(ExperimentTreeNodeRow(**self._node_to_row(node)))
            session.commit()
        return node

    def get_node(self, tree_node_id: str) -> TreeNode | None:
        with self._session_factory() as session:
            row = session.get(ExperimentTreeNodeRow, tree_node_id)
            return self._node_from_row(row) if row else None

    def get_node_by_experiment(
        self, tree_id: str, experiment_node_id: str
    ) -> TreeNode | None:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = select(ExperimentTreeNodeRow).where(
                ExperimentTreeNodeRow.tree_id == tree_id,
                ExperimentTreeNodeRow.experiment_node_id == experiment_node_id,
            )
            row = session.scalars(stmt).first()
            return self._node_from_row(row) if row else None

    def list_nodes(self, tree_id: str) -> list[TreeNode]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = (
                select(ExperimentTreeNodeRow)
                .where(ExperimentTreeNodeRow.tree_id == tree_id)
                .order_by(
                    ExperimentTreeNodeRow.depth.asc(),
                    ExperimentTreeNodeRow.created_at.asc(),
                    ExperimentTreeNodeRow.tree_node_id.asc(),
                )
            )
            return [self._node_from_row(row) for row in session.scalars(stmt).all()]

    def count_nodes(self, tree_id: str) -> int:
        from sqlalchemy import func, select

        with self._session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(ExperimentTreeNodeRow)
                .where(ExperimentTreeNodeRow.tree_id == tree_id)
            )
            return int(session.scalar(stmt) or 0)

    @staticmethod
    def _tree_to_row(tree: ExperimentTree) -> dict[str, Any]:
        return {
            "tree_id": tree.tree_id,
            "project_id": tree.project_id,
            "protocol_id": tree.protocol_id,
            "root_node_id": tree.root_node_id,
            "status": tree.status,
            "max_depth": tree.max_depth,
            "max_nodes": tree.max_nodes,
            "max_children_per_node": tree.max_children_per_node,
            "budget_id": tree.budget_id,
            "selected_node_id": tree.selected_node_id,
            "stop_reason": tree.stop_reason,
            "no_improvement_rounds": tree.no_improvement_rounds,
            "best_score_seen": tree.best_score_seen,
            "max_no_improvement_rounds": tree.max_no_improvement_rounds,
            "minimum_score_improvement": tree.minimum_score_improvement,
            "created_at": _dt_to_str(tree.created_at),
            "updated_at": _dt_to_str(tree.updated_at),
        }

    @staticmethod
    def _tree_from_row(row: ExperimentTreeRow) -> ExperimentTree:
        return ExperimentTree.model_validate(
            {
                "tree_id": row.tree_id,
                "project_id": row.project_id,
                "protocol_id": row.protocol_id,
                "root_node_id": row.root_node_id,
                "status": row.status,
                "max_depth": row.max_depth,
                "max_nodes": row.max_nodes,
                "max_children_per_node": row.max_children_per_node,
                "budget_id": row.budget_id,
                "selected_node_id": row.selected_node_id,
                "stop_reason": getattr(row, "stop_reason", None),
                "no_improvement_rounds": getattr(row, "no_improvement_rounds", 0) or 0,
                "best_score_seen": getattr(row, "best_score_seen", None),
                "max_no_improvement_rounds": getattr(
                    row, "max_no_improvement_rounds", 2
                )
                or 2,
                "minimum_score_improvement": getattr(
                    row, "minimum_score_improvement", 0.01
                )
                or 0.01,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
        )

    @staticmethod
    def _node_to_row(node: TreeNode) -> dict[str, Any]:
        return {
            "tree_node_id": node.tree_node_id,
            "tree_id": node.tree_id,
            "experiment_node_id": node.experiment_node_id,
            "parent_tree_node_id": node.parent_tree_node_id,
            "depth": node.depth,
            "node_type": node.node_type,
            "status": node.status,
            "score": node.score,
            "expansion_priority": node.expansion_priority,
            "plan_id": node.plan_id,
            "candidate_id": node.candidate_id,
            "iteration_id": node.iteration_id,
            "decision_id": node.decision_id,
            "evidence_ids_json": _list_to_json(node.evidence_ids),
            "resolved_evidence_gaps_json": _list_to_json(node.resolved_evidence_gaps),
            "new_evidence_gaps_json": _list_to_json(node.new_evidence_gaps),
            "claim_matrix_path": node.claim_matrix_path,
            "created_at": _dt_to_str(node.created_at),
            "updated_at": _dt_to_str(node.updated_at),
        }

    @staticmethod
    def _node_from_row(row: ExperimentTreeNodeRow) -> TreeNode:
        return TreeNode.model_validate(
            {
                "tree_node_id": row.tree_node_id,
                "tree_id": row.tree_id,
                "experiment_node_id": row.experiment_node_id,
                "parent_tree_node_id": row.parent_tree_node_id,
                "depth": row.depth,
                "node_type": row.node_type,
                "status": row.status,
                "score": row.score,
                "expansion_priority": row.expansion_priority,
                "plan_id": row.plan_id,
                "candidate_id": row.candidate_id,
                "iteration_id": row.iteration_id,
                "decision_id": row.decision_id,
                "evidence_ids": _json_to_list(
                    getattr(row, "evidence_ids_json", None)
                ),
                "resolved_evidence_gaps": _json_to_list(
                    getattr(row, "resolved_evidence_gaps_json", None)
                ),
                "new_evidence_gaps": _json_to_list(
                    getattr(row, "new_evidence_gaps_json", None)
                ),
                "claim_matrix_path": getattr(row, "claim_matrix_path", None),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
        )
