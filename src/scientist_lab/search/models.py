from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


TreeStatus = Literal[
    "created",
    "active",
    "waiting_approval",
    "running",
    "evaluating",
    "completed",
    "budget_exhausted",
    "max_depth_reached",
    "no_valid_candidates",
    "user_stopped",
    "failed",
]

TreeNodeType = Literal[
    "root",
    "improve",
    "ablation",
    "replication",
    "debug",
    "efficiency",
]

TreeNodeStatus = Literal[
    "created",
    "proposed",
    "waiting_approval",
    "running",
    "evaluated",
    "selected",
    "pruned",
    "failed",
    "stopped",
]

TREE_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "completed",
        "budget_exhausted",
        "max_depth_reached",
        "no_valid_candidates",
        "user_stopped",
        "failed",
    }
)


class ExperimentTree(BaseModel):
    tree_id: str
    project_id: str
    protocol_id: str
    root_node_id: str

    status: TreeStatus = "created"

    max_depth: int
    max_nodes: int
    max_children_per_node: int

    budget_id: str | None = None
    selected_node_id: str | None = None

    stop_reason: str | None = None
    no_improvement_rounds: int = 0
    best_score_seen: float | None = None
    max_no_improvement_rounds: int = 2
    minimum_score_improvement: float = 0.01

    created_at: datetime
    updated_at: datetime

    @field_validator("tree_id", "project_id", "protocol_id", "root_node_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @field_validator("max_depth")
    @classmethod
    def _depth_ok(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("max_depth must be > 0")
        return int(value)

    @field_validator("max_nodes")
    @classmethod
    def _nodes_ok(cls, value: int) -> int:
        if int(value) <= 1:
            raise ValueError("max_nodes must be > 1")
        return int(value)

    @field_validator("max_children_per_node")
    @classmethod
    def _children_ok(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("max_children_per_node must be > 0")
        return int(value)

    @model_validator(mode="after")
    def _nodes_vs_children(self) -> ExperimentTree:
        if self.max_nodes < self.max_children_per_node:
            raise ValueError("max_nodes must be >= max_children_per_node")
        return self

    def is_terminal(self) -> bool:
        return self.status in TREE_TERMINAL_STATUSES


class TreeNode(BaseModel):
    tree_node_id: str
    tree_id: str
    experiment_node_id: str

    parent_tree_node_id: str | None = None
    depth: int = 0

    node_type: TreeNodeType = "root"
    status: TreeNodeStatus = "created"

    score: float | None = None
    expansion_priority: float | None = None

    plan_id: str | None = None
    candidate_id: str | None = None
    iteration_id: str | None = None
    decision_id: str | None = None

    evidence_ids: list[str] = Field(default_factory=list)
    resolved_evidence_gaps: list[str] = Field(default_factory=list)
    new_evidence_gaps: list[str] = Field(default_factory=list)
    claim_matrix_path: str | None = None

    created_at: datetime
    updated_at: datetime

    @field_validator("tree_node_id", "tree_id", "experiment_node_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @field_validator("depth")
    @classmethod
    def _depth_nonneg(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("depth must be >= 0")
        return int(value)

    @field_validator(
        "evidence_ids",
        "resolved_evidence_gaps",
        "new_evidence_gaps",
        mode="before",
    )
    @classmethod
    def _coerce_str_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            # JSON list stored as text in SQLite.
            if text.startswith("["):
                import json

                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except Exception:  # noqa: BLE001
                    return [text]
            return [text]
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
