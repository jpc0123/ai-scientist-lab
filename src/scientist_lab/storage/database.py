from __future__ import annotations

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class ResearchProjectRow(Base):
    __tablename__ = "research_projects"

    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    research_goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class ExperimentNodeRow(Base):
    __tablename__ = "experiment_nodes"

    node_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("research_projects.project_id"), nullable=False
    )
    parent_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    node_type: Mapped[str] = mapped_column(String, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    contract_json: Mapped[str] = mapped_column(Text, nullable=False)
    feedback_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class ExecutionAttemptRow(Base):
    __tablename__ = "execution_attempts"

    execution_id: Mapped[str] = mapped_column(String, primary_key=True)
    node_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiment_nodes.node_id"), nullable=False
    )
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False)
    runner_profile: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    container_id: Mapped[str | None] = mapped_column(String, nullable=True)
    image_reference: Mapped[str] = mapped_column(String, nullable=False)
    code_version: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class ExperimentArtifactRow(Base):
    __tablename__ = "experiment_artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)
    execution_id: Mapped[str] = mapped_column(
        String, ForeignKey("execution_attempts.execution_id"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class IterationSessionRow(Base):
    __tablename__ = "iteration_sessions"
    __table_args__ = (
        Index("idx_iteration_project", "project_id"),
        Index("idx_iteration_status", "status"),
    )

    iteration_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    source_baseline_node_id: Mapped[str] = mapped_column(String, nullable=False)
    source_candidate_node_id: Mapped[str] = mapped_column(String, nullable=False)
    proposed_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    seeds_json: Mapped[str] = mapped_column(Text, nullable=False)
    feedback_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    proposal_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    proposal_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    execution_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    comparison_paths_json: Mapped[str] = mapped_column(Text, nullable=False)
    selected_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    decision_id: Mapped[str | None] = mapped_column(String, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class ApprovalRecordRow(Base):
    __tablename__ = "approval_records"

    approval_id: Mapped[str] = mapped_column(String, primary_key=True)
    iteration_id: Mapped[str] = mapped_column(
        String, ForeignKey("iteration_sessions.iteration_id"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    contract_path: Mapped[str] = mapped_column(Text, nullable=False)
    contract_sha256: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class DatasetRegistrationRow(Base):
    """Logical table name in docs: datasets. Physical name kept for compatibility."""

    __tablename__ = "dataset_registrations"
    __table_args__ = (Index("idx_datasets_task_type", "task_type"),)

    dataset_key: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    host_path: Mapped[str] = mapped_column(Text, nullable=False)
    container_path: Mapped[str] = mapped_column(Text, nullable=False)
    read_only: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def make_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", future=True)


def init_db(db_path: str) -> sessionmaker:
    from scientist_lab.ablations.repository import ensure_ablation_schema
    from scientist_lab.agents.repository import ensure_agent_plan_schema
    from scientist_lab.budget.repository import ensure_budget_schema
    from scientist_lab.checkpoints.repository import ensure_checkpoint_schema
    from scientist_lab.datasets.repository import ensure_dataset_schema
    from scientist_lab.evidence.repository import ensure_evidence_schema
    from scientist_lab.llm_eval.repository import ensure_llm_eval_schema
    from scientist_lab.protocols.repository import ensure_protocol_schema
    from scientist_lab.runners.profile_repository import ensure_runner_profile_schema
    from scientist_lab.search.repository import ensure_search_tree_schema

    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    ensure_dataset_schema(engine)
    ensure_runner_profile_schema(engine)
    ensure_checkpoint_schema(engine)
    ensure_protocol_schema(engine)
    ensure_ablation_schema(engine)
    ensure_evidence_schema(engine)
    ensure_agent_plan_schema(engine)
    ensure_budget_schema(engine)
    ensure_search_tree_schema(engine)
    ensure_llm_eval_schema(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
