from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from scientist_lab.budget.models import ExperimentBudget
from scientist_lab.budget.repository import BudgetRepository


class BudgetService:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._repo = BudgetRepository(session_factory)

    def get(self, project_id: str) -> ExperimentBudget | None:
        return self._repo.get(project_id)

    def require(self, project_id: str) -> ExperimentBudget:
        budget = self.get(project_id)
        if budget is None:
            raise KeyError(f"budget not set for project: {project_id}")
        return budget

    def set_budget(
        self,
        project_id: str,
        *,
        max_new_nodes: int = 3,
        max_total_executions: int = 15,
        max_gpu_hours: float = 10.0,
        max_storage_gb: float = 20.0,
        preserve_usage: bool = True,
    ) -> ExperimentBudget:
        existing = self.get(project_id)
        used = {
            "used_nodes": existing.used_nodes if existing and preserve_usage else 0,
            "used_executions": existing.used_executions if existing and preserve_usage else 0,
            "used_gpu_hours": existing.used_gpu_hours if existing and preserve_usage else 0.0,
            "used_storage_gb": existing.used_storage_gb if existing and preserve_usage else 0.0,
        }
        budget = ExperimentBudget(
            project_id=project_id,
            max_new_nodes=max_new_nodes,
            max_total_executions=max_total_executions,
            max_gpu_hours=max_gpu_hours,
            max_storage_gb=max_storage_gb,
            **used,
        )
        return self._repo.upsert(budget)

    def remaining(self, project_id: str) -> dict:
        budget = self.get(project_id)
        if budget is None:
            return ExperimentBudget(project_id=project_id).remaining()
        return budget.remaining()

    def consume(
        self,
        project_id: str,
        *,
        nodes: int = 0,
        executions: int = 0,
        gpu_hours: float = 0.0,
        storage_gb: float = 0.0,
    ) -> ExperimentBudget:
        budget = self.get(project_id) or ExperimentBudget(project_id=project_id)
        updated = budget.model_copy(
            update={
                "used_nodes": budget.used_nodes + int(nodes),
                "used_executions": budget.used_executions + int(executions),
                "used_gpu_hours": float(budget.used_gpu_hours) + float(gpu_hours),
                "used_storage_gb": float(budget.used_storage_gb) + float(storage_gb),
            }
        )
        return self._repo.upsert(updated)
