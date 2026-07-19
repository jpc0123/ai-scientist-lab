from __future__ import annotations

from abc import ABC, abstractmethod

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.domain.results import (
    ExecutionResult,
    ExecutionStatus,
    SubmissionResult,
)


class ExperimentRunner(ABC):
    @abstractmethod
    def validate(self, contract: ExperimentContract) -> None:
        ...

    @abstractmethod
    def submit(self, contract: ExperimentContract) -> SubmissionResult:
        ...

    @abstractmethod
    def get_status(self, execution_id: str) -> ExecutionStatus:
        ...

    @abstractmethod
    def collect_result(self, execution_id: str) -> ExecutionResult:
        ...

    @abstractmethod
    def cancel(self, execution_id: str) -> None:
        ...
