"""Project lifecycle helpers (v2.0.1)."""

from __future__ import annotations

from scientist_lab.domain import ProjectStatus

ALLOWED_TRANSITIONS: dict[ProjectStatus, set[ProjectStatus]] = {
    ProjectStatus.DRAFT: {ProjectStatus.CONFIGURING, ProjectStatus.READY, ProjectStatus.ARCHIVED},
    ProjectStatus.CONFIGURING: {
        ProjectStatus.READY,
        ProjectStatus.DRAFT,
        ProjectStatus.ARCHIVED,
    },
    ProjectStatus.READY: {
        ProjectStatus.RUNNING,
        ProjectStatus.CONFIGURING,
        ProjectStatus.ARCHIVED,
    },
    ProjectStatus.RUNNING: {
        ProjectStatus.REVIEWING,
        ProjectStatus.READY,
        ProjectStatus.ARCHIVED,
    },
    ProjectStatus.REVIEWING: {
        ProjectStatus.COMPLETED,
        ProjectStatus.RUNNING,
        ProjectStatus.READY,
        ProjectStatus.ARCHIVED,
    },
    ProjectStatus.COMPLETED: {ProjectStatus.ARCHIVED, ProjectStatus.READY},
    ProjectStatus.ARCHIVED: set(),
    # Legacy: treat like ready for outgoing transitions
    ProjectStatus.ACTIVE: {
        ProjectStatus.RUNNING,
        ProjectStatus.CONFIGURING,
        ProjectStatus.ARCHIVED,
        ProjectStatus.READY,
    },
}

TASK_TYPES = frozenset({"general_ml", "rgbt_detection", "custom_registered_task"})


def normalize_status(status: ProjectStatus | str) -> ProjectStatus:
    value = ProjectStatus(str(status))
    if value == ProjectStatus.ACTIVE:
        return ProjectStatus.READY
    return value


def assert_transition(current: ProjectStatus, target: ProjectStatus) -> None:
    cur = normalize_status(current) if current == ProjectStatus.ACTIVE else current
    # Allow identity
    if cur == target or (
        current == ProjectStatus.ACTIVE and target == ProjectStatus.READY
    ):
        return
    allowed = ALLOWED_TRANSITIONS.get(current) or ALLOWED_TRANSITIONS.get(cur) or set()
    if target not in allowed:
        raise ValueError(
            f"illegal project status transition: {current.value!r} → {target.value!r}"
        )


def validate_task_type(task_type: str) -> str:
    cleaned = (task_type or "").strip() or "general_ml"
    if cleaned not in TASK_TYPES:
        raise ValueError(
            f"unsupported task_type {cleaned!r}; "
            f"allowed={sorted(TASK_TYPES)}"
        )
    return cleaned
