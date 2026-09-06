"""Replaceable HOW plugin author worker.

Produces a Unified Diff for how_plugins/<HOW>/plugin.py only.
Does not register. Does not start GPU.
Harness/dsh/docker resolve to HarnessPluginWorker (Linux container).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from scientist_lab.patching.context_models import CodeContextBundle
from scientist_lab.patching.real_patch_planner import RealPatchPlanner, RealPatchPlannerError


class PluginAuthorWorkerError(ValueError):
    """Worker refused to emit a plugin Diff. Main tree is unchanged."""


@dataclass(frozen=True)
class PluginAuthorDraft:
    unified_diff: str
    worker_id: str
    authored_by: str
    notes: dict[str, Any] = field(default_factory=dict)


class PluginAuthorWorker(Protocol):
    worker_id: str
    authored_by: str

    def propose_plugin_diff(
        self,
        bundle: CodeContextBundle,
        *,
        how_id: str,
        system_prompt: str,
    ) -> PluginAuthorDraft:
        """Return a Unified Diff. Caller still runs PathPolicy, sandbox, smoke."""


class FakePluginWorker:
    """Deterministic stand-in: copy the checked-in example plugin. No network."""

    worker_id = "fake"
    authored_by = "fake"

    def __init__(
        self,
        *,
        source: str | None = None,
        project_root: Any | None = None,
        unified_diff: str | None = None,
        plugin_kind: str = "fusion",
    ) -> None:
        self._source = source
        self._project_root = project_root
        self._unified_diff = str(unified_diff or "").strip() or None
        self._plugin_kind = str(plugin_kind or "fusion").strip().lower() or "fusion"

    def propose_plugin_diff(
        self,
        bundle: CodeContextBundle,
        *,
        how_id: str,
        system_prompt: str,
    ) -> PluginAuthorDraft:
        del bundle, system_prompt
        if self._unified_diff is not None:
            diff = self._unified_diff
        else:
            from scientist_lab.core.how_plugin_author import (
                example_plugin_source,
                plugin_unified_diff_from_source,
            )

            body = self._source
            if body is None:
                body = example_plugin_source(
                    self._project_root, plugin_kind=self._plugin_kind
                )
            diff = plugin_unified_diff_from_source(how_id, body)
        if not str(diff or "").strip():
            raise PluginAuthorWorkerError("fake worker produced an empty Diff")
        return PluginAuthorDraft(
            unified_diff=diff,
            worker_id=self.worker_id,
            authored_by=self.authored_by,
            notes={"gpu": False, "registered": False, "network": False},
        )


class LlmPatchPluginWorker:
    """Existing RealPatchPlanner wrapped as a PluginAuthorWorker."""

    worker_id = "llm"
    authored_by = "llm"

    def __init__(self, provider: Any, *, real_only: bool = False) -> None:
        self._provider = provider
        self._real_only = real_only

    def propose_plugin_diff(
        self,
        bundle: CodeContextBundle,
        *,
        how_id: str,
        system_prompt: str,
    ) -> PluginAuthorDraft:
        del how_id
        planner = RealPatchPlanner(
            self._provider,
            requested_provider=str(
                getattr(self._provider, "name", None) or "openai-compatible"
            ),
            real_only=self._real_only,
        )
        try:
            proposal, audit = planner.propose(bundle, system_prompt=system_prompt)
        except RealPatchPlannerError as exc:
            raise PluginAuthorWorkerError(str(exc)) from exc
        diff = str(proposal.unified_diff or "")
        if not diff.strip():
            raise PluginAuthorWorkerError("llm worker produced an empty Diff")
        return PluginAuthorDraft(
            unified_diff=diff,
            worker_id=self.worker_id,
            authored_by=self.authored_by,
            notes={"audit": dict(audit) if isinstance(audit, dict) else {}, "gpu": False, "registered": False},
        )


def resolve_plugin_worker(
    name: str | None,
    *,
    provider: Any | None = None,
    project_root: Any | None = None,
    work_root: Any | None = None,
    runner: Any | None = None,
) -> PluginAuthorWorker:
    token = str(name or "llm").strip().lower()
    if token == "fake":
        return FakePluginWorker(project_root=project_root)
    if token in {"harness", "dsh", "docker"}:
        from scientist_lab.core.how_plugin_harness import HarnessPluginWorker

        if project_root is None:
            raise PluginAuthorWorkerError("harness worker needs project_root")
        return HarnessPluginWorker(
            project_root=project_root,
            runner=runner,
            work_root=work_root,
        )
    if token == "llm":
        if provider is None:
            raise PluginAuthorWorkerError("llm worker needs a provider")
        return LlmPatchPluginWorker(provider)
    raise PluginAuthorWorkerError(f"unknown plugin worker: {name}")
