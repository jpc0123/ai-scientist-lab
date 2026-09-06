"""Docker-backed DeepSeek Harness worker for HOW plugin.py only.

Stages an isolated checkout (not the lab tree). Collects only plugin.py.
Does not register. Does not start GPU. Does not merge the main tree.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from scientist_lab.core.how_plugin_worker import (
    PluginAuthorDraft,
    PluginAuthorWorkerError,
)
from scientist_lab.patching.context_models import CodeContextBundle

DEFAULT_DSH_IMAGE = "scientist-lab/dsh-plugin-author:dev"
PLUGIN_REL_PREFIX = "experiment_apps/rgbt_detection_real/models/how_plugins"
APP_MODELS = Path("experiment_apps") / "rgbt_detection_real" / "models"
PLUGIN_STUB = (
    "# Replace this stub with a HOW plugin.\n"
    "# Export build_fusion, build_neck, or build_backbone_wrap for PLUGIN_KIND.\n"
)


def _kind_from_prompt(system_prompt: str, goal: str) -> str:
    blob = f"{system_prompt}\n{goal}".lower()
    if "plugin_kind=neck" in blob or "build_neck" in blob:
        return "neck"
    if "plugin_kind=backbone_wrap" in blob or "build_backbone_wrap" in blob:
        return "backbone_wrap"
    return "fusion"


def _example_folder(kind: str) -> str:
    return {
        "fusion": "_example_weighted",
        "neck": "_example_neck",
        "backbone_wrap": "_example_backbone_wrap",
    }.get(kind, "_example_weighted")


def _plugin_has_builder(body: str) -> bool:
    text = str(body or "")
    return any(
        token in text
        for token in ("def build_fusion", "def build_neck", "def build_backbone_wrap")
    )


FORBIDDEN_STAGE_NAMES = frozenset(
    {
        "train_dfine.py",
        "fusion_factory.py",
        "how_plugin_loader.py",
    }
)


@dataclass(frozen=True)
class HarnessContainerRequest:
    workspace: Path
    plugin_relpath: str
    how_id: str
    prompt: str
    image: str
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessContainerResult:
    exit_code: int
    log_text: str
    image: str
    gpu: bool = False


class HarnessContainerRunner(Protocol):
    def run(self, request: HarnessContainerRequest) -> HarnessContainerResult:
        """Run one isolated authoring job. Must not touch the lab main tree."""


def connect_docker() -> Any:
    try:
        import docker
    except ImportError as exc:
        raise PluginAuthorWorkerError(
            "docker_unavailable: python docker package not installed"
        ) from exc
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # noqa: BLE001
        raise PluginAuthorWorkerError(
            f"docker_unavailable: cannot connect to Docker: {exc}"
        ) from exc
    return client


def plugin_relpath_for(how_id: str) -> str:
    token = str(how_id or "").strip().upper()
    if token.startswith("PLUGIN:"):
        token = token.split(":", 1)[1].strip().upper()
    return f"{PLUGIN_REL_PREFIX}/{token}/plugin.py"


def harness_plugin_task_prompt(
    *,
    how_id: str,
    rel: str,
    system_prompt: str,
    goal: str,
) -> str:
    """Tell the coding agent to write plugin.py. Do not ask for a JSON Diff."""
    token = str(how_id or "").strip().upper()
    kind = _kind_from_prompt(system_prompt, goal)
    example = f"{PLUGIN_REL_PREFIX}/{_example_folder(kind)}/plugin.py"
    if kind == "neck":
        contract = [
            "The module MUST define PLUGIN_KIND = \"neck\" and:",
            "def build_neck(in_channels, hidden_dim=256, feat_strides=(8,16,32), **kwargs) -> nn.Module.",
            "forward(feats) must keep HybridEncoder I/O: list×3 in, list×3 hidden_dim out.",
        ]
    elif kind == "backbone_wrap":
        contract = [
            "The module MUST define PLUGIN_KIND = \"backbone_wrap\" and:",
            "def build_backbone_wrap(rgb_backbone, **kwargs) -> nn.Module.",
            "forward(x) must return list×3 backbone feature maps.",
        ]
    else:
        contract = [
            "The module MUST define PLUGIN_KIND = \"fusion\" and exactly:",
            "def build_fusion(channels, residual=True) -> FeatureFusion.",
            "Do NOT name it build_feature_fusion, create_fusion, or build_plugin.",
            "build_fusion must return models.feature_fusion.FeatureFusion.",
            "forward(rgb_features, thermal_features) must keep NCHW per P3/P4/P5.",
        ]
    lines = [
        f"You are a restricted HOW {kind}-plugin author for Scientist Lab.",
        f"plugin_kind={kind}.",
        f"Write a complete Python module to {rel} using editor or bash.",
        "Overwrite that file. Do not return JSON. Do not emit a Unified Diff.",
        f"Copy the contract in {example} and adapt it for HOW {token}.",
        *contract,
        "Never touch third_party, train_dfine.py, fusion_factory.py, or scientist_lab.llm.",
        "Never emit subprocess, eval, or network calls in plugin.py.",
    ]
    goal_text = str(goal or "").strip()
    if goal_text:
        lines.extend(["", "Goal:", goal_text])
    _ = system_prompt  # Lab LLM patch prompt is JSON-Diff; do not forward it.
    return "\n".join(lines) + "\n"


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def stage_harness_workspace(
    *,
    project_root: Path,
    work_root: Path,
    how_id: str,
    prompt: str,
) -> Path:
    """Copy plugin contract + matching example only. No vendor, no trainer."""
    root = Path(project_root).resolve()
    models = root / APP_MODELS
    fusion = models / "feature_fusion.py"
    kind = _kind_from_prompt(prompt, "")
    example = models / "how_plugins" / _example_folder(kind) / "plugin.py"
    if not fusion.is_file():
        raise PluginAuthorWorkerError(f"missing FeatureFusion contract: {fusion}")
    if not example.is_file():
        raise PluginAuthorWorkerError(f"missing example plugin: {example}")
    workspace = Path(work_root).resolve() / f"ws_{str(how_id).upper()}"
    if workspace.exists():
        shutil.rmtree(workspace)
    rel = plugin_relpath_for(how_id)
    dest_plugin = workspace / rel
    dest_plugin.parent.mkdir(parents=True, exist_ok=True)
    dest_plugin.write_text(PLUGIN_STUB, encoding="utf-8")
    (workspace / APP_MODELS / "__init__.py").write_text(
        "# Isolated harness workspace. Not the lab models package.\n",
        encoding="utf-8",
    )
    (workspace / APP_MODELS / "how_plugins" / "__init__.py").write_text(
        "# Isolated HOW plugin dir.\n",
        encoding="utf-8",
    )
    _copy_file(fusion, workspace / APP_MODELS / "feature_fusion.py")
    _copy_file(
        example,
        workspace / APP_MODELS / "how_plugins" / _example_folder(kind) / "plugin.py",
    )
    (workspace / "PLUGIN_TASK.md").write_text(prompt.strip() + "\n", encoding="utf-8")
    for path in workspace.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_STAGE_NAMES:
            raise PluginAuthorWorkerError(f"refused to stage {path.name}")
    if (workspace / "third_party").exists():
        raise PluginAuthorWorkerError("refused to stage third_party")
    return workspace


def _harness_env(how_id: str, plugin_relpath: str) -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "HOW_ID": str(how_id).upper(),
        "HOW_PLUGIN_TARGET": f"/workspace/{plugin_relpath.replace(chr(92), '/')}",
        "DSH_SESSION_ROOT": "/sessions",
    }
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    if key:
        env["DEEPSEEK_API_KEY"] = key
    base = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("LLM_BASE_URL") or ""
    if base:
        env["DEEPSEEK_BASE_URL"] = base
    model = os.environ.get("DSH_MODEL") or os.environ.get("LLM_MODEL") or ""
    if model:
        env["DSH_MODEL"] = model
    # OpenAI-compatible lab keys (Aliyun Qwen) use the pi-ai "lab" route,
    # not deepseek-official. Explicit DSH_PROVIDER still wins.
    explicit = os.environ.get("DSH_PROVIDER") or ""
    llm_provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        env["DSH_PROVIDER"] = explicit
    elif llm_provider in {"openai-compatible", "openai", "real", "qwen", "aliyun"}:
        env["DSH_PROVIDER"] = "lab"
    else:
        env["DSH_PROVIDER"] = "lab"
    env["DSH_TIMEOUT"] = os.environ.get("DSH_TIMEOUT") or "180"
    env["DSH_SESSION_ID"] = os.environ.get("DSH_SESSION_ID") or f"{str(how_id).upper()}-{os.getpid()}"
    return env


def redact_harness_env(env: Mapping[str, str]) -> dict[str, str]:
    out = dict(env)
    for secret in ("DEEPSEEK_API_KEY", "LLM_API_KEY"):
        if out.get(secret):
            out[secret] = "***"
    return out


class InjectedHarnessRunner:
    """Test double: writes plugin.py into the staged workspace. No Docker."""

    def __init__(self, write_plugin: Callable[[Path, HarnessContainerRequest], str]) -> None:
        self._write_plugin = write_plugin

    def run(self, request: HarnessContainerRequest) -> HarnessContainerResult:
        dest = Path(request.workspace) / request.plugin_relpath
        body = self._write_plugin(dest, request)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
        return HarnessContainerResult(
            exit_code=0,
            log_text="injected harness runner; gpu=false",
            image="injected",
            gpu=False,
        )


class DockerHarnessRunner:
    """Run dsh in a Linux container. Image must already exist. Never requests GPU."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        image: str | None = None,
        timeout_seconds: int = 180,
        mem_gb: int = 2,
        cpu_count: int = 2,
    ) -> None:
        self._client = client
        self.image = str(image or os.environ.get("SCIENTIST_DSH_IMAGE") or DEFAULT_DSH_IMAGE)
        self.timeout_seconds = max(30, int(timeout_seconds))
        self.mem_gb = max(1, int(mem_gb))
        self.cpu_count = max(1, int(cpu_count))

    def run(self, request: HarnessContainerRequest) -> HarnessContainerResult:
        client = self._client if self._client is not None else connect_docker()
        image = request.image or self.image
        try:
            client.images.get(image)
        except Exception as exc:  # noqa: BLE001
            raise PluginAuthorWorkerError(
                f"image_not_found: {image}. Build docker/dsh-plugin-author; "
                "this worker does not pull or use the GPU experiment image."
            ) from exc
        sessions = Path(request.workspace).parent / "sessions"
        if sessions.exists():
            shutil.rmtree(sessions)
        sessions.mkdir(parents=True, exist_ok=True)
        name = f"scientist-dsh-{request.how_id}".replace("_", "-").lower()
        try:
            old = client.containers.get(name)
            old.remove(force=True)
        except Exception:  # noqa: BLE001
            pass
        run_kwargs: dict[str, Any] = {
            "image": image,
            "detach": True,
            "name": name,
            "working_dir": "/workspace",
            "volumes": {
                str(Path(request.workspace).resolve()): {
                    "bind": "/workspace",
                    "mode": "rw",
                },
                str(sessions.resolve()): {"bind": "/sessions", "mode": "rw"},
            },
            "environment": dict(request.environment),
            "mem_limit": f"{self.mem_gb}g",
            "nano_cpus": int(self.cpu_count * 1_000_000_000),
            "auto_remove": False,
        }
        try:
            container = client.containers.run(**run_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise PluginAuthorWorkerError(f"harness docker run failed: {exc}") from exc
        try:
            exit_code, log_text = _wait_container(
                container, timeout_seconds=self.timeout_seconds
            )
        finally:
            try:
                container.remove(force=True)
            except Exception:  # noqa: BLE001
                pass
        return HarnessContainerResult(
            exit_code=int(exit_code),
            log_text=log_text,
            image=image,
            gpu=False,
        )


def _wait_container(container: Any, *, timeout_seconds: int) -> tuple[int, str]:
    deadline = time.time() + timeout_seconds
    while True:
        container.reload()
        state = container.attrs.get("State") or {}
        if not state.get("Running", False):
            raw = b""
            try:
                raw = container.logs(stdout=True, stderr=True)
            except Exception:  # noqa: BLE001
                raw = b""
            return int(state.get("ExitCode") or 0), raw.decode("utf-8", errors="replace")
        if time.time() > deadline:
            try:
                container.kill()
            except Exception:  # noqa: BLE001
                pass
            raise PluginAuthorWorkerError(
                f"timed_out: harness container exceeded {timeout_seconds}s"
            )
        time.sleep(0.4)


class HarnessPluginWorker:
    """DeepSeek Harness as the HOW plugin author. Lab still smokes and registers."""

    worker_id = "harness"
    authored_by = "harness"

    def __init__(
        self,
        *,
        project_root: Path | str,
        runner: HarnessContainerRunner | None = None,
        work_root: Path | str | None = None,
        image: str | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._runner = runner
        self._work_root = Path(work_root) if work_root is not None else None
        self._image = image

    def propose_plugin_diff(
        self,
        bundle: CodeContextBundle,
        *,
        how_id: str,
        system_prompt: str,
    ) -> PluginAuthorDraft:
        rel = plugin_relpath_for(how_id)
        work_root = self._work_root
        if work_root is None:
            work_root = self._project_root / "outputs" / "_how_plugin_harness"
        work_root = Path(work_root).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        prompt = harness_plugin_task_prompt(
            how_id=how_id,
            rel=rel,
            system_prompt=system_prompt,
            goal=str(bundle.goal or ""),
        )
        workspace = stage_harness_workspace(
            project_root=self._project_root,
            work_root=work_root,
            how_id=how_id,
            prompt=prompt,
        )
        env = _harness_env(how_id, rel)
        image = self._image or os.environ.get("SCIENTIST_DSH_IMAGE") or DEFAULT_DSH_IMAGE
        runner = self._runner if self._runner is not None else DockerHarnessRunner(image=image)
        timeout = getattr(runner, "timeout_seconds", None)
        if timeout:
            env["DSH_TIMEOUT"] = str(int(timeout))
        request = HarnessContainerRequest(
            workspace=workspace,
            plugin_relpath=rel,
            how_id=str(how_id).upper(),
            prompt=prompt,
            image=image,
            environment=env,
        )
        result = runner.run(request)
        if result.gpu:
            raise PluginAuthorWorkerError("harness worker must not request GPU")
        if int(result.exit_code) != 0:
            raise PluginAuthorWorkerError(
                f"harness container exit {result.exit_code}: {result.log_text[-800:]}"
            )
        dest = workspace / rel
        if not dest.is_file():
            raise PluginAuthorWorkerError(f"harness did not write {rel}")
        body = dest.read_text(encoding="utf-8")
        if not body.strip() or not _plugin_has_builder(body):
            raise PluginAuthorWorkerError("harness wrote an empty plugin.py")
        from scientist_lab.core.how_plugin_author import plugin_unified_diff_from_source

        diff = plugin_unified_diff_from_source(how_id, body)
        return PluginAuthorDraft(
            unified_diff=diff,
            worker_id=self.worker_id,
            authored_by=self.authored_by,
            notes={
                "gpu": False,
                "registered": False,
                "docker": self._runner is None,
                "image": result.image,
                "workspace": str(workspace),
                "plugin_relpath": rel,
                "env": redact_harness_env(env),
            },
        )
