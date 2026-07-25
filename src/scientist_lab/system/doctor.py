"""System Doctor checks for Scientist Lab workbench (v2.0.6)."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from scientist_lab import API_VERSION

CheckLevel = Literal["ok", "warning", "error"]


@dataclass
class DoctorCheck:
    id: str
    title: str
    level: CheckLevel
    message: str
    impact: str = ""
    suggested_action: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "message": self.message,
            "impact": self.impact,
            "suggested_action": self.suggested_action,
            "details": self.details,
        }


class SystemDoctor:
    """Run local environment diagnostics. Never opens a shell for the user."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.settings = service.settings

    def run(self) -> dict[str, Any]:
        checks = [
            self._check_python(),
            self._check_database(),
            self._check_schema(),
            self._check_docker(),
            self._check_images(),
            self._check_outputs_writable(),
            self._check_datasets(),
            self._check_runner_profiles(),
            self._check_llm(),
            self._check_replay(),
            self._check_disk(),
            self._check_git(),
            self._check_version(),
            self._check_security_defaults(),
            self._check_dfine_cuda(),
        ]
        summary = {
            "ok": sum(1 for c in checks if c.level == "ok"),
            "warning": sum(1 for c in checks if c.level == "warning"),
            "error": sum(1 for c in checks if c.level == "error"),
        }
        overall: CheckLevel = "ok"
        if summary["error"]:
            overall = "error"
        elif summary["warning"]:
            overall = "warning"
        return {
            "overall": overall,
            "summary": summary,
            "checks": [c.to_dict() for c in checks],
            "components": self._component_snapshot(),
            "version": API_VERSION,
        }

    def _component_snapshot(self) -> dict[str, Any]:
        return {
            "python": sys.version.split()[0],
            "db_path": str(self.settings.db_path),
            "outputs_dir": str(self.settings.outputs_dir),
            "runtime_dir": str(self.settings.runtime_dir),
            "project_root": str(self.settings.project_root),
            "image_registry": dict(self.settings.image_registry or {}),
            "default_llm": "mock",
            "allow_network_by_default": False,
            "allow_real_llm_by_default": False,
            "shell_available_in_ui": False,
        }

    def _check_python(self) -> DoctorCheck:
        major, minor = sys.version_info[:2]
        ok = (major, minor) >= (3, 11)
        return DoctorCheck(
            id="python",
            title="Python 环境",
            level="ok" if ok else "error",
            message=f"Python {sys.version.split()[0]}",
            impact="" if ok else "部分类型注解与依赖可能不可用",
            suggested_action="" if ok else "升级到 Python 3.11+",
            details={"executable": sys.executable},
        )

    def _check_database(self) -> DoctorCheck:
        path = Path(self.settings.db_path)
        try:
            with self.service.session_factory() as session:
                session.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                )
            return DoctorCheck(
                id="database",
                title="数据库连接",
                level="ok",
                message=f"SQLite 可读写: {path}",
                details={"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0},
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="database",
                title="数据库连接",
                level="error",
                message=str(exc),
                impact="无法加载项目与执行记录",
                suggested_action="检查 SCIENTIST_LAB_DB_PATH 与磁盘权限",
            )

    def _check_schema(self) -> DoctorCheck:
        try:
            from sqlalchemy import text

            with self.service.session_factory() as session:
                rows = session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
                tables = sorted(str(r[0]) for r in rows)
            required = {"research_projects", "experiment_nodes", "execution_attempts"}
            missing = sorted(required - set(tables))
            has_payload = False
            if "research_projects" in tables:
                cols = session_columns(self.service.session_factory, "research_projects")
                has_payload = "payload_json" in cols
            if missing:
                return DoctorCheck(
                    id="schema",
                    title="数据库 Schema",
                    level="error",
                    message=f"缺少表: {', '.join(missing)}",
                    impact="核心数据无法持久化",
                    suggested_action="删除损坏库或重新 init_db 后重启服务",
                    details={"tables": tables},
                )
            level: CheckLevel = "ok" if has_payload else "warning"
            return DoctorCheck(
                id="schema",
                title="数据库 Schema",
                level=level,
                message=(
                    "核心表齐全；项目 payload_json 已就绪"
                    if has_payload
                    else "核心表齐全；建议确保 research_projects.payload_json 存在"
                ),
                suggested_action="" if has_payload else "重启 API 以触发 ensure_project_schema",
                details={
                    "tables": tables,
                    "payload_json": has_payload,
                    "schema_version": "v2.0-workbench",
                },
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="schema",
                title="数据库 Schema",
                level="error",
                message=str(exc),
                impact="无法验证 Schema",
                suggested_action="检查数据库文件完整性",
            )

    def _check_docker(self) -> DoctorCheck:
        try:
            self.service.runner.client.ping()
            return DoctorCheck(
                id="docker",
                title="Docker 连接",
                level="ok",
                message="Docker daemon 可达",
                suggested_action="",
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="docker",
                title="Docker 连接",
                level="warning",
                message=str(exc),
                impact="本地 Docker 实验无法运行；Mock/规划仍可用",
                suggested_action="启动 Docker Desktop 后重试 system-doctor",
            )

    def _check_images(self) -> DoctorCheck:
        registry = dict(self.settings.image_registry or {})
        missing: list[str] = []
        present: list[str] = []
        try:
            client = self.service.runner.client
            for key, ref in registry.items():
                try:
                    client.images.get(ref)
                    present.append(f"{key}={ref}")
                except Exception:  # noqa: BLE001
                    missing.append(f"{key}={ref}")
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="docker_images",
                title="Docker 镜像",
                level="warning",
                message=f"无法列举镜像: {exc}",
                impact="可能无法启动受控实验容器",
                suggested_action="确认 Docker 可用后构建 scientist-experiment 镜像",
                details={"registry": registry},
            )
        if missing:
            return DoctorCheck(
                id="docker_images",
                title="Docker 镜像",
                level="warning",
                message=f"缺少 {len(missing)} 个注册镜像",
                impact="对应实验镜像的 run 会失败",
                suggested_action="按启动说明构建缺失镜像",
                details={"missing": missing, "present": present},
            )
        return DoctorCheck(
            id="docker_images",
            title="Docker 镜像",
            level="ok",
            message=f"已找到 {len(present)} 个注册镜像",
            details={"present": present},
        )

    def _check_outputs_writable(self) -> DoctorCheck:
        root = Path(self.settings.outputs_dir)
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".doctor_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return DoctorCheck(
                id="outputs_writable",
                title="输出目录写权限",
                level="ok",
                message=f"可写: {root}",
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="outputs_writable",
                title="输出目录写权限",
                level="error",
                message=str(exc),
                impact="无法保存 Artifact / 报告 / Evidence",
                suggested_action="调整 outputs 目录权限或 SCIENTIST_LAB_OUTPUTS_DIR",
            )

    def _check_datasets(self) -> DoctorCheck:
        try:
            items = self.service.list_datasets()
            broken: list[str] = []
            for item in items:
                path = (
                    item.get("host_path")
                    or item.get("resolved_path")
                    or item.get("path")
                    or item.get("root")
                )
                if path and not Path(str(path)).exists():
                    broken.append(
                        str(item.get("dataset_key") or item.get("key") or path)
                    )
            if broken:
                return DoctorCheck(
                    id="datasets",
                    title="数据集路径",
                    level="warning",
                    message=f"{len(broken)} 个注册数据集路径不存在",
                    impact="相关实验会在启动时失败",
                    suggested_action="在数据注册页重新登记有效路径",
                    details={"broken": broken, "count": len(items)},
                )
            return DoctorCheck(
                id="datasets",
                title="数据集路径",
                level="ok",
                message=f"已注册 {len(items)} 个数据集",
                details={"count": len(items)},
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="datasets",
                title="数据集路径",
                level="warning",
                message=str(exc),
                impact="无法校验数据集注册表",
                suggested_action="检查 Dataset Registry",
            )

    def _check_runner_profiles(self) -> DoctorCheck:
        try:
            profiles = self.service.list_runner_profiles()
            return DoctorCheck(
                id="runner_profiles",
                title="Runner Profiles",
                level="ok" if profiles else "warning",
                message=f"{len(profiles)} 个 runner profile",
                suggested_action="" if profiles else "确认默认 local_docker / mock profile 已创建",
                details={"keys": [p.get("profile_key") or p.get("key") for p in profiles[:20]]},
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="runner_profiles",
                title="Runner Profiles",
                level="warning",
                message=str(exc),
                impact="可能无法选择执行环境",
                suggested_action="重启服务以 ensure_defaults",
            )

    def _check_llm(self) -> DoctorCheck:
        key_set = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SCIENTIST_LAB_OPENAI_API_KEY")
        )
        return DoctorCheck(
            id="llm",
            title="LLM 配置",
            level="ok",
            message=(
                "默认 Mock；真实 API Key 已配置（仍须显式启用）"
                if key_set
                else "默认 Mock；未检测到真实 API Key（符合默认安全）"
            ),
            impact="真实 LLM 需显式 allow_network + quality gate",
            suggested_action="勿在 YAML 写入 API Key；使用环境变量",
            details={
                "default_provider": "mock",
                "api_key_configured": key_set,
                "require_quality_gate_default": True,
            },
        )

    def _check_replay(self) -> DoctorCheck:
        root = Path(self.settings.outputs_dir)
        replay_dirs = list(root.glob("*/llm"))[:20] if root.exists() else []
        return DoctorCheck(
            id="replay",
            title="Replay 可用性",
            level="ok",
            message=(
                f"发现 {len(replay_dirs)} 个项目 LLM 审计目录"
                if replay_dirs
                else "尚无 LLM 审计目录（首次规划后会出现）"
            ),
            details={"sample": [str(p) for p in replay_dirs[:5]]},
        )

    def _check_disk(self) -> DoctorCheck:
        root = Path(self.settings.outputs_dir)
        try:
            usage = shutil.disk_usage(root if root.exists() else Path(self.settings.project_root))
            free_gb = usage.free / (1024**3)
            level: CheckLevel = "ok"
            if free_gb < 1:
                level = "error"
            elif free_gb < 5:
                level = "warning"
            return DoctorCheck(
                id="disk",
                title="磁盘空间",
                level=level,
                message=f"剩余约 {free_gb:.1f} GB",
                impact="" if level == "ok" else "Artifact / 报告可能写入失败",
                suggested_action="" if level == "ok" else "清理 outputs 或扩容磁盘",
                details={
                    "free_bytes": usage.free,
                    "total_bytes": usage.total,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="disk",
                title="磁盘空间",
                level="warning",
                message=str(exc),
                suggested_action="手动检查磁盘剩余空间",
            )

    def _check_git(self) -> DoctorCheck:
        root = Path(self.settings.project_root)
        git_dir = root / ".git"
        if not git_dir.exists():
            # project_root may be scientist-lab; check parent
            parent_git = root.parent / ".git"
            if parent_git.exists():
                git_dir = parent_git
                root = root.parent
            else:
                return DoctorCheck(
                    id="git",
                    title="Git 状态",
                    level="warning",
                    message="未检测到 .git 目录",
                    impact="受控合并 / RC 需要 Git 仓库",
                    suggested_action="在仓库根目录操作或初始化 Git",
                )
        try:
            import subprocess

            head = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            branch = (head.stdout or "").strip() or "unknown"
            is_dirty = bool((dirty.stdout or "").strip())
            return DoctorCheck(
                id="git",
                title="Git 状态",
                level="warning" if is_dirty else "ok",
                message=f"分支 {branch}" + ("（工作区有未提交变更）" if is_dirty else ""),
                impact="" if not is_dirty else "合并前请确认变更来源",
                suggested_action="" if not is_dirty else "仅通过受控 Merge 流程提交",
                details={"branch": branch, "dirty": is_dirty, "root": str(root)},
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="git",
                title="Git 状态",
                level="warning",
                message=str(exc),
                suggested_action="确认 git 在 PATH 中",
            )

    def _check_version(self) -> DoctorCheck:
        return DoctorCheck(
            id="version",
            title="版本标签",
            level="ok",
            message=f"Workbench API {API_VERSION}（基线 v2.1.0）",
            details={"api_version": API_VERSION, "baseline_tag": "v2.1.0"},
        )

    def _check_security_defaults(self) -> DoctorCheck:
        return DoctorCheck(
            id="security",
            title="安全边界",
            level="ok",
            message="默认 Mock · 默认无网络 · 无 UI Shell · 人工审批",
            details={
                "allow_network_by_default": False,
                "allow_real_llm_by_default": False,
                "allow_main_tree_patch": False,
                "shell_in_ui": False,
            },
        )

    def _check_dfine_cuda(self) -> DoctorCheck:
        """Offline Vendor DFINE / CUDA readiness (v2.3.7; no GPU required)."""
        try:
            from scientist_lab.tasks.rgbt_detection.cuda_doctor import (
                build_dfine_cuda_doctor,
            )

            report = build_dfine_cuda_doctor(
                self.settings.project_root,
                image_registry=dict(self.settings.image_registry or {}),
                probe_runtime=False,
            )
            offline_ok = bool(report.get("ok"))
            live = bool(report.get("live_ready"))
            level: CheckLevel = "ok" if offline_ok else "warning"
            return DoctorCheck(
                id="dfine_cuda",
                title="CUDA + Vendor DFINE",
                level=level,
                message=(
                    "离线资源就绪；live_ready 需另跑 dfine-cuda-doctor 探测 GPU"
                    if offline_ok
                    else "Vendor DFINE / CUDA 离线资源不完整"
                ),
                impact=(
                    ""
                    if offline_ok
                    else "无法启动 Vendor DFINE Fast Eval / accept_v23_real"
                ),
                suggested_action=(
                    "scientist-lab dfine-cuda-doctor；指南 docs/dfine-cuda-runthrough.md"
                ),
                details={
                    "overall": report.get("overall"),
                    "live_ready": live,
                    "doctor_version": report.get("doctor_version"),
                    "environment_key": report.get("environment_key"),
                    "check_ids": [c.get("id") for c in (report.get("checks") or [])],
                },
            )
        except Exception as exc:  # noqa: BLE001
            return DoctorCheck(
                id="dfine_cuda",
                title="CUDA + Vendor DFINE",
                level="warning",
                message=str(exc),
                impact="无法汇总 DFINE CUDA 就绪状态",
                suggested_action="检查 third_party/DFINE 与 docker/rgbt-detection-v2-cuda",
            )


def session_columns(session_factory: Any, table: str) -> set[str]:
    from sqlalchemy import text

    with session_factory() as session:
        rows = session.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {str(r[1]) for r in rows}
