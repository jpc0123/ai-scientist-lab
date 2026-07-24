"""Security posture report for the local workbench (v2.0.9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab import API_VERSION
from scientist_lab.workbench.config import load_workbench_config


def build_security_posture(*, project_root: Path | None = None) -> dict[str, Any]:
    """Return enforced security boundaries (display + audit, not the enforcer)."""
    cfg = load_workbench_config(project_root=project_root)
    security = dict(cfg.get("security") or {})
    llm = dict(cfg.get("llm") or {})

    allow_network = bool(security.get("allow_network_by_default", False))
    allow_real_llm = bool(security.get("allow_real_llm_by_default", False))
    allow_main_tree = bool(security.get("allow_main_tree_patch", False))
    default_provider = str(llm.get("default_provider") or "mock")

    boundaries = [
        {
            "id": "default_mock_llm",
            "label": "默认 Mock LLM",
            "enforced": default_provider == "mock" and not allow_real_llm,
            "detail": f"default_provider={default_provider}",
        },
        {
            "id": "no_network_by_default",
            "label": "默认无外网科研调用",
            "enforced": not allow_network,
            "detail": "allow_network_by_default=false",
        },
        {
            "id": "explicit_real_llm",
            "label": "真实 LLM 需显式启用",
            "enforced": not allow_real_llm,
            "detail": "allow_real_llm_by_default=false",
        },
        {
            "id": "human_approval",
            "label": "人工审批驱动",
            "enforced": True,
            "detail": "Plan / Iteration / Patch / Merge 需人工确认",
        },
        {
            "id": "no_arbitrary_shell",
            "label": "无任意 Shell",
            "enforced": True,
            "detail": "API 无通用 shell 端点；前端无终端输入框",
        },
        {
            "id": "no_auto_git_push",
            "label": "无自动 Git Push",
            "enforced": True,
            "detail": "合入走 Merge Center；回滚仅 revert",
        },
        {
            "id": "no_llm_auto_merge",
            "label": "无 LLM 自动合并",
            "enforced": True,
            "detail": "合并意图与 Finalize 均需人工",
        },
        {
            "id": "claim_gate",
            "label": "Claim Gate",
            "enforced": True,
            "detail": "不提供关闭 Claim Gate 的 UI",
        },
        {
            "id": "no_main_tree_patch_from_ui",
            "label": "UI 不可写主工作树",
            "enforced": not allow_main_tree,
            "detail": "allow_main_tree_patch=false；补丁仅沙箱",
        },
        {
            "id": "artifact_sha256",
            "label": "Artifact SHA256",
            "enforced": True,
            "detail": "执行产物校验哈希",
        },
        {
            "id": "no_auto_rerun_on_recover",
            "label": "恢复不自动重跑昂贵实验",
            "enforced": True,
            "detail": "recover 仅标记 interrupted / 刷新远程状态",
        },
    ]

    ui_forbidden = [
        "Shell 输入框",
        "任意 Docker 参数",
        "任意挂载路径",
        "任意 Git 命令",
        "API Key 明文显示",
        "关闭 Claim Gate 的按钮",
        "绕过审批的按钮",
    ]

    weakened = [b for b in boundaries if not b["enforced"]]
    overall = "ok" if not weakened else "warning"

    return {
        "version": API_VERSION,
        "overall": overall,
        "boundaries": boundaries,
        "ui_forbidden": ui_forbidden,
        "config_snapshot": {
            "llm_default_provider": default_provider,
            "allow_network_by_default": allow_network,
            "allow_real_llm_by_default": allow_real_llm,
            "allow_main_tree_patch": allow_main_tree,
        },
        "note": (
            "本报告用于展示与审计；真正权限仍由后端状态机与 Verifier 强制。"
            "前端隐藏 ≠ 安全。"
        ),
    }
