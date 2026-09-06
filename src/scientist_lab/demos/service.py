"""Built-in demo project seeders (v2.0.8).

Creates ready-to-explore projects without auto-running expensive experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from scientist_lab.domain import NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExperimentNode, utc_now_iso

DemoKind = Literal["digits", "rgbt-debug"]

DEMO_KINDS: tuple[str, ...] = ("digits", "rgbt-debug")


class DemoService:
    def __init__(self, experiments: Any) -> None:
        self.experiments = experiments
        self.root = Path(experiments.settings.project_root)
        self.examples = self.root / "examples"

    def list_demos(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": "digits",
                "title": "Digits 快速演示",
                "description": (
                    "CPU 友好的 sklearn Digits + MLP。"
                    "适合多 seed、节点比较、Evidence、有限实验树与报告演示。"
                ),
                "requires_cuda": False,
                "requires_docker": True,
                "task_type": "general_ml",
            },
            {
                "kind": "rgbt-debug",
                "title": "RGB-T Debug 演示",
                "description": (
                    "双模态 RGB / Thermal / Fusion 协议与 Claim Gate 结构。"
                    "不要求真实 CUDA + DFINE；用于走通数据注册与证据流程。"
                ),
                "requires_cuda": False,
                "requires_docker": False,
                "task_type": "rgbt_detection",
            },
        ]

    def create(self, kind: str, *, force: bool = False) -> dict[str, Any]:
        key = (kind or "").strip().lower()
        if key not in DEMO_KINDS:
            raise ValueError(
                f"unknown demo kind: {kind!r}; expected one of {list(DEMO_KINDS)}"
            )
        if key == "digits":
            return self._create_digits(force=force)
        return self._create_rgbt_debug(force=force)

    def _create_digits(self, *, force: bool) -> dict[str, Any]:
        project_id = "demo_digits_v20"
        existing = self.experiments.repo.get_project(project_id)
        if existing is not None and not force:
            return {
                "status": "exists",
                "kind": "digits",
                "project": self.experiments.get_project(project_id),
                "message": "Digits demo already exists; pass force=true to recreate metadata",
                "next_steps": self._digits_next_steps(project_id),
            }

        protocol_path = self.examples / "digits_demo_protocol.json"
        if not protocol_path.is_file():
            self._write_digits_protocol(protocol_path)
        protocol = self.experiments.protocols.create_from_path(protocol_path)
        protocol_id = str(protocol.protocol_id)

        project = self.experiments.create_project(
            title="Digits 演示项目",
            research_question="单隐层 MLP 在 Digits 上能否超过 85% 准确率？",
            research_goal="用可复现多 seed 流程演示比较、证据与报告",
            description=(
                "内置 Digits demo（v2.0.8）。默认不自动跑实验；"
                "可用 examples/digits_real_contract*.json 手动或 CLI 执行。"
            ),
            task_type="general_ml",
            dataset_keys=["sklearn:digits"],
            protocol_ids=[protocol_id],
            runner_profile_keys=["local"],
            expected_metrics={"accuracy": 0.85},
            constraints={"gpu_required": False, "max_epochs": 30},
            protocol_draft={
                "seeds": [42, 43, 44],
                "claim_level": "exploratory_comparison",
                "primary_metric": "accuracy",
            },
            project_id=project_id,
            mark_ready=True,
        )

        nodes = self._register_contract_nodes(
            project_id=project_id,
            contracts=[
                "digits_real_contract.json",
                "digits_real_contract_02.json",
            ],
            rewrite_project_id=True,
        )

        return {
            "status": "created",
            "kind": "digits",
            "project": project,
            "protocol_id": protocol_id,
            "nodes": nodes,
            "contracts": [
                "examples/digits_real_contract.json",
                "examples/digits_real_contract_02.json",
            ],
            "auto_ran_experiments": False,
            "next_steps": self._digits_next_steps(project_id),
        }

    def _create_rgbt_debug(self, *, force: bool) -> dict[str, Any]:
        project_id = "demo_rgbt_debug_v20"
        existing = self.experiments.repo.get_project(project_id)
        if existing is not None and not force:
            return {
                "status": "exists",
                "kind": "rgbt-debug",
                "project": self.experiments.get_project(project_id),
                "message": "RGB-T debug demo already exists; pass force=true to refresh",
                "next_steps": self._rgbt_next_steps(project_id),
            }

        protocol_path = self.examples / "rgbt_protocol.json"
        protocol = self.experiments.protocols.create_from_path(protocol_path)
        protocol_id = str(protocol.protocol_id)

        project = self.experiments.create_project(
            title="RGB-T Debug 演示",
            research_question="在固定协议下，Fusion 是否相对 RGB/Thermal 单模态有可辩护增益？",
            research_goal="演示双模态验证、Smoke/Fast Eval、Claim Gate 与证据结构",
            description=(
                "内置 RGB-T debug demo（v2.0.8）。"
                "不要求真实 CUDA + DFINE；先用协议与节点骨架走通工作台。"
            ),
            task_type="rgbt_detection",
            dataset_keys=["dataset:rgbt_fast_eval_v1"],
            protocol_ids=[protocol_id],
            runner_profile_keys=["local", "mock"],
            expected_metrics={"mAP50_95": 0.0},
            constraints={
                "gpu_required": False,
                "claim_gate": True,
                "no_auto_scientific_claim": True,
            },
            protocol_draft={
                "execution_mode": "fast_eval",
                "seeds": [42, 43, 44],
                "claim_level": "exploratory_comparison",
                "allowed_variables": ["input_mode", "fusion_method"],
            },
            project_id=project_id,
            mark_ready=True,
        )

        nodes = self._register_contract_nodes(
            project_id=project_id,
            contracts=[
                "rgbt_rgb_smoke_contract.json",
                "rgbt_thermal_smoke_contract.json",
                "rgbt_fusion_smoke_contract.json",
                "rgbt_fast_rgb_contract.json",
                "rgbt_fast_thermal_contract.json",
                "rgbt_fast_fusion_contract.json",
            ],
            rewrite_project_id=True,
        )

        return {
            "status": "created",
            "kind": "rgbt-debug",
            "project": project,
            "protocol_id": protocol_id,
            "nodes": nodes,
            "contracts": [
                "examples/rgbt_*_smoke_contract.json",
                "examples/rgbt_fast_*_contract.json",
            ],
            "auto_ran_experiments": False,
            "claim_gate_note": (
                "正式科学主张仍受 Claim Gate 约束；demo 仅提供结构，不自动宣称验证结果。"
            ),
            "next_steps": self._rgbt_next_steps(project_id),
        }

    def _register_contract_nodes(
        self,
        *,
        project_id: str,
        contracts: list[str],
        rewrite_project_id: bool,
    ) -> list[dict[str, Any]]:
        now = utc_now_iso()
        created: list[dict[str, Any]] = []
        for name in contracts:
            path = self.examples / name
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if rewrite_project_id:
                payload["project_id"] = project_id
            node_id = str(payload.get("node_id") or path.stem)
            existing = self.experiments.repo.get_node(node_id)
            if existing is not None:
                created.append(
                    {
                        "node_id": node_id,
                        "status": str(existing.status),
                        "source": name,
                        "skipped": True,
                    }
                )
                continue
            node = ExperimentNode(
                node_id=node_id,
                project_id=project_id,
                node_type=NodeType.BASELINE,
                stage=NodeStage.INTAKE,
                status=NodeStatus.PLANNED,
                depth=0,
                contract_json=payload,
                created_at=now,
                updated_at=now,
            )
            self.experiments.repo.upsert_node(node)
            created.append(
                {
                    "node_id": node_id,
                    "status": "planned",
                    "source": name,
                    "skipped": False,
                }
            )
        return created

    def _write_digits_protocol(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol_id": "protocol_digits_demo_001",
            "project_id": "demo_digits_v20",
            "title": "Digits MLP exploratory protocol",
            "task_type": "general_ml",
            "dataset_reference": "sklearn:digits",
            "dataset_version": "builtin",
            "split_reference": "split:digits_holdout_v1",
            "environment_key": "digits-mlp-v1",
            "code_reference": "local:experiment_app",
            "code_version": "v0.1.0",
            "execution_mode": "fast_eval",
            "seeds": [42, 43, 44],
            "primary_metric": "accuracy",
            "secondary_metrics": ["f1_macro"],
            "fixed_parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "batch_size": 64,
                "test_size": 0.2,
            },
            "allowed_variables": ["hidden_units"],
            "resource_metrics": ["duration_seconds"],
            "claim_level": "exploratory_comparison",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _digits_next_steps(project_id: str) -> list[str]:
        return [
            f"打开项目 /projects/{project_id}",
            "（可选）scientist-lab run examples/digits_real_contract.json",
            "在比较工作台比较 node_003 / node_004",
            "在证据中心构建 Claim Matrix，再生成报告",
        ]

    @staticmethod
    def _rgbt_next_steps(project_id: str) -> list[str]:
        return [
            f"打开项目 /projects/{project_id}",
            "查看协议与节点（Smoke / Fast Eval）",
            "在 Claim Matrix 查看 blocked/supported 结构",
            "无需 CUDA 即可走通规划与证据页面；真实检测需另配镜像",
        ]
