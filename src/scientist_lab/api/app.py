"""FastAPI application factory for Scientist Lab console + /api/v1."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scientist_lab.api.errors import install_exception_handlers
from scientist_lab.api.schemas import CompareExecutionsBody, CompareNodesBody
from scientist_lab.api.v1 import build_v1_router
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.services.experiment_service import (
    ExperimentService,
    get_shared_service,
)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app(
    *,
    service: ExperimentService | None = None,
    service_factory: Callable[[], ExperimentService] | None = None,
) -> FastAPI:
    """Create the console app.

    Prefer injecting ``service`` / ``service_factory`` in tests so the
    process-wide shared singleton is not required.
    """
    if service_factory is None:
        if service is not None:
            bound = service

            def service_factory() -> ExperimentService:
                return bound

        else:
            service_factory = get_shared_service

    app = FastAPI(
        title="Scientist Lab Web Console API",
        version="1.7.1",
        description=(
            "Local single-user research workbench API. "
            "All mutations go through ExperimentService state machines."
        ),
    )
    install_exception_handlers(app)
    app.include_router(build_v1_router(service_factory))

    # Legacy static console (pre-v1.7 SPA).
    if WEB_DIR.exists() and (WEB_DIR / "static").exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(WEB_DIR / "static")),
            name="static",
        )

    @app.get("/")
    def index() -> FileResponse:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="前端页面不存在")
        return FileResponse(index_path)

    # ---- legacy /api/* (kept for old console.js) -------------------------
    @app.get("/api/health")
    def legacy_health() -> dict:
        svc = service_factory()
        docker_ok = False
        docker_error = None
        try:
            svc.runner.client.ping()
            docker_ok = True
        except Exception as exc:  # noqa: BLE001
            docker_error = str(exc)
        return {
            "ok": True,
            "docker_ok": docker_ok,
            "docker_error": docker_error,
            "db_path": str(svc.settings.db_path),
            "deprecated": True,
            "see": "/api/v1/health",
        }

    @app.get("/api/projects")
    def legacy_list_projects() -> list[dict]:
        return service_factory().list_projects()

    @app.get("/api/nodes")
    def legacy_list_nodes(project_id: str | None = None) -> list[dict]:
        return service_factory().list_nodes(project_id=project_id)

    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict]:
        return [
            {
                "id": "success",
                "title": "正常完成 (Success)",
                "file": "examples/smoke_test_contract.json",
                "description": "标准冒烟实验，应返回 completed + metrics",
            },
            {
                "id": "exception",
                "title": "程序异常 (Exception)",
                "file": "examples/fail_exception.json",
                "description": "容器内抛错，应返回 failed",
            },
            {
                "id": "missing_metrics",
                "title": "缺少 metrics (Missing metrics)",
                "file": "examples/fail_missing_metrics.json",
                "description": "不写 metrics.json，收集阶段失败",
            },
            {
                "id": "timeout",
                "title": "超时 (Timeout)",
                "file": "examples/fail_timeout.json",
                "description": "长时间挂起，约 8 秒后 timed_out",
            },
            {
                "id": "multi_a",
                "title": "同节点尝试 A (Multi-attempt A)",
                "file": "examples/multi_attempt_a.json",
                "description": "同一 node_retry_demo 第一次尝试",
            },
            {
                "id": "multi_b",
                "title": "同节点尝试 B (Multi-attempt B)",
                "file": "examples/multi_attempt_b.json",
                "description": "同一节点第二次尝试，便于对比",
            },
            {
                "id": "digits_real",
                "title": "Digits 真实训练 (Real MLP)",
                "file": "examples/digits_real_contract.json",
                "description": "sklearn Digits + MLP，环境 digits-mlp-v1 / v2 镜像",
            },
            {
                "id": "digits_real_02",
                "title": "Digits 更大隐层 (Real MLP 128)",
                "file": "examples/digits_real_contract_02.json",
                "description": "单变量：仅 hidden_units=128，便于与 node_003 对比",
            },
            {
                "id": "digits_invalid_seed",
                "title": "Digits 不同 seed (Verifier 负例)",
                "file": "examples/compare_invalid_seed.json",
                "description": "seed=99，与 baseline 比较时应 inconclusive",
            },
        ]

    @app.get("/api/scenarios/{scenario_id}")
    def get_scenario(scenario_id: str) -> dict:
        mapping = {item["id"]: item for item in list_scenarios()}
        item = mapping.get(scenario_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"未知场景: {scenario_id}")
        path = service_factory().settings.project_root / item["file"]
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"场景文件不存在: {path}")
        import json

        contract = json.loads(path.read_text(encoding="utf-8"))
        return {**item, "contract": contract}

    @app.get("/api/executions")
    def legacy_list_executions(
        project_id: str | None = None,
        node_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        attempts = service_factory().list_executions(
            limit=limit, project_id=project_id, node_id=node_id
        )
        return [a.model_dump() for a in attempts]

    @app.get("/api/executions/{execution_id}")
    def legacy_get_execution(execution_id: str) -> dict:
        try:
            return service_factory().get_execution(execution_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/executions/{execution_id}/log")
    def legacy_get_log(execution_id: str) -> dict:
        try:
            text = service_factory().get_log_text(execution_id)
            return {"execution_id": execution_id, "log": text}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/executions/{execution_id}/cancel")
    def legacy_cancel_execution(execution_id: str) -> dict:
        try:
            attempt = service_factory().cancel_execution(execution_id)
            return attempt.model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/contracts/submit")
    def submit_contract(contract: ExperimentContract) -> dict:
        try:
            result = service_factory().submit_contract(contract)
            return {
                "execution_id": result.execution_id,
                "status": str(result.status),
                "output_directory": result.output_directory,
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/compare")
    def compare(req: CompareExecutionsBody) -> dict:
        try:
            return service_factory().compare_executions(
                req.execution_id_a, req.execution_id_b
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/compare/nodes")
    def compare_nodes(req: CompareNodesBody) -> dict:
        try:
            return service_factory().compare_nodes(req.node_id_a, req.node_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/compare/node-groups")
    def compare_node_groups(req: CompareNodesBody) -> dict:
        try:
            return service_factory().compare_node_groups(
                req.node_id_a, req.node_id_b
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/nodes/{node_id}/aggregate")
    def aggregate_node(node_id: str) -> dict:
        try:
            return service_factory().aggregate_node(node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/feedback/analyze")
    def analyze_feedback(req: CompareNodesBody) -> dict:
        try:
            return service_factory().analyze_feedback(req.node_id_a, req.node_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/feedback/{node_id}")
    def show_feedback(node_id: str) -> dict:
        try:
            return service_factory().show_feedback(node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/feedback/propose-next")
    def propose_next(req: CompareNodesBody) -> dict:
        try:
            return service_factory().propose_next(req.node_id_a, req.node_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
