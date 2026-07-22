from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.services.experiment_service import get_shared_service

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app() -> FastAPI:
    app = FastAPI(title="scientist-lab console", version="1.4.0")
    service = get_shared_service()

    if WEB_DIR.exists():
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

    @app.get("/api/health")
    def health() -> dict:
        docker_ok = False
        docker_error = None
        try:
            service.runner.client.ping()
            docker_ok = True
        except Exception as exc:  # noqa: BLE001
            docker_error = str(exc)
        return {
            "ok": True,
            "docker_ok": docker_ok,
            "docker_error": docker_error,
            "db_path": str(service.settings.db_path),
        }

    @app.get("/api/projects")
    def list_projects() -> list[dict]:
        return service.list_projects()

    @app.get("/api/nodes")
    def list_nodes(project_id: str | None = None) -> list[dict]:
        return service.list_nodes(project_id=project_id)

    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict]:
        """内置演示场景，对应第二步失败/成功路径。"""
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
        path = service.settings.project_root / item["file"]
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"场景文件不存在: {path}")
        import json

        contract = json.loads(path.read_text(encoding="utf-8"))
        return {**item, "contract": contract}

    @app.get("/api/executions")
    def list_executions(
        project_id: str | None = None,
        node_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        attempts = service.list_executions(
            limit=limit, project_id=project_id, node_id=node_id
        )
        return [a.model_dump() for a in attempts]

    @app.get("/api/executions/{execution_id}")
    def get_execution(execution_id: str) -> dict:
        try:
            return service.get_execution(execution_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/executions/{execution_id}/log")
    def get_log(execution_id: str) -> dict:
        try:
            text = service.get_log_text(execution_id)
            return {"execution_id": execution_id, "log": text}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/executions/{execution_id}/cancel")
    def cancel_execution(execution_id: str) -> dict:
        try:
            attempt = service.cancel_execution(execution_id)
            return attempt.model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/contracts/submit")
    def submit_contract(contract: ExperimentContract) -> dict:
        try:
            result = service.submit_contract(contract)
            return {
                "execution_id": result.execution_id,
                "status": str(result.status),
                "output_directory": result.output_directory,
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    class CompareRequest(BaseModel):
        execution_id_a: str
        execution_id_b: str

    class CompareNodesRequest(BaseModel):
        node_id_a: str
        node_id_b: str

    @app.post("/api/compare")
    def compare(req: CompareRequest) -> dict:
        try:
            return service.compare_executions(req.execution_id_a, req.execution_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/compare/nodes")
    def compare_nodes(req: CompareNodesRequest) -> dict:
        try:
            return service.compare_nodes(req.node_id_a, req.node_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/compare/node-groups")
    def compare_node_groups(req: CompareNodesRequest) -> dict:
        try:
            return service.compare_node_groups(req.node_id_a, req.node_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/nodes/{node_id}/aggregate")
    def aggregate_node(node_id: str) -> dict:
        try:
            return service.aggregate_node(node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/feedback/analyze")
    def analyze_feedback(req: CompareNodesRequest) -> dict:
        try:
            return service.analyze_feedback(req.node_id_a, req.node_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/feedback/{node_id}")
    def show_feedback(node_id: str) -> dict:
        try:
            return service.show_feedback(node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/feedback/propose-next")
    def propose_next(req: CompareNodesRequest) -> dict:
        try:
            return service.propose_next(req.node_id_a, req.node_id_b)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
