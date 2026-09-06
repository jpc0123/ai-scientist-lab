from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from scientist_worker.models import JobSubmitRequest
from scientist_worker.security import SecurityError, require_token
from scientist_worker.service import WorkerService
from scientist_worker.settings import WorkerSettings


def create_app(
    settings: WorkerSettings | None = None,
    service: WorkerService | None = None,
) -> FastAPI:
    settings = (settings or WorkerSettings()).resolve()
    worker = service or WorkerService(settings)
    app = FastAPI(title="Scientist GPU Worker", version=settings.version)
    app.state.settings = settings
    app.state.worker = worker

    def auth(authorization: str | None = Header(default=None)) -> None:
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
        try:
            require_token(
                token,
                settings.auth_token,
                require_auth=settings.require_auth or bool(settings.auth_token),
            )
        except SecurityError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get("/v1/health")
    def health() -> dict:
        return worker.health()

    @app.get("/v1/capabilities")
    def capabilities(_: None = Depends(auth)) -> dict:
        return worker.capabilities()

    @app.post("/v1/jobs")
    def submit_job(payload: JobSubmitRequest, _: None = Depends(auth)) -> dict:
        try:
            return worker.submit(payload).model_dump()
        except SecurityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str, _: None = Depends(auth)) -> dict:
        try:
            return worker.get_status(job_id).model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/logs")
    def job_logs(
        job_id: str, cursor: str | None = None, _: None = Depends(auth)
    ) -> dict:
        try:
            return worker.get_logs(job_id, cursor=cursor).model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, _: None = Depends(auth)) -> dict:
        try:
            return worker.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/result")
    def job_result(job_id: str, _: None = Depends(auth)) -> dict:
        try:
            return worker.get_result(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str, _: None = Depends(auth)):
        try:
            path = worker.artifacts_path(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type="application/gzip",
            filename=Path(path).name,
        )

    return app
