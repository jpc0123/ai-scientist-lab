"""Dataset Workspace API. Read-only freeze + bind to SQLite registry. Not an Agent."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from scientist_lab.api.errors import http_error
from scientist_lab.services.experiment_service import ExperimentService


class DatasetResolveBody(BaseModel):
    dataset_id: str = Field(min_length=1)
    slice_id: str | None = None


class DatasetBindBody(BaseModel):
    dataset_id: str = Field(min_length=1)
    host_path: str | None = Field(
        default=None,
        description="Optional local processed root. Same dataset_id only; does not change fingerprint.",
    )
    rebind: bool = Field(
        default=False,
        description="If true, update SQLite host_path for an already-bound dataset_id.",
    )


def build_dataset_workspace_router(
    get_service: Callable[[], ExperimentService],
) -> APIRouter:
    router = APIRouter(prefix="/dataset-workspace", tags=["dataset-workspace"])

    def service_dep() -> ExperimentService:
        return get_service()

    @router.get("")
    @router.get("/")
    def overview(service: ExperimentService = Depends(service_dep)) -> dict[str, Any]:
        return service.list_dataset_workspace()

    @router.get("/datasets/{dataset_id}")
    def get_dataset(
        dataset_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.resolve_dataset_contract(dataset_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.get("/slices/{slice_id}")
    def get_slice(
        slice_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.datasets.workspace import DatasetContractError

        try:
            spec = service.dataset_workspace().load_slice(slice_id)
        except DatasetContractError as exc:
            msg = str(exc)
            if "mismatch" in msg or "not frozen" in msg:
                raise http_error(409, code="conflict", message=msg) from exc
            raise http_error(404, code="not_found", message=msg) from exc
        spec = dict(spec)
        spec.pop("membership", None)
        spec["membership_omitted"] = True
        spec["note"] = "Id lists are frozen on disk; API does not return them for rewrite."
        return spec

    @router.post("/resolve")
    def resolve_contract(
        body: DatasetResolveBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.resolve_dataset_contract(body.dataset_id, slice_id=body.slice_id)
        except (KeyError, FileNotFoundError) as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc

    @router.post("/bind")
    def bind_processed(
        body: DatasetBindBody,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.bind_dataset_workspace(
                body.dataset_id,
                host_path=body.host_path,
                rebind=bool(body.rebind),
            )
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except FileNotFoundError as exc:
            raise http_error(409, code="conflict", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/enable")
    def enable_dataset(
        dataset_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.enable_dataset(dataset_id)
        except KeyError as exc:
            raise http_error(
                404,
                code="not_found",
                message=str(exc) + "；请先绑定 processed 根到 SQLite registry。",
            ) from exc

    @router.post("/datasets/{dataset_id}/disable")
    def disable_dataset(
        dataset_id: str,
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        try:
            return service.disable_dataset(dataset_id)
        except KeyError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/pair-previews")
    def list_pair_previews(
        dataset_id: str,
        split: str = Query("train"),
        limit: int = Query(3, ge=1, le=8),
        service: ExperimentService = Depends(service_dep),
    ) -> dict[str, Any]:
        from scientist_lab.datasets.rgbt_pair_preview import preview_manifest

        try:
            return preview_manifest(
                service.settings.project_root,
                dataset_id,
                split=split,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/pair-previews/{index}/png")
    def pair_preview_png(
        dataset_id: str,
        index: int,
        split: str = Query("train"),
        service: ExperimentService = Depends(service_dep),
    ) -> Response:
        from scientist_lab.datasets.rgbt_pair_preview import (
            compose_pair_png,
            list_pair_stems,
            modality_path,
            resolve_processed_root,
        )

        try:
            processed = resolve_processed_root(service.settings.project_root, dataset_id)
            names = list_pair_stems(processed, split=split, limit=max(index + 1, 1))
            if index < 0 or index >= len(names):
                raise FileNotFoundError(f"preview index {index} missing")
            name = names[index]
            rgb = modality_path(processed, split=split, modality="rgb", name=name)
            thermal = modality_path(processed, split=split, modality="thermal", name=name)
            png = compose_pair_png(
                rgb,
                thermal,
                caption=f"{split}/{name}  |  left RGB · right grayscale thermal",
            )
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        return Response(content=png, media_type="image/png")

    @router.get("/datasets/{dataset_id}/images/{split}/{modality}/{name}")
    def pair_source_image(
        dataset_id: str,
        split: str,
        modality: str,
        name: str,
        service: ExperimentService = Depends(service_dep),
    ) -> FileResponse:
        from scientist_lab.datasets.rgbt_pair_preview import (
            modality_path,
            resolve_processed_root,
        )

        try:
            processed = resolve_processed_root(service.settings.project_root, dataset_id)
            path = modality_path(processed, split=split, modality=modality, name=name)
        except FileNotFoundError as exc:
            raise http_error(404, code="not_found", message=str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, code="bad_request", message=str(exc)) from exc
        return FileResponse(path)

    return router
