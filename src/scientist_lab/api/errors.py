"""Unified API error envelope (v1.7.1)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def error_body(
    *,
    code: str,
    message: str,
    details: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    return payload


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            body = detail
            code = str((detail.get("error") or {}).get("code") or "http_error")
        else:
            message = detail if isinstance(detail, str) else str(detail)
            code = {
                400: "bad_request",
                404: "not_found",
                409: "conflict",
                422: "validation_error",
            }.get(exc.status_code, "http_error")
            body = error_body(code=code, message=message)
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_body(
                code="validation_error",
                message="request validation failed",
                details=exc.errors(),
            ),
        )


def http_error(
    status_code: int,
    *,
    code: str,
    message: str,
    details: Any = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=error_body(code=code, message=message, details=details),
    )
