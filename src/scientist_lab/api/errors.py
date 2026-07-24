"""Product-facing API error envelope (v2.0.9).

Plan shape:
  error.type / message / retryable / details / suggested_action

``code`` is kept as an alias of ``type`` for older clients.
Tracebacks never go to the client — only to server logs.
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("scientist_lab.api")

# Known error types → defaults (override per call when needed).
ERROR_CATALOG: dict[str, dict[str, Any]] = {
    "not_found": {
        "retryable": False,
        "suggested_action": "确认资源 ID 是否正确，或先在列表页刷新。",
    },
    "bad_request": {
        "retryable": False,
        "suggested_action": "检查请求参数后重试。",
    },
    "invalid_request": {
        "retryable": False,
        "suggested_action": "检查请求参数后重试。",
    },
    "validation_error": {
        "retryable": False,
        "suggested_action": "按字段提示修正输入后再提交。",
    },
    "conflict": {
        "retryable": False,
        "suggested_action": "刷新页面查看当前状态，再决定下一步。",
    },
    "protocol_mismatch": {
        "retryable": False,
        "suggested_action": "核对协议固定参数、seed 与允许变量后再提交。",
    },
    "permission_denied": {
        "retryable": False,
        "suggested_action": "此操作被安全边界拒绝；不要尝试绕过审批或 Claim Gate。",
    },
    "unsafe_operation": {
        "retryable": False,
        "suggested_action": "使用受控流程（审批 / 沙箱 / Merge Center），禁止任意 Shell 或主树写入。",
    },
    "docker_unavailable": {
        "retryable": True,
        "suggested_action": "启动 Docker Desktop 后执行 system-doctor，再重试实验。",
    },
    "internal_error": {
        "retryable": True,
        "suggested_action": "可稍后重试；若持续失败，运行 scientist-lab system-doctor。",
    },
    "http_error": {
        "retryable": False,
        "suggested_action": "查看错误说明；必要时打开系统页做诊断。",
    },
    "demo_create_failed": {
        "retryable": True,
        "suggested_action": "确认 examples/ 完整后重试；或使用 --force。",
    },
}

_TRACE_HINT = re.compile(
    r"(Traceback \(most recent call last\)|File \".+\", line \d+)",
    re.IGNORECASE,
)


def sanitize_message(message: str) -> str:
    text = (message or "").strip() or "request failed"
    if _TRACE_HINT.search(text) or "\n  File " in text:
        return "内部错误已记录到服务日志；详情未向界面暴露。"
    # Cap runaway messages
    if len(text) > 800:
        return text[:800] + "…"
    return text


def error_body(
    *,
    code: str,
    message: str,
    details: Any = None,
    retryable: bool | None = None,
    suggested_action: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    kind = (error_type or code or "http_error").strip() or "http_error"
    meta = ERROR_CATALOG.get(kind) or ERROR_CATALOG.get(code) or {}
    retry = bool(meta.get("retryable", False) if retryable is None else retryable)
    action = (
        suggested_action
        if suggested_action is not None
        else str(meta.get("suggested_action") or "")
    )
    payload: dict[str, Any] = {
        "error": {
            "type": kind,
            "code": kind,
            "message": sanitize_message(message),
            "retryable": retry,
            "details": {} if details is None else details,
            "suggested_action": action,
        }
    }
    return payload


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            err = dict(detail.get("error") or {})
            # Normalize legacy / partial bodies
            body = error_body(
                code=str(err.get("type") or err.get("code") or "http_error"),
                message=str(err.get("message") or "request failed"),
                details=err.get("details"),
                retryable=err.get("retryable") if "retryable" in err else None,
                suggested_action=(
                    str(err["suggested_action"])
                    if err.get("suggested_action") is not None
                    else None
                ),
            )
            return JSONResponse(status_code=exc.status_code, content=body)

        message = detail if isinstance(detail, str) else str(detail)
        code = {
            400: "bad_request",
            403: "permission_denied",
            404: "not_found",
            409: "conflict",
            422: "validation_error",
        }.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code=code, message=message),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_body(
                code="validation_error",
                message="请求参数校验失败",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Never send traceback to the client.
        logger.error(
            "unhandled_api_error path=%s\n%s",
            request.url.path,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        return JSONResponse(
            status_code=500,
            content=error_body(
                code="internal_error",
                message="服务器内部错误（已写入日志，未向界面暴露堆栈）",
                details={"path": str(request.url.path)},
            ),
        )


def http_error(
    status_code: int,
    *,
    code: str,
    message: str,
    details: Any = None,
    retryable: bool | None = None,
    suggested_action: str | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=error_body(
            code=code,
            message=message,
            details=details,
            retryable=retryable,
            suggested_action=suggested_action,
        ),
    )
