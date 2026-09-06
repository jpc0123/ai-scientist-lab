import type { ApiError } from "../types/api";

export class ApiClientError extends Error {
  status: number;
  code: string;
  type: string;
  details?: unknown;
  retryable: boolean;
  suggestedAction: string;

  constructor(status: number, payload: ApiError | string) {
    if (typeof payload === "string") {
      super(payload);
      this.status = status;
      this.code = "http_error";
      this.type = "http_error";
      this.retryable = status >= 500;
      this.suggestedAction = status >= 500
        ? "可稍后重试；若持续失败，打开系统页运行诊断。"
        : "检查请求后重试。";
    } else {
      const err = payload.error || { code: "http_error", message: "API request failed" };
      super(err.message || "API request failed");
      this.status = status;
      this.code = err.code || err.type || "http_error";
      this.type = err.type || err.code || "http_error";
      this.details = err.details;
      this.retryable = Boolean(err.retryable);
      this.suggestedAction = err.suggested_action || "";
    }
    this.name = "ApiClientError";
  }
}

async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    headers: { Accept: "application/json" },
  });
  const body = await parseBody(res);
  if (!res.ok) {
    throw new ApiClientError(
      res.status,
      (body as ApiError) || res.statusText || "request failed",
    );
  }
  return body as T;
}

export async function apiPost<T>(
  path: string,
  payload?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const body = await parseBody(res);
  if (!res.ok) {
    throw new ApiClientError(
      res.status,
      (body as ApiError) || res.statusText || "request failed",
    );
  }
  return body as T;
}

export async function apiPatch<T>(
  path: string,
  payload?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    method: "PATCH",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const body = await parseBody(res);
  if (!res.ok) {
    throw new ApiClientError(
      res.status,
      (body as ApiError) || res.statusText || "request failed",
    );
  }
  return body as T;
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    method: "DELETE",
    headers: { Accept: "application/json" },
  });
  const body = await parseBody(res);
  if (!res.ok) {
    throw new ApiClientError(
      res.status,
      (body as ApiError) || res.statusText || "request failed",
    );
  }
  return body as T;
}
