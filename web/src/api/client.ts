import type { ApiError } from "../types/api";

export class ApiClientError extends Error {
  status: number;
  code: string;
  details?: unknown;

  constructor(status: number, payload: ApiError | string) {
    if (typeof payload === "string") {
      super(payload);
      this.status = status;
      this.code = "http_error";
    } else {
      super(payload.error?.message || "API request failed");
      this.status = status;
      this.code = payload.error?.code || "http_error";
      this.details = payload.error?.details;
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
