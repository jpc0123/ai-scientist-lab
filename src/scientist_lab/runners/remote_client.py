from __future__ import annotations

from typing import Any

import httpx

from scientist_lab.runners.remote_errors import (
    RemoteAuthFailed,
    RemoteCancelFailed,
    RemoteResultFailed,
    RemoteStatusFailed,
    RemoteSubmissionFailed,
    RemoteWorkerUnavailable,
)


class WorkerHttpClient:
    def __init__(
        self,
        endpoint: str,
        *,
        auth_token: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.endpoint}{path}"
        try:
            # Local worker endpoints must not inherit Windows/system HTTP proxies
            # (httpx trust_env=True can yield 502 for 127.0.0.1 while curl works).
            with httpx.Client(
                timeout=self.timeout_seconds, trust_env=False
            ) as client:
                response = client.request(
                    method, url, headers=self._headers(), **kwargs
                )
        except httpx.HTTPError as exc:
            raise RemoteWorkerUnavailable(str(exc)) from exc
        if response.status_code == 401:
            raise RemoteAuthFailed(response.text)
        return response

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/health")
        if response.status_code >= 400:
            raise RemoteWorkerUnavailable(response.text)
        return response.json()

    def capabilities(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/capabilities")
        if response.status_code >= 400:
            raise RemoteWorkerUnavailable(response.text)
        return response.json()

    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request("POST", "/v1/jobs", json=payload)
        if response.status_code >= 400:
            raise RemoteSubmissionFailed(response.text)
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/v1/jobs/{job_id}")
        if response.status_code == 404:
            raise KeyError(f"remote job not found: {job_id}")
        if response.status_code >= 400:
            raise RemoteStatusFailed(response.text)
        return response.json()

    def get_logs(self, job_id: str, cursor: str | None = None) -> dict[str, Any]:
        params = {"cursor": cursor} if cursor else None
        response = self._request("GET", f"/v1/jobs/{job_id}/logs", params=params)
        if response.status_code >= 400:
            raise RemoteStatusFailed(response.text)
        return response.json()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        response = self._request("POST", f"/v1/jobs/{job_id}/cancel")
        if response.status_code >= 400:
            raise RemoteCancelFailed(response.text)
        return response.json()

    def get_result(self, job_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/v1/jobs/{job_id}/result")
        if response.status_code >= 400:
            raise RemoteResultFailed(response.text)
        return response.json()

    def download_artifacts(self, job_id: str, dest: Any) -> None:
        from pathlib import Path

        response = self._request("GET", f"/v1/jobs/{job_id}/artifacts")
        if response.status_code >= 400:
            from scientist_lab.runners.remote_errors import RemoteArtifactDownloadFailed

            raise RemoteArtifactDownloadFailed(response.text)
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
