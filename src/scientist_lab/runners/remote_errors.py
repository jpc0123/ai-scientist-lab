from __future__ import annotations


class RemoteRunnerError(RuntimeError):
    def __init__(self, error_type: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


class RemoteWorkerUnavailable(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_worker_unavailable", message, retryable=True)


class RemoteAuthFailed(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_auth_failed", message, retryable=False)


class RemoteCapabilityMismatch(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_capability_mismatch", message, retryable=False)


class RemoteSubmissionFailed(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_submission_failed", message, retryable=True)


class RemoteStatusFailed(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_status_failed", message, retryable=True)


class RemoteCancelFailed(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_cancel_failed", message, retryable=True)


class RemoteResultFailed(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_result_failed", message, retryable=False)


class RemoteArtifactDownloadFailed(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "remote_artifact_download_failed", message, retryable=True
        )


class RemoteArtifactCorrupt(RemoteRunnerError):
    def __init__(self, message: str) -> None:
        super().__init__("remote_artifact_corrupt", message, retryable=False)
