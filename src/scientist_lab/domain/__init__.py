from __future__ import annotations

from enum import StrEnum


class ProjectStatus(StrEnum):
    """Project lifecycle (v2.0.1). Legacy ``active`` maps to ready on read."""

    DRAFT = "draft"
    CONFIGURING = "configuring"
    READY = "ready"
    RUNNING = "running"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    # Pre-v2.0 rows
    ACTIVE = "active"


class NodeType(StrEnum):
    BASELINE = "baseline"
    IMPROVEMENT = "improvement"
    ABLATION = "ablation"
    DEBUG = "debug"
    SMOKE = "smoke"


class NodeStatus(StrEnum):
    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStage(StrEnum):
    INTAKE = "intake"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    DONE = "done"


class JobStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ErrorType(StrEnum):
    DOCKER_UNAVAILABLE = "docker_unavailable"
    IMAGE_NOT_FOUND = "image_not_found"
    CONTAINER_START_FAILED = "container_start_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    OUT_OF_MEMORY = "out_of_memory"
    INVALID_CONTRACT = "invalid_contract"
    MISSING_ARTIFACT = "missing_artifact"
    INVALID_METRICS = "invalid_metrics"
    EXPERIMENT_CODE_ERROR = "experiment_code_error"
    DATASET_NOT_FOUND = "dataset_not_found"
    DATASET_NOT_REGISTERED = "dataset_not_registered"
    DATASET_PAIR_MISMATCH = "dataset_pair_mismatch"
    INVALID_ANNOTATION = "invalid_annotation"
    CORRUPT_IMAGE = "corrupt_image"
    MODEL_CONFIG_ERROR = "model_config_error"
    CHECKPOINT_MISSING = "checkpoint_missing"
    CHECKPOINT_CORRUPT = "checkpoint_corrupt"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    TRAINING_DIVERGED = "training_diverged"
    NAN_LOSS = "nan_loss"
    EVALUATION_FAILED = "evaluation_failed"
    INVALID_DETECTION_METRICS = "invalid_detection_metrics"
    REMOTE_WORKER_UNAVAILABLE = "remote_worker_unavailable"
    REMOTE_AUTH_FAILED = "remote_auth_failed"
    REMOTE_CAPABILITY_MISMATCH = "remote_capability_mismatch"
    REMOTE_SUBMISSION_FAILED = "remote_submission_failed"
    REMOTE_STATUS_FAILED = "remote_status_failed"
    REMOTE_CANCEL_FAILED = "remote_cancel_failed"
    REMOTE_RESULT_FAILED = "remote_result_failed"
    REMOTE_ARTIFACT_DOWNLOAD_FAILED = "remote_artifact_download_failed"
    REMOTE_ARTIFACT_CORRUPT = "remote_artifact_corrupt"
    WORKER_BUSY = "worker_busy"
    GPU_UNAVAILABLE = "gpu_unavailable"
    UNKNOWN_ERROR = "unknown_error"
