from __future__ import annotations

from enum import StrEnum


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


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
    CUDA_UNAVAILABLE = "cuda_unavailable"
    TRAINING_DIVERGED = "training_diverged"
    NAN_LOSS = "nan_loss"
    EVALUATION_FAILED = "evaluation_failed"
    INVALID_DETECTION_METRICS = "invalid_detection_metrics"
    UNKNOWN_ERROR = "unknown_error"
