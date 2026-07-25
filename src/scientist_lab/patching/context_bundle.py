"""Build restricted CodeContextBundle (v2.2.1) — no LLM / network."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from scientist_lab.patching.context_models import (
    AllowedSourceFile,
    CodeContextBundle,
    ContextSizeBudget,
    PatchRequest,
    SourceSnapshot,
)
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.storage.artifact_store import sha256_file


class ContextBundleError(ValueError):
    """Raised when a CodeContextBundle cannot be built safely."""


def _stable_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_context_sha256(bundle: CodeContextBundle | dict[str, Any]) -> str:
    """Deterministic hash of context contents (excludes volatile ids/timestamps)."""
    if isinstance(bundle, CodeContextBundle):
        data = bundle.model_dump(mode="json")
    else:
        data = dict(bundle)
    for key in ("context_sha256", "bundle_id", "created_at"):
        data.pop(key, None)
    return _stable_json_sha256(data)


def resolve_source_commit(project_root: Path, explicit: str | None = None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    try:
        from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError

        adapter = GitAdapter(repo_root=project_root)
        return adapter.rev_parse("HEAD")
    except Exception:
        return ""


def _read_text_capped(path: Path, *, max_bytes: int) -> tuple[str, int, bool]:
    raw = path.read_bytes()
    size = len(raw)
    truncated = False
    if size > max_bytes:
        raw = raw[:max_bytes]
        truncated = True
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContextBundleError(f"non-utf8 file forbidden: {path}") from exc
    return text, size, truncated


def build_code_context_bundle(
    request: PatchRequest,
    *,
    project_root: Path,
    policy: PathPolicy | None = None,
    budget: ContextSizeBudget | None = None,
    bundle_id: str | None = None,
) -> CodeContextBundle:
    """Pack allowed source files into a reproducible CodeContextBundle.

    Never calls a provider. Only reads paths that pass PathPolicy.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ContextBundleError(f"project_root is not a directory: {root}")

    path_policy = policy or PathPolicy.for_code_context()
    size_budget = budget or ContextSizeBudget()
    if not request.allowed_files:
        raise ContextBundleError("PatchRequest.allowed_files is empty")

    if len(request.allowed_files) > size_budget.max_files:
        raise ContextBundleError(
            f"too many allowed files: {len(request.allowed_files)} > "
            f"max_files={size_budget.max_files}"
        )

    source_commit = resolve_source_commit(root, request.source_commit)
    snapshots: list[SourceSnapshot] = []
    excluded: list[dict[str, str]] = []
    total_bytes = 0
    seen: set[str] = set()

    for item in request.allowed_files:
        norm = path_policy.normalize(item.path)
        if not norm or norm in seen:
            if norm in seen:
                excluded.append({"path": norm, "reason": "duplicate path"})
            continue
        seen.add(norm)

        ok, reason = path_policy.is_allowed(norm)
        if not ok:
            excluded.append({"path": norm, "reason": reason})
            continue

        abs_path = (root / norm).resolve()
        try:
            abs_path.relative_to(root)
        except ValueError:
            excluded.append({"path": norm, "reason": "path escapes project_root"})
            continue

        if not abs_path.is_file():
            excluded.append({"path": norm, "reason": "file not found"})
            continue

        content, size_bytes, truncated = _read_text_capped(
            abs_path, max_bytes=size_budget.max_bytes_per_file
        )
        encoded_len = len(content.encode("utf-8"))
        if total_bytes + encoded_len > size_budget.max_bytes_total:
            excluded.append({"path": norm, "reason": "context size budget exceeded"})
            continue

        if truncated:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            content_hash = sha256_file(abs_path)

        snapshots.append(
            SourceSnapshot(
                path=norm,
                content=content,
                content_sha256=content_hash,
                size_bytes=size_bytes,
                truncated=truncated,
            )
        )
        total_bytes += len(content.encode("utf-8"))

    if not snapshots:
        raise ContextBundleError(
            "no allowed source files could be snapshotted; "
            f"excluded={excluded}"
        )

    bundle = CodeContextBundle(
        bundle_id=bundle_id or f"ctx_{uuid.uuid4().hex[:12]}",
        request_id=request.request_id,
        project_id=request.project_id,
        source_commit=source_commit,
        goal=request.goal,
        failure_summary=request.failure_summary,
        test_errors=list(request.test_errors),
        evidence_gap_ids=list(request.evidence_gap_ids),
        patch_target=request.patch_target,
        interface_notes=list(request.interface_notes),
        allowed_files=list(request.allowed_files),
        snapshots=snapshots,
        excluded_paths=excluded,
        budget=size_budget,
        total_bytes=total_bytes,
        metadata=dict(request.metadata or {}),
    )
    bundle.touch()
    bundle.context_sha256 = code_context_sha256(bundle)
    return bundle


def digits_improvement_patch_request(
    *,
    request_id: str | None = None,
    project_id: str = "digits_context_v22",
    source_commit: str | None = None,
) -> PatchRequest:
    """Fixture-style Digits small-improvement PatchRequest (offline)."""
    return PatchRequest(
        request_id=request_id or f"preq_{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        goal="Small improvement: make Digits entrypoint failure summary clearer for patch context.",
        failure_summary=(
            "Accuracy plateau on Digits MLP; need a tightly scoped code change "
            "inside experiment_app/run_experiment.py only."
        ),
        test_errors=[
            "synthetic: mean accuracy below local target on seed=0 fixture run",
        ],
        evidence_gap_ids=["gap_digits_entrypoint_logging"],
        patch_target="experiment_app/run_experiment.py",
        allowed_files=[
            AllowedSourceFile(
                path="experiment_app/run_experiment.py",
                reason="Digits real entrypoint; only file in scope for this request",
                role="entrypoint",
            ),
        ],
        interface_notes=[
            "CLI argparse: --config, --seed, --output-dir",
            "Primary metric written to metrics.json as accuracy",
            "Do not touch Dockerfiles, .env, or scientist_lab.llm",
        ],
        source_commit=source_commit,
        metadata={"demo": "digits_v22_1", "provider_call": False},
    )
