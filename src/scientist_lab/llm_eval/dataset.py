"""Load versioned LLM evaluation suites from JSONL + manifest."""

from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.llm_eval.models import (
    EvaluationCase,
    EvaluationSuite,
    EvaluationSuiteManifest,
)


DEFAULT_EVALS_ROOT = Path(__file__).resolve().parents[3] / "evals" / "llm"


class DatasetError(ValueError):
    """Raised when an evaluation suite cannot be loaded or validated."""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def resolve_suite_manifest_path(
    suite: str | Path,
    *,
    evals_root: Path | None = None,
) -> Path:
    root = Path(evals_root) if evals_root is not None else DEFAULT_EVALS_ROOT
    path = Path(suite)
    if path.is_file():
        return path.resolve()
    name = str(suite).strip()
    if name.endswith(".json"):
        candidate = root / "manifests" / name
    else:
        candidate = root / "manifests" / f"{name}.json"
    if not candidate.is_file():
        raise DatasetError(f"suite manifest not found: {candidate}")
    return candidate.resolve()


def load_evaluation_suite(
    suite: str | Path = "eval_suite_v1",
    *,
    evals_root: Path | None = None,
    validate: bool = True,
) -> EvaluationSuite:
    """Load a versioned suite. Paths in the manifest are relative to evals/llm."""
    root = Path(evals_root) if evals_root is not None else DEFAULT_EVALS_ROOT
    manifest_path = resolve_suite_manifest_path(suite, evals_root=root)
    raw = _read_json(manifest_path)
    manifest = EvaluationSuiteManifest.model_validate(raw)

    cases: list[EvaluationCase] = []
    missing: list[str] = []
    for rel in manifest.case_files:
        path = (root / rel).resolve()
        if not path.is_file():
            missing.append(rel)
            continue
        for row in _read_jsonl(path):
            cases.append(EvaluationCase.model_validate(row))

    if validate:
        if missing:
            raise DatasetError(
                "suite manifest references missing case files: " + ", ".join(missing)
            )
        for rel in manifest.expected_files:
            path = (root / rel).resolve()
            if not path.is_file():
                raise DatasetError(f"expected file missing: {rel}")
        counts: dict[str, int] = {}
        for case in cases:
            counts[case.task_type] = counts.get(case.task_type, 0) + 1
        for task, expected in (manifest.case_counts or {}).items():
            actual = counts.get(task, 0)
            if actual != int(expected):
                raise DatasetError(
                    f"case_counts mismatch for {task}: expected {expected}, got {actual}"
                )
        ids = [c.case_id for c in cases]
        if len(ids) != len(set(ids)):
            raise DatasetError("duplicate case_id in suite")

    return EvaluationSuite(manifest=manifest, cases=cases)
