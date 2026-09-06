"""Dataset Workspace: registry / slices / contracts. Not an Agent.

Git stores identity (registry JSON, split ids, checksums).
Disk stores image bodies under raw/ processed/ (gitignored) or existing registered roots.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.datasets.low_light_subset import (
    SLICE_ID,
    membership_hash,
    rule_hash,
)
from scientist_lab.datasets.registry import parse_dataset_reference

WORKSPACE_NAME = "data"
SCHEMA_VERSION = "1.0.0"


class DatasetContractError(ValueError):
    """Dataset / slice is missing, unfrozen, or not consumable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def default_workspace_root(project_root: Path | str | None = None) -> Path:
    if project_root is not None:
        return Path(project_root) / WORKSPACE_NAME
    return Path(__file__).resolve().parents[3] / WORKSPACE_NAME


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DatasetContractError(f"expected object: {path}")
    return raw


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(item) for item in ids if str(item).strip())
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _exists(path: Path | None) -> bool:
    return bool(path) and Path(path).exists()


class DatasetWorkspace:
    """Filesystem Dataset Registry. LLM may select ids; it may not rewrite slices."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.registry_dir = self.root / "registry"
        self.slices_dir = self.root / "slices"
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.cache_dir = self.root / "cache"

    @classmethod
    def from_project(cls, project_root: Path | str | None = None) -> "DatasetWorkspace":
        return cls(default_workspace_root(project_root))

    def ensure_layout(self) -> None:
        for folder in (self.registry_dir, self.slices_dir, self.raw_dir, self.processed_dir, self.cache_dir):
            folder.mkdir(parents=True, exist_ok=True)
        for folder in (self.raw_dir, self.processed_dir, self.cache_dir):
            keep = folder / ".gitkeep"
            if not keep.exists():
                keep.write_text("", encoding="utf-8")

    def dataset_path(self, dataset_id: str) -> Path:
        return self.registry_dir / f"{dataset_id}.json"

    def slice_dir(self, slice_id: str) -> Path:
        return self.slices_dir / slice_id

    def list_dataset_ids(self) -> list[str]:
        if not self.registry_dir.is_dir():
            return []
        return sorted(path.stem for path in self.registry_dir.glob("*.json"))

    def list_slice_ids(self) -> list[str]:
        if not self.slices_dir.is_dir():
            return []
        return sorted(
            path.name
            for path in self.slices_dir.iterdir()
            if path.is_dir() and (path / "slice_spec.json").is_file()
        )

    def load_dataset(self, dataset_id: str) -> dict[str, Any]:
        path = self.dataset_path(dataset_id)
        if not path.is_file():
            raise DatasetContractError(f"dataset not registered: {dataset_id}")
        doc = _read_json(path)
        if str(doc.get("dataset_id") or "") != dataset_id:
            raise DatasetContractError(f"dataset_id mismatch in {path}")
        return doc

    def load_slice(self, slice_id: str) -> dict[str, Any]:
        spec_path = self.slice_dir(slice_id) / "slice_spec.json"
        if not spec_path.is_file():
            raise DatasetContractError(f"slice not registered: {slice_id}")
        spec = _read_json(spec_path)
        if str(spec.get("slice_id") or "") != slice_id:
            raise DatasetContractError(f"slice_id mismatch in {spec_path}")
        if spec.get("llm_may_rewrite") is True or spec.get("frozen") is False:
            raise DatasetContractError(f"slice {slice_id} is not frozen")
        ids = {
            split: _read_ids(self.slice_dir(slice_id) / f"{split}_ids.txt")
            for split in ("train", "val", "test")
        }
        spec = dict(spec)
        spec["membership"] = ids
        spec["counts"] = {split: len(rows) for split, rows in ids.items()}
        all_ids = [item for split in ("train", "val", "test") for item in ids[split]]
        spec["membership_hash_computed"] = membership_hash(all_ids)
        stored = spec.get("membership_hash")
        if stored and stored != spec["membership_hash_computed"]:
            raise DatasetContractError(
                f"slice {slice_id} membership_hash mismatch; refusing mutated id lists"
            )
        return spec

    def probe_dataset(self, doc: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
        raw = doc.get("raw_root")
        processed = doc.get("processed_root")
        raw_path = Path(str(raw)) if raw else None
        processed_path = Path(str(processed)) if processed else None
        if processed_path and not processed_path.is_absolute():
            processed_path = project_root / processed_path
        if raw_path and not raw_path.is_absolute():
            raw_path = project_root / raw_path
        return {
            "raw_present": _exists(raw_path),
            "processed_present": _exists(processed_path),
            "raw_root": str(raw_path) if raw_path else None,
            "processed_root": str(processed_path) if processed_path else None,
            "in_git": False,
        }

    def resolve(
        self,
        dataset_id: str,
        *,
        slice_id: str | None = None,
        project_root: Path | str | None = None,
    ) -> dict[str, Any]:
        ds = self.load_dataset(dataset_id)
        root = Path(project_root) if project_root is not None else self.root.parent
        probe = self.probe_dataset(ds, project_root=root)
        slice_doc = None
        split_reference = str(ds.get("split_reference") or "official_split")
        if slice_id:
            slice_doc = self.load_slice(slice_id)
            parent = str(slice_doc.get("parent_dataset") or "").replace("dataset:", "")
            if parent != dataset_id:
                raise DatasetContractError(
                    f"slice {slice_id} parent_dataset={parent!r} does not match {dataset_id}"
                )
            if not slice_doc.get("frozen"):
                raise DatasetContractError(f"slice {slice_id} is not frozen")
            split_reference = f"{slice_id}@{slice_doc.get('rule_hash') or slice_doc.get('membership_hash')}"
        fingerprint = _stable_hash(
            {
                "dataset_id": dataset_id,
                "version": ds.get("version"),
                "manifest_hashes": ds.get("manifest_hashes"),
                "annotation_hashes": ds.get("annotation_hashes"),
                "split_reference": split_reference,
                "slice_id": slice_id,
                "slice_rule_hash": (slice_doc or {}).get("rule_hash"),
                "slice_membership_hash": (slice_doc or {}).get("membership_hash"),
            }
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "dataset_ref": f"dataset:{dataset_id}",
            "task": ds.get("task") or "rgbt_object_detection",
            "version": ds.get("version"),
            "slice_id": slice_id,
            "split_reference": split_reference,
            "fingerprint": fingerprint,
            "annotation_format": ds.get("annotation_format") or "coco",
            "splits": ds.get("splits") or {},
            "adapters": list(ds.get("adapters") or ["dfine"]),
            "read_only": True,
            "llm_may_rewrite": False,
            "llm_may_select": True,
            "probe": probe,
            "slice": None
            if slice_doc is None
            else {
                "slice_id": slice_doc.get("slice_id"),
                "frozen": True,
                "official_labels": bool(slice_doc.get("official_labels")),
                "counts": slice_doc.get("counts"),
                "rule_hash": slice_doc.get("rule_hash"),
                "membership_hash": slice_doc.get("membership_hash"),
                "condition": slice_doc.get("condition"),
                "method": slice_doc.get("method") or slice_doc.get("selection_rule"),
            },
        }

    def overview(self, *, project_root: Path | str | None = None) -> dict[str, Any]:
        root = Path(project_root) if project_root is not None else self.root.parent
        datasets = []
        for dataset_id in self.list_dataset_ids():
            doc = self.load_dataset(dataset_id)
            parent_slices = []
            for sid in self.list_slice_ids():
                parent = str(self.load_slice(sid).get("parent_dataset") or "").replace("dataset:", "")
                if parent == dataset_id:
                    parent_slices.append(sid)
            datasets.append(
                {
                    "dataset_id": doc.get("dataset_id"),
                    "task": doc.get("task"),
                    "version": doc.get("version"),
                    "annotation_format": doc.get("annotation_format"),
                    "probe": self.probe_dataset(doc, project_root=root),
                    "slices": parent_slices,
                }
            )
        slices = []
        for slice_id in self.list_slice_ids():
            spec = self.load_slice(slice_id)
            slices.append(
                {
                    "slice_id": spec.get("slice_id"),
                    "parent_dataset": spec.get("parent_dataset"),
                    "frozen": True,
                    "official_labels": spec.get("official_labels"),
                    "counts": spec.get("counts"),
                    "rule_hash": spec.get("rule_hash"),
                    "membership_hash": spec.get("membership_hash"),
                    "condition": spec.get("condition"),
                }
            )
        return {
            "workspace_root": str(self.root),
            "layout": ["raw", "processed", "slices", "registry", "cache"],
            "git_policy": "Git stores registry/slice ids/checksums. Image bodies stay on disk.",
            "permissions": {
                "planner": "may select dataset_id / slice_id; cannot rewrite labels or membership",
                "adapter": "translates Dataset Contract into model loader config; cannot swap splits",
                "runner": "read-only freeze; writes checkpoints/logs/predictions only",
            },
            "datasets": datasets,
            "slices": slices,
            "not_an_agent": True,
        }

    def import_slice_from_freeze(
        self,
        freeze: Mapping[str, Any],
        *,
        parent_dataset: str = "rgbt_tiny_v1",
    ) -> dict[str, Any]:
        slice_id = str(freeze.get("slice_id") or SLICE_ID)
        if freeze.get("llm_may_rewrite") is True or freeze.get("frozen") is False:
            raise DatasetContractError("refusing unfrozen freeze document")
        membership = dict(freeze.get("membership") or {})
        if not any(membership.get(split) for split in ("train", "val", "test")):
            raise DatasetContractError("refusing import of empty / rule-only freeze")
        rule = dict(freeze.get("rule") or {})
        parent = parent_dataset
        if rule.get("dataset"):
            try:
                parent = parse_dataset_id(str(rule["dataset"]))
            except DatasetContractError:
                parent = parent_dataset
        dest = self.slice_dir(slice_id)
        dest.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            _write_ids(dest / f"{split}_ids.txt", list(membership.get(split) or []))
        all_ids = [item for split in ("train", "val", "test") for item in membership.get(split) or []]
        computed = membership_hash(all_ids)
        stored = freeze.get("membership_hash")
        if stored and stored != computed:
            raise DatasetContractError(
                f"freeze membership_hash mismatch for {slice_id}: refusing mutated id lists"
            )
        spec = {
            "schema_version": SCHEMA_VERSION,
            "slice_id": slice_id,
            "parent_dataset": parent,
            "condition": "low_light",
            "official_labels": False,
            "selection_rule": (freeze.get("rule") or {}).get("method")
            or "rgb_mean_rec709_luminance_train_p25",
            "method": (freeze.get("rule") or {}).get("method"),
            "threshold": freeze.get("threshold"),
            "train_percentile": (freeze.get("rule") or {}).get("train_percentile"),
            "rule_hash": freeze.get("rule_hash") or rule_hash(freeze.get("rule")),
            "membership_hash": freeze.get("membership_hash") or membership_hash(all_ids),
            "frozen": True,
            "llm_may_rewrite": False,
            "created_at": utc_now(),
            "amendment": "Protocol Amendment required to change this slice",
            "note": (
                "Not an official night label. Membership is deterministic RGB Rec.709 "
                "luminance at or below the train-split percentile."
            ),
        }
        _write_json(dest / "slice_spec.json", spec)
        return spec


def parse_dataset_id(reference: str) -> str:
    key = parse_dataset_reference(reference)
    if key:
        return key
    text = str(reference or "").strip()
    if not text:
        raise DatasetContractError("empty dataset reference")
    return text
