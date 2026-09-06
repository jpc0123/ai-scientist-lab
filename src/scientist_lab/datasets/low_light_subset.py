"""Frozen low-light condition slice. Not official night labels. LLM cannot rewrite it."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

SLICE_ID = "low_light_subset_v1"
SLICE_VERSION = "v1"
METHOD = "rgb_mean_rec709_luminance_train_p25"
TRAIN_PERCENTILE = 25.0
REC709 = (0.2126, 0.7152, 0.0722)
OFFICIAL_LABELS = False
DATASET_KEY = "rgbt_tiny_v1"

# Amendment only. Plans may not change these keys.
FROZEN_RULE_KEYS = (
    "slice_id",
    "version",
    "method",
    "train_percentile",
    "rec709",
    "official_labels",
    "threshold_split",
)


class SliceAmendmentRequired(ValueError):
    """Changing low_light_subset_v1 requires a Protocol Amendment."""


def frozen_rule() -> dict[str, Any]:
    return {
        "slice_id": SLICE_ID,
        "version": SLICE_VERSION,
        "method": METHOD,
        "train_percentile": TRAIN_PERCENTILE,
        "rec709": list(REC709),
        "official_labels": OFFICIAL_LABELS,
        "threshold_split": "train",
        "dataset": f"dataset:{DATASET_KEY}",
        "note": (
            "RGBT-Tiny has no day/night field. Membership is deterministic RGB "
            "mean Rec.709 luminance at or below the train-split percentile. "
            "Not an official low-light label. LLM must not pick a favorable subset."
        ),
    }


def rule_hash(rule: Mapping[str, Any] | None = None) -> str:
    body = dict(rule or frozen_rule())
    payload = {key: body[key] for key in FROZEN_RULE_KEYS if key in body}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def membership_hash(sample_ids: Iterable[str]) -> str:
    ordered = sorted({str(item) for item in sample_ids if str(item).strip()})
    blob = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def linear_percentile(values: Sequence[float], percentile: float) -> float:
    """Numpy-default linear interpolation. Frozen; do not swap algorithms."""
    if not values:
        raise ValueError("percentile of empty luminance list")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    p = min(100.0, max(0.0, float(percentile)))
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def mean_rec709_luminance(path: Path | str) -> float:
    import numpy as np

    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"empty image: {path}")
    wr, wg, wb = REC709
    return float((wr * arr[..., 0] + wg * arr[..., 1] + wb * arr[..., 2]).mean() / 255.0)


def default_registered_root() -> Path:
    return Path(__file__).resolve().parents[3] / "datasets" / "registered" / DATASET_KEY


def load_pairs(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def resolve_rgb_path(row: Mapping[str, str], *, registered_root: Path) -> Path:
    raw = Path(str(row.get("rgb_path") or ""))
    if raw.is_file():
        return raw
    split = str(row.get("split") or "val")
    name = str(row.get("file_name") or "")
    alt = registered_root / "images" / split / "rgb" / name
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"RGB image missing for {row.get('sample_id')}: {raw}")


def _pairs_by_split(registered_root: Path) -> dict[str, list[dict[str, str]]]:
    manifests = registered_root / "manifests"
    out: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        path = manifests / f"{split}_pairs.csv"
        if path.is_file():
            out[split] = load_pairs(path)
    if "train" not in out:
        raise FileNotFoundError(f"missing train pairs: {manifests / 'train_pairs.csv'}")
    return out


def build_low_light_subset(
    registered_root: Path | str | None = None,
    *,
    rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute membership from RGB images. Threshold is fit on train only."""
    root = Path(registered_root) if registered_root is not None else default_registered_root()
    spec = dict(rule or frozen_rule())
    if spec.get("slice_id") != SLICE_ID or spec.get("method") != METHOD:
        raise SliceAmendmentRequired("refusing a non-frozen slice rule")
    pairs = _pairs_by_split(root)
    train_luma: list[tuple[str, float]] = []
    for row in pairs["train"]:
        sample_id = str(row.get("sample_id") or "").strip()
        luma = mean_rec709_luminance(resolve_rgb_path(row, registered_root=root))
        train_luma.append((sample_id, luma))
    threshold = linear_percentile([item[1] for item in train_luma], TRAIN_PERCENTILE)

    members: dict[str, list[str]] = {}
    luminances: dict[str, dict[str, float]] = {}
    for split, rows in pairs.items():
        ids: list[str] = []
        split_luma: dict[str, float] = {}
        for row in rows:
            sample_id = str(row.get("sample_id") or "").strip()
            if split == "train":
                luma = dict(train_luma)[sample_id]
            else:
                luma = mean_rec709_luminance(resolve_rgb_path(row, registered_root=root))
            split_luma[sample_id] = luma
            if luma <= threshold:
                ids.append(sample_id)
        members[split] = sorted(ids)
        luminances[split] = split_luma

    all_ids = [item for split in ("train", "val", "test") for item in members.get(split, [])]
    freeze = {
        "schema_version": "1.0.0",
        "slice_id": SLICE_ID,
        "official_labels": False,
        "rule": spec,
        "rule_hash": rule_hash(spec),
        "threshold": threshold,
        "membership": members,
        "counts": {split: len(ids) for split, ids in members.items()},
        "membership_hash": membership_hash(all_ids),
        "dataset_root": "datasets/registered/rgbt_tiny_v1",
        "llm_may_rewrite": False,
        "amendment": "Protocol Amendment required to change this slice",
    }
    freeze["luminance_summary"] = {
        split: {
            "n": len(vals),
            "min": min(vals.values()) if vals else None,
            "max": max(vals.values()) if vals else None,
        }
        for split, vals in luminances.items()
    }
    return freeze


def subset_ids(freeze: Mapping[str, Any], split: str | None = None) -> set[str]:
    membership = dict(freeze.get("membership") or {})
    if split:
        return set(membership.get(split) or [])
    out: set[str] = set()
    for ids in membership.values():
        out.update(str(item) for item in ids)
    return out


_SLICE_SCOPE = frozenset(
    {"dataset_split", "evaluator", "metric_definition", "condition_slice", "low_light_subset"}
)


def assert_slice_not_rewritten(
    plan: Mapping[str, Any] | None,
    *,
    freeze: Mapping[str, Any] | None = None,
) -> None:
    payload = dict(plan or {})
    scope = [str(tok) for tok in (payload.get("modification_scope") or [])]
    if any(tok in _SLICE_SCOPE for tok in scope):
        raise SliceAmendmentRequired(
            "LLM cannot change low_light_subset_v1; Protocol Amendment required"
        )
    for row in payload.get("proposed_changes") or []:
        if not isinstance(row, Mapping):
            continue
        target = str(row.get("target") or "")
        if target in _SLICE_SCOPE:
            raise SliceAmendmentRequired(
                "LLM cannot change low_light_subset_v1; Protocol Amendment required"
            )
    proposed = freeze.get("rule_hash") if freeze else None
    if proposed and proposed != rule_hash():
        raise SliceAmendmentRequired("slice rule_hash mismatch vs frozen low_light_subset_v1")


def write_freeze(freeze: Mapping[str, Any], path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(freeze), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest


def load_freeze(path: Path | str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("low-light freeze must be an object")
    if raw.get("slice_id") != SLICE_ID:
        raise SliceAmendmentRequired(f"unexpected slice_id={raw.get('slice_id')!r}")
    if raw.get("llm_may_rewrite") is True:
        raise SliceAmendmentRequired("freeze must keep llm_may_rewrite=false")
    if raw.get("rule_hash") != rule_hash(raw.get("rule") or frozen_rule()):
        raise SliceAmendmentRequired("stored rule_hash does not match frozen rule")
    return raw
