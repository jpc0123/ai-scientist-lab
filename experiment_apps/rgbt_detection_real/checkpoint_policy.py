"""Checkpoint selection policy helpers (best-on-val + last)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_POLICY: dict[str, Any] = {
    "primary": "best_on_validation",
    "selection_metric": "mAP50_95",
    "selection_mode": "max",
    "tie_breaker": "earliest_epoch",
    "secondary": "last_epoch",
    "test_set_used_for_selection": False,
    "eval_frequency": "every_epoch",
}


def parse_checkpoint_policy(parameters: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(parameters or {}).get("checkpoint_policy")
    if raw is None:
        return dict(DEFAULT_CHECKPOINT_POLICY)
    if not isinstance(raw, dict):
        raise ValueError("parameters.checkpoint_policy must be a mapping")
    policy = dict(DEFAULT_CHECKPOINT_POLICY)
    policy.update(raw)
    if policy.get("primary") != "best_on_validation":
        raise ValueError(
            f"unsupported checkpoint_policy.primary={policy.get('primary')!r}; "
            "formal candidate requires best_on_validation"
        )
    if policy.get("selection_metric") != "mAP50_95":
        raise ValueError("formal candidate requires selection_metric=mAP50_95")
    if policy.get("selection_mode") != "max":
        raise ValueError("formal candidate requires selection_mode=max")
    if policy.get("tie_breaker") != "earliest_epoch":
        raise ValueError("formal candidate requires tie_breaker=earliest_epoch")
    if bool(policy.get("test_set_used_for_selection")):
        raise ValueError("test_set_used_for_selection must be false")
    return policy


def select_best_and_last_from_dfine_log(
    dfine_out: Path,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select best-on-val and last epoch from DFINE log.txt JSON lines.

    Selection uses strict greater-than on mAP50_95 so ties keep the earliest epoch.
    """
    policy = dict(policy or DEFAULT_CHECKPOINT_POLICY)
    log_txt = Path(dfine_out) / "log.txt"
    rows: list[dict[str, Any]] = []
    if log_txt.is_file():
        for line in log_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "epoch" not in payload:
                continue
            bbox = payload.get("test_coco_eval_bbox") or []
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
                continue
            rows.append(
                {
                    "epoch_0based": int(payload["epoch"]),
                    "epoch": int(payload["epoch"]) + 1,
                    "mAP50_95": float(bbox[0]),
                    "mAP50": float(bbox[1]),
                    "AP75": float(bbox[2]) if len(bbox) > 2 else None,
                    "APS": float(bbox[3]) if len(bbox) > 3 else None,
                    "train_loss": payload.get("train_loss"),
                    "loss_bbox": payload.get("train_loss_bbox"),
                    "loss_giou": payload.get("train_loss_giou"),
                    "lr": payload.get("train_lr"),
                }
            )

    if not rows:
        return {
            "policy": policy,
            "ok": False,
            "error": "no_epoch_rows_in_dfine_log",
            "best": None,
            "last": None,
            "best_minus_last_mAP50_95": None,
            "n_epochs": 0,
        }

    last = rows[-1]
    best = rows[0]
    for row in rows[1:]:
        if row["mAP50_95"] > best["mAP50_95"]:
            best = row

    return {
        "policy": policy,
        "ok": True,
        "error": None,
        "best": best,
        "last": last,
        "best_epoch": best["epoch"],
        "last_epoch": last["epoch"],
        "best_minus_last_mAP50_95": float(best["mAP50_95"] - last["mAP50_95"]),
        "n_epochs": len(rows),
        "rows": rows,
    }


def resolve_dfine_checkpoint_files(dfine_out: Path) -> dict[str, Path | None]:
    """Map DFINE native checkpoint names to Lab formal names."""
    out = Path(dfine_out)
    candidates_best = [
        out / "best_stg1.pth",
        out / "best_stg2.pth",
        out / "best.pth",
    ]
    candidates_last = [
        out / "last.pth",
        out / "checkpoint.pth",
    ]
    best = next((p for p in candidates_best if p.is_file()), None)
    last = next((p for p in candidates_last if p.is_file()), None)
    return {"best_native": best, "last_native": last}
