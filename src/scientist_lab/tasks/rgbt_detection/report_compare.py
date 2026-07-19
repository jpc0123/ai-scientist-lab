from __future__ import annotations

from typing import Any


COMPARE_KEYS = (
    "valid",
    "paired_image_count",
    "rgb_image_count",
    "thermal_image_count",
    "claim_level",
)


def _split_counts(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    splits = report.get("splits") or {}
    out: dict[str, dict[str, int]] = {}
    for name, payload in splits.items():
        if not isinstance(payload, dict):
            continue
        out[name] = {
            "rgb_count": int(payload.get("rgb_count") or 0),
            "thermal_count": int(payload.get("thermal_count") or 0),
            "paired_count": int(payload.get("paired_count") or 0),
        }
    return out


def _stem_set(values: list[Any] | None) -> set[str]:
    result: set[str] = set()
    for item in values or []:
        text = str(item)
        # host may store "train:000001", container same — compare normalized tails
        result.add(text.split(":")[-1] if ":" in text else text)
    return result


def compare_host_container_reports(
    host_report: dict[str, Any],
    container_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare key validation fields between host and container reports."""
    mismatches: list[str] = []

    for key in COMPARE_KEYS:
        if host_report.get(key) != container_report.get(key):
            mismatches.append(
                f"{key}: host={host_report.get(key)!r} container={container_report.get(key)!r}"
            )

    host_splits = _split_counts(host_report)
    container_splits = _split_counts(container_report)
    if host_splits != container_splits:
        mismatches.append(
            f"splits: host={host_splits} container={container_splits}"
        )

    for field in ("missing_rgb", "missing_thermal"):
        host_set = _stem_set(host_report.get(field))
        container_set = _stem_set(container_report.get(field))
        if host_set != container_set:
            mismatches.append(
                f"{field}: host={sorted(host_set)[:10]} container={sorted(container_set)[:10]}"
            )

    mount_probe = container_report.get("mount_probe") or {}
    if container_report.get("trained") is True:
        mismatches.append("container report marks trained=True for validate_data")
    if mount_probe and mount_probe.get("dataset_writable") is True:
        mismatches.append("container dataset mount is writable")

    return {
        "schema_version": "1.0",
        "consistent": not mismatches,
        "mismatches": mismatches,
        "host_valid": bool(host_report.get("valid")),
        "container_valid": bool(container_report.get("valid")),
        "mount_probe": mount_probe,
    }
