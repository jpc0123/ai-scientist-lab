"""List pagination helpers (v1.7.1)."""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

T = TypeVar("T")


def clamp_limit(limit: int | None, *, default: int = 50, max_limit: int = 200) -> int:
    value = default if limit is None else int(limit)
    if value < 1:
        return 1
    return min(value, max_limit)


def clamp_offset(offset: int | None) -> int:
    value = 0 if offset is None else int(offset)
    return max(0, value)


def page_items(items: Sequence[T], *, limit: int, offset: int) -> list[T]:
    return list(items[offset : offset + limit])


def page_response(
    items: Sequence[Any],
    *,
    limit: int,
    offset: int,
    total: int | None = None,
) -> dict[str, Any]:
    sliced = page_items(items, limit=limit, offset=offset)
    return {
        "items": sliced,
        "limit": limit,
        "offset": offset,
        "total": len(items) if total is None else total,
        "count": len(sliced),
    }
