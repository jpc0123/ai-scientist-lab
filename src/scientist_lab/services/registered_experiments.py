"""Registered detection experiments: one protocol + adapter + catalog per id.

The V26 RGB-T D-FINE / APS_lowlight protocol is the built-in experiment, not
the only runtime. New experiments must register their own protocol. This is
not a Claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.how_catalog import CATALOG_ID, plan_how_id
from scientist_lab.adapters.registry import adapter_for_protocol
from scientist_lab.core.schema_registry import SCHEMA_DIR, SchemaValidationError, load_json, validate_named

BUILTIN_EXPERIMENT_ID = "exp_rgbt_dfine_v26_lowlight"
BUILTIN_PROTOCOL = SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json"
BUILTIN_PLAN = SCHEMA_DIR / "examples" / "experiment_plan_rgbt_dfine_v26_r0.json"

BUILTIN_RTDETR_TRANSFER_ID = "exp_rgbt_rtdetr_v26_transfer"
BUILTIN_RTDETR_PROTOCOL = SCHEMA_DIR / "examples" / "research_protocol_rgbt_rtdetr_transfer_v26.json"
BUILTIN_RTDETR_PLAN = SCHEMA_DIR / "examples" / "experiment_plan_rgbt_rtdetr_p4_f1.json"

SUPPORTED_TASK_ALIASES = {
    "object_detection": "object_detection",
    "detection": "object_detection",
    "rgbt_detection": "object_detection",
}
SUPPORTED_ADAPTERS = {"dfine", "rtdetr", "rt_detr"}


class RegisteredExperimentError(ValueError):
    """Illegal experiment register / bind request."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def protocol_content_hash(protocol: Mapping[str, Any]) -> str:
    blob = json.dumps(protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    named = str(protocol.get("fingerprint_id") or "").strip()
    digest = protocol_content_hash(protocol)
    return f"{named}:{digest}" if named else digest


def canonicalize_task_type(raw: str | None) -> str:
    token = str(raw or "").strip().lower().replace("-", "_")
    return SUPPORTED_TASK_ALIASES.get(token, token or "unsupported")


def parse_dataset_id(raw: str | None) -> str:
    text = str(raw or "").strip()
    if text.startswith("dataset:"):
        return text.split(":", 1)[1].strip()
    return text


def primary_metric_token(protocol: Mapping[str, Any]) -> str:
    primary = dict(((protocol or {}).get("objective") or {}).get("primary") or {})
    return str(primary.get("metric") or "").strip()


def slice_id_token(protocol: Mapping[str, Any]) -> str | None:
    token = str(((protocol or {}).get("condition_slice") or {}).get("id") or "").strip()
    return token or None


def science_identity(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Idle-loop fingerprint. Adapter-only changes do not count as a new experiment."""
    how = plan_how_id(plan or {}) if plan else None
    return {
        "dataset_id": parse_dataset_id(
            str(((protocol or {}).get("baseline") or {}).get("dataset") or "")
        ),
        "slice_id": slice_id_token(protocol),
        "primary_metric": primary_metric_token(protocol),
        "seed_how_id": str(how).strip().upper() if how else None,
    }


def builtin_science_identity() -> dict[str, Any]:
    return science_identity(load_json(BUILTIN_PROTOCOL), load_json(BUILTIN_PLAN))


def same_science_as_builtin(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
) -> bool:
    return science_identity(protocol, plan) == builtin_science_identity()


def sanitize_experiment_id(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(raw or "").strip()).strip("._-")
    if not slug:
        raise RegisteredExperimentError("experiment_id is empty")
    if not slug.lower().startswith("exp_"):
        slug = f"exp_{slug}"
    return slug[:80]


def default_experiment_id(protocol: Mapping[str, Any]) -> str:
    return sanitize_experiment_id(str(protocol.get("protocol_id") or "experiment"))


def adapter_token(protocol: Mapping[str, Any]) -> str:
    raw = str(((protocol or {}).get("baseline") or {}).get("adapter") or "dfine")
    return raw.strip().lower().replace("-", "_") or "dfine"


def assert_seed_how_materializable(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed if Adapter cannot materialize this experiment's seed HOW.

    Does not invent F1 early_concat. Does not run GPU.
    """
    if not plan:
        raise RegisteredExperimentError(
            "该实验没有 seed plan，拒绝启动战役；不会悄悄改成 F1 early_concat"
        )
    how_id = plan_how_id(plan)
    if not how_id:
        raise RegisteredExperimentError(
            "seed plan 缺少 how_id，拒绝启动；不会由 Adapter 发明 F1"
        )
    token = adapter_token(protocol)
    if token not in SUPPORTED_ADAPTERS:
        raise RegisteredExperimentError(
            f"Adapter {token!r} 不在可物化名单 {sorted(SUPPORTED_ADAPTERS)} 内，"
            "拒绝启动，不会悄悄映射成 D-FINE F1 early_concat"
        )
    adapter = adapter_for_protocol(protocol)
    try:
        how = adapter._resolve_how(plan, protocol)  # noqa: SLF001 — Adapter contract, no GPU
    except MaterializeRejected as exc:
        raise RegisteredExperimentError(
            f"Adapter 不能物化该实验的 seed HOW {how_id}：{exc}。"
            "拒绝启动，不会悄悄改成 F1 early_concat"
        ) from exc
    return dict(how)


class RegisteredExperimentService:
    """File-backed experiment registry under ``.run/experiments``."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)
        self.root = self.project_root / ".run" / "experiments"

    def _builtin_canonical(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return load_json(BUILTIN_PROTOCOL), load_json(BUILTIN_PLAN)

    def _builtin_needs_refresh(self, existing: Mapping[str, Any]) -> bool:
        protocol, plan = self._builtin_canonical()
        protocol_path = Path(str(existing.get("protocol_path") or ""))
        if not protocol_path.is_file():
            return True
        stored_protocol = load_json(protocol_path)
        if protocol_content_hash(stored_protocol) != protocol_content_hash(protocol):
            return True
        plan_path = Path(str(existing.get("seed_plan_path") or ""))
        if not plan_path.is_file():
            return True
        stored_plan = load_json(plan_path)
        return protocol_content_hash(stored_plan) != protocol_content_hash(plan)

    def ensure_builtin(self) -> dict[str, Any]:
        protocol, plan = self._builtin_canonical()
        notes = (
            "Built-in V26 RGB-T D-FINE / APS_lowlight / low_light_subset_v1. "
            "This is one registered experiment, not the only runtime. "
            "KEEP ≠ Claim. This document is not a Claim."
        )
        existing = self.get(BUILTIN_EXPERIMENT_ID)
        if existing is not None and not self._builtin_needs_refresh(existing):
            primary = existing
        else:
            primary = self._write_record(
                experiment_id=BUILTIN_EXPERIMENT_ID,
                protocol=protocol,
                plan=plan,
                builtin=True,
                catalog_id=CATALOG_ID,
                notes=notes,
            )
        self.ensure_rtdetr_transfer()
        return primary

    def ensure_rtdetr_transfer(self) -> dict[str, Any]:
        protocol = load_json(BUILTIN_RTDETR_PROTOCOL)
        plan = load_json(BUILTIN_RTDETR_PLAN)
        notes = (
            "Built-in V26 RT-DETR transfer / APS_lowlight / same low_light_subset_v1. "
            "G3 cross-detector comparison requires this separate registered experiment. "
            "KEEP ≠ Claim."
        )
        existing = self.get(BUILTIN_RTDETR_TRANSFER_ID)
        if existing is not None:
            protocol_path = Path(str(existing.get("protocol_path") or ""))
            if protocol_path.is_file():
                stored = load_json(protocol_path)
                if protocol_content_hash(stored) == protocol_content_hash(protocol):
                    return existing
        return self._write_record(
            experiment_id=BUILTIN_RTDETR_TRANSFER_ID,
            protocol=protocol,
            plan=plan,
            builtin=True,
            catalog_id=CATALOG_ID,
            notes=notes,
        )

    def list(self) -> dict[str, Any]:
        self.ensure_builtin()
        items: list[dict[str, Any]] = []
        if self.root.is_dir():
            for path in sorted(self.root.glob("*/experiment.json")):
                items.append(self._public(_load_record(path)))
        items.sort(key=lambda row: (not bool(row.get("builtin")), str(row.get("experiment_id"))))
        return {
            "items": items,
            "count": len(items),
            "is_claim": False,
            "note": (
                "Registered object-detection experiments. "
                "A campaign must bind one of these. KEEP ≠ Claim."
            ),
        }

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        path = self._dir(experiment_id) / "experiment.json"
        if not path.is_file():
            return None
        return self._public(_load_record(path))

    def require(self, experiment_id: str) -> dict[str, Any]:
        row = self.get(experiment_id)
        if row is None:
            raise RegisteredExperimentError(f"unknown experiment: {experiment_id}")
        return row

    def require_for_start(self, experiment_id: str | None) -> dict[str, Any]:
        eid = str(experiment_id or "").strip()
        if not eid:
            raise RegisteredExperimentError(
                "战役启动必须选择已登记实验（experiment_id）。"
                "V26 RGB-T D-FINE / APS_lowlight 只是其中一场，不是唯一运行时"
            )
        self.ensure_builtin()
        row = self.require(eid)
        if str(row.get("status") or "") == "archived":
            raise RegisteredExperimentError(f"experiment {eid} is archived")
        if str(row.get("task_type") or "") != "object_detection":
            raise RegisteredExperimentError(
                f"不支持的 task_type={row.get('task_type_raw') or row.get('task_type')}；"
                "当前仅 object_detection 可开战役"
            )
        if str(row.get("status") or "") != "ready":
            reason = row.get("unsupported_reason") or "status is not ready"
            raise RegisteredExperimentError(f"experiment {eid} cannot start: {reason}")
        protocol = load_json(row["protocol_path"])
        plan = load_json(row["seed_plan_path"]) if row.get("seed_plan_path") else None
        assert_seed_how_materializable(protocol, plan)
        return row

    def register(
        self,
        *,
        protocol: Mapping[str, Any],
        seed_plan: Mapping[str, Any] | None = None,
        experiment_id: str | None = None,
        title: str | None = None,
        catalog_id: str | None = None,
        idea_brief: Mapping[str, Any] | None = None,
        user_intent: str | None = None,
        interview_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate protocol schema and freeze a new experiment. No GPU."""
        self.ensure_builtin()
        body = dict(protocol)
        try:
            validate_named("research_protocol", body)
        except SchemaValidationError as exc:
            raise RegisteredExperimentError(str(exc)) from exc
        plan_body = dict(seed_plan) if seed_plan else None
        intent = str(user_intent or "").strip() or None
        brief = dict(idea_brief) if isinstance(idea_brief, Mapping) else None
        iid = str(interview_id or "").strip() or None
        if plan_body is not None:
            if intent or brief:
                stamp_parts = [
                    str(plan_body.get("rationale") or "").strip(),
                    f"Idea Interview intent: {intent}" if intent else "",
                    (
                        f"core_intent={brief.get('core_intent')}; "
                        f"direction={brief.get('research_direction')}; "
                        f"hypothesis={brief.get('hypothesis')}"
                        if brief
                        else ""
                    ),
                    "KEEP ≠ Claim. Snapshot only; not campaign Planner.",
                ]
                plan_body["rationale"] = " ".join(p for p in stamp_parts if p)
                if intent and not str(plan_body.get("hypothesis") or "").strip():
                    plan_body["hypothesis"] = intent[:400]
            try:
                validate_named("experiment_plan", plan_body)
            except SchemaValidationError as exc:
                raise RegisteredExperimentError(str(exc)) from exc
            proto_id = str(body.get("protocol_id") or "")
            plan_proto = str(plan_body.get("protocol_id") or "")
            if plan_proto and proto_id and plan_proto != proto_id:
                raise RegisteredExperimentError(
                    f"seed plan protocol_id={plan_proto} does not match protocol {proto_id}"
                )
        eid = sanitize_experiment_id(experiment_id or default_experiment_id(body))
        if eid == BUILTIN_EXPERIMENT_ID:
            raise RegisteredExperimentError(
                f"refusing to overwrite built-in experiment {BUILTIN_EXPERIMENT_ID}; "
                "write a new protocol and register a new experiment_id"
            )
        if self.get(eid) is not None:
            raise RegisteredExperimentError(f"experiment already exists: {eid}")
        notes = (
            "Registered from a submitted protocol JSON. Not a Claim. "
            "KEEP ≠ Claim. Campaigns must bind this experiment_id."
        )
        if intent:
            notes = f"{notes} Idea intent frozen at register."
        return self._write_record(
            experiment_id=eid,
            protocol=body,
            plan=plan_body,
            builtin=False,
            catalog_id=catalog_id or CATALOG_ID,
            title=title,
            notes=notes,
            idea_brief=brief,
            user_intent=intent,
            interview_id=iid,
        )

    def protocol_and_plan(self, experiment_id: str) -> tuple[Path, Path]:
        row = self.require_for_start(experiment_id)
        protocol_path = Path(row["protocol_path"])
        plan_path = Path(row["seed_plan_path"])
        if not protocol_path.is_file() or not plan_path.is_file():
            raise RegisteredExperimentError(
                f"experiment {experiment_id} is missing frozen protocol/plan files"
            )
        return protocol_path, plan_path

    def _dir(self, experiment_id: str) -> Path:
        return self.root / str(experiment_id)

    def _write_record(
        self,
        *,
        experiment_id: str,
        protocol: Mapping[str, Any],
        plan: Mapping[str, Any] | None,
        builtin: bool,
        catalog_id: str,
        title: str | None = None,
        notes: str = "",
        idea_brief: Mapping[str, Any] | None = None,
        user_intent: str | None = None,
        interview_id: str | None = None,
    ) -> dict[str, Any]:
        dest = self._dir(experiment_id)
        dest.mkdir(parents=True, exist_ok=True)
        protocol_path = dest / "protocol.json"
        _dump(protocol_path, protocol)
        plan_path: Path | None = None
        if plan is not None:
            plan_path = dest / "plan.json"
            _dump(plan_path, plan)
        if idea_brief or user_intent or interview_id:
            _dump(
                dest / "idea_snapshot.json",
                {
                    "interview_id": interview_id,
                    "user_intent": user_intent,
                    "brief": dict(idea_brief) if idea_brief else None,
                    "frozen_at_register": True,
                    "is_claim": False,
                    "note": "Idea Interview snapshot. Not campaign Planner. KEEP ≠ Claim.",
                },
            )
        task_raw = str((protocol.get("goal") or {}).get("task_type") or "")
        task = canonicalize_task_type(task_raw)
        adapter = adapter_token(protocol)
        unsupported: str | None = None
        status = "ready"
        if task != "object_detection":
            status = "draft"
            unsupported = (
                f"unsupported task_type={task_raw or task}; "
                "only object_detection campaigns are enabled"
            )
        if plan is None:
            status = "draft"
            unsupported = (unsupported or "missing seed plan").strip()
        if adapter not in SUPPORTED_ADAPTERS:
            status = "draft"
            unsupported = (
                f"Adapter {adapter!r} cannot be materialized; "
                "will not silently run as D-FINE F1"
            )
        adapter_ready = False
        seed_how_id = plan_how_id(plan or {}) if plan else None
        identity = science_identity(protocol, plan)
        if status == "ready" and plan is not None:
            try:
                how = assert_seed_how_materializable(protocol, plan)
                adapter_ready = True
                seed_how_id = str(how.get("how_id") or seed_how_id or "")
                identity["seed_how_id"] = str(seed_how_id or identity.get("seed_how_id") or "") or None
            except RegisteredExperimentError as exc:
                status = "draft"
                unsupported = str(exc)
                adapter_ready = False
        if (
            not builtin
            and status == "ready"
            and identity == builtin_science_identity()
        ):
            status = "draft"
            unsupported = (
                "science fingerprint matches builtin V26 "
                f"(dataset_id/slice_id/primary_metric/seed_how_id={identity}). "
                "Refusing idle loop; register a different scientific object."
            )
            adapter_ready = False
        now = _now()
        record = {
            "experiment_id": experiment_id,
            "title": str(title or protocol.get("title") or experiment_id),
            "task_type": task if task else "unsupported",
            "task_type_raw": task_raw or task,
            "protocol_id": protocol.get("protocol_id"),
            "protocol_version": protocol.get("protocol_version"),
            "protocol_fingerprint": protocol_fingerprint(protocol),
            "protocol_content_hash": protocol_content_hash(protocol),
            "protocol_path": str(protocol_path),
            "seed_plan_path": str(plan_path) if plan_path else None,
            "seed_plan_id": (plan or {}).get("plan_id") if plan else None,
            "seed_how_id": seed_how_id,
            "science_identity": identity,
            "dataset_id": parse_dataset_id(
                str(((protocol.get("baseline") or {}).get("dataset") or ""))
            ),
            "slice_id": str(((protocol.get("condition_slice") or {}).get("id") or "") or "")
            or None,
            "adapter": "rtdetr" if adapter in {"rtdetr", "rt_detr"} else adapter,
            "detector": str(((protocol.get("baseline") or {}).get("model") or "") or "")
            or None,
            "catalog_id": catalog_id,
            "status": status,
            "builtin": bool(builtin),
            "is_claim": False,
            "adapter_ready": adapter_ready,
            "unsupported_reason": unsupported,
            "notes": notes,
            "user_intent": user_intent,
            "interview_id": interview_id,
            "idea_brief": dict(idea_brief) if idea_brief else None,
            "created_at": now,
            "updated_at": now,
        }
        _dump(dest / "experiment.json", record)
        return self._public(record)

    def _public(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(record)
        protocol_path = Path(str(row.get("protocol_path") or ""))
        protocol: dict[str, Any] = {}
        if protocol_path.is_file():
            try:
                protocol = load_json(protocol_path)
            except (OSError, json.JSONDecodeError):
                protocol = {}
        primary = dict(((protocol.get("objective") or {}).get("primary") or {}))
        claim = dict(protocol.get("claim_policy") or {})
        stop = dict(protocol.get("stop_rules") or {})
        risk = dict(protocol.get("risk_policy") or {})
        # Safe protocol JSON for human preview (constitution). No secrets in schema.
        # Fail closed: empty protocol means UI must block start.
        row["protocol"] = protocol
        row["protocol_summary"] = {
            "protocol_id": row.get("protocol_id"),
            "protocol_version": protocol.get("protocol_version") or row.get("protocol_version"),
            "title": protocol.get("title") or row.get("title"),
            "goal": protocol.get("goal"),
            "primary_metric": primary.get("metric"),
            "dataset": ((protocol.get("baseline") or {}).get("dataset")),
            "adapter": ((protocol.get("baseline") or {}).get("adapter")),
            "slice_id": ((protocol.get("condition_slice") or {}).get("id")),
            "condition_slice": protocol.get("condition_slice"),
            "editable_scope": protocol.get("editable_scope"),
            "frozen_scope": protocol.get("frozen_scope"),
            "risk_policy": risk,
            "stop_rules": stop,
            "claim_policy": claim,
            "fingerprint_id": protocol.get("fingerprint_id"),
            "max_rounds": stop.get("max_rounds"),
            "allow_scientific_claims": claim.get("allow_scientific_claims"),
            "default_claim_level": claim.get("default_claim_level"),
        }
        if not isinstance(row.get("science_identity"), Mapping) or not row.get("science_identity"):
            plan_path = Path(str(row.get("seed_plan_path") or ""))
            plan: dict[str, Any] = {}
            if plan_path.is_file():
                try:
                    plan = load_json(plan_path)
                except (OSError, json.JSONDecodeError):
                    plan = {}
            row["science_identity"] = science_identity(protocol, plan or None)
        row["is_claim"] = False
        return row


def _load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
