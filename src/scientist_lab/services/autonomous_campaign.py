"""P0 unattended Manager campaigns: one Human Gate, then multi-round GPU.

Does not invent HOW. Does not bypass Gate. Does not forge metrics.
DeepSeek Harness is not involved. Cursor is not involved.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from scientist_lab.core.manager import Manager
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.core.state_machine import OrchestrationAction, RunState
from scientist_lab.services.live_gpu_mutex import (
    LIVE_CAMPAIGN_STATUSES,
    LiveExecuteLease,
    acquire_live_execute,
    campaign_worker_alive,
)
from scientist_lab.services.registered_experiments import (
    BUILTIN_EXPERIMENT_ID,
    RegisteredExperimentError,
    RegisteredExperimentService,
)

# Soft ceiling for unattended campaigns (start + extend). Not a scientific Claim.
MAX_EXTRA_ROUNDS_HARD_CAP = 47

DEFAULT_PROTOCOL = SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json"
DEFAULT_PLAN = SCHEMA_DIR / "examples" / "experiment_plan_rgbt_dfine_v26_r0.json"

_TERMINAL = {
    OrchestrationAction.STOP.value,
    OrchestrationAction.NEED_HUMAN.value,
    OrchestrationAction.NOVELTY_EXHAUSTED.value,
    OrchestrationAction.IDLE.value,
    OrchestrationAction.PROTOCOL_AMENDMENT_REQUIRED.value,
}

LiveRunner = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_VIEW_KEYS = frozenset({"progress", "notebook", "how_pending", "protocol"})


def _disk_doc(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dict(doc).items() if k not in _VIEW_KEYS}


def _stamp_worker(spec: dict[str, Any]) -> dict[str, Any]:
    spec["worker_pid"] = os.getpid()
    return spec


def _scout_evidence(
    campaign: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from scientist_lab.literature.query_synth import protocol_metric, protocol_research_question

    notebook = dict(campaign.get("notebook") or {})
    last_how = ""
    last_verdict = ""
    for row in list(notebook.get("lab_log") or []):
        if not isinstance(row, Mapping):
            continue
        if row.get("how_id"):
            last_how = str(row.get("how_id"))
        body = str(row.get("body") or "")
        for token in ("KEEP", "DISCARD", "REPLICATE"):
            if token in body:
                last_verdict = token
    pending = []
    store = dict(campaign.get("how_pending") or {})
    for row in list(store.get("candidates") or []):
        if isinstance(row, Mapping) and str(row.get("status") or "") == "proposed":
            hid = str(row.get("how_id") or "").strip()
            if hid:
                pending.append(hid)
    progress = dict(campaign.get("progress") or {})
    metrics = dict(dict(progress.get("result") or {}).get("metrics") or {})
    proto = dict(protocol or campaign.get("protocol") or {})
    metric = protocol_metric(proto) if proto else "APS_lowlight"
    return {
        "last_how_id": last_how,
        "last_review_decision": last_verdict,
        "pending_how_ids": pending,
        "metric": metric,
        "research_question": protocol_research_question(proto),
        "last_primary": metrics.get(metric) or metrics.get("APS_lowlight"),
        "keep_is_not_claim": True,
    }


class AutonomousCampaignError(ValueError):
    """Illegal campaign request."""


class AutonomousCampaignService:
    """Start Manager in a background thread after a one-time Human Gate."""

    def __init__(
        self,
        *,
        project_root: Path | str,
        live_runner: LiveRunner | None = None,
        doctor_fn: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.root = self.project_root / ".run" / "autonomous"
        self.experiments = RegisteredExperimentService(self.project_root)
        self.experiments.ensure_builtin()
        self._live_runner = live_runner
        self._doctor_fn = doctor_fn
        self._threads: dict[str, threading.Thread] = {}
        self._leases: dict[str, LiveExecuteLease] = {}
        self._lock = threading.Lock()
        self.reclaim_orphaned_campaigns()

    def bind_live_runner(self, runner: LiveRunner | None) -> None:
        """Tests inject a stub. Production leaves this None → CUDA live_runner."""
        self._live_runner = runner

    def start(
        self,
        *,
        experiment_id: str | None = None,
        confirm_human_gate: bool = False,
        execute: bool = False,
        llm_live: bool = False,
        max_extra_rounds: int = 1,
        max_steps: int = 48,
        planner_backend: str | None = None,
        reviewer_backend: str | None = None,
        live_ready: bool | None = None,
        llm_ready: bool | None = None,
        background: bool = True,
        protocol_path: Path | str | None = None,
        plan_path: Path | str | None = None,
        baseline_metrics: Mapping[str, Any] | None = None,
        llm_how_lifecycle: bool | None = None,
        plugin_worker: str | None = None,
        llm_may_invent_how: bool = False,
    ) -> dict[str, Any]:
        extra = int(max_extra_rounds)
        if extra < 1:
            raise AutonomousCampaignError("P0 requires max_extra_rounds>=1 (two GPU rounds)")
        if extra > MAX_EXTRA_ROUNDS_HARD_CAP:
            raise AutonomousCampaignError(
                f"max_extra_rounds cap is {MAX_EXTRA_ROUNDS_HARD_CAP}"
            )
        if llm_may_invent_how and not confirm_human_gate:
            raise PermissionError(
                "Stage B llm_may_invent_how requires Campaign Human Gate（confirm_human_gate）"
            )
        try:
            experiment = self.experiments.require_for_start(experiment_id)
            src_protocol, src_plan = self.experiments.protocol_and_plan(
                str(experiment["experiment_id"])
            )
        except RegisteredExperimentError as exc:
            raise AutonomousCampaignError(str(exc)) from exc
        if protocol_path or plan_path:
            raise AutonomousCampaignError(
                "campaign start copies the bound experiment protocol+seed plan; "
                "do not pass a side protocol_path. Register a new experiment instead"
            )
        if execute and not confirm_human_gate:
            raise PermissionError(
                "真实 GPU 需要一次 Campaign Human Gate（confirm_human_gate）"
            )
        if llm_live and llm_ready is False:
            return {
                "ok": False,
                "fail_closed": True,
                "metrics_forged": False,
                "error": "LLM 未就绪（缺 Key / 未允许联网）；拒绝伪造成功",
                "status": "blocked",
            }
        if execute and live_ready is False:
            return {
                "ok": False,
                "fail_closed": True,
                "metrics_forged": False,
                "error": "cuda doctor live_ready=false; refusing GPU and forged metrics.json",
                "status": "blocked",
                "gpu": False,
            }

        stamp = _stamp()
        campaign_id = f"{experiment['experiment_id']}_{stamp}"
        lease: LiveExecuteLease | None = None
        try:
            if execute:
                self.reclaim_orphaned_campaigns()
                # Stub tests stay on this project_root (no Docker scan).
                # Real GPU also refuses if scientist-exec-* already holds the card.
                lease = acquire_live_execute(
                    self.project_root,
                    owner_id=campaign_id,
                    inspect_docker=self._live_runner is None,
                )
            work = self.root / campaign_id
            work.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_protocol, work / "protocol.json")
            shutil.copyfile(src_plan, work / "plan.json")
            seeded = dict(baseline_metrics or {})
            if not seeded and experiment["experiment_id"] == BUILTIN_EXPERIMENT_ID:
                seeded = self._optional_r0_baseline()
            if seeded:
                _dump(work / "baseline_metrics.json", seeded)

            backend = planner_backend or ("llm" if llm_live else "rules")
            review_backend = reviewer_backend or backend
            lifecycle = bool(llm_live) if llm_how_lifecycle is None else bool(llm_how_lifecycle)
            steps_cap = max(int(max_steps), (extra + 1) * 12 + 32)
            frozen_protocol = load_json(work / "protocol.json")
            frozen_plan = load_json(work / "plan.json")
            idea_snapshot = None
            snap_path = Path(str(experiment.get("protocol_path") or "")).parent / "idea_snapshot.json"
            if snap_path.is_file():
                try:
                    idea_snapshot = _load(snap_path)
                except (OSError, TypeError, ValueError):
                    idea_snapshot = None
            if idea_snapshot is None and (
                experiment.get("user_intent") or experiment.get("idea_brief")
            ):
                idea_snapshot = {
                    "interview_id": experiment.get("interview_id"),
                    "user_intent": experiment.get("user_intent"),
                    "brief": experiment.get("idea_brief"),
                    "frozen_at_register": True,
                    "is_claim": False,
                }
            doc = {
                "campaign_id": campaign_id,
                "experiment_id": experiment["experiment_id"],
                "experiment_title": experiment.get("title"),
                "status": "queued",
                "ok": True,
                "fail_closed": False,
                "metrics_forged": False,
                "execute": bool(execute),
                "llm_live": bool(llm_live),
                "confirm_human_gate": bool(confirm_human_gate),
                "max_extra_rounds": extra,
                "max_steps": steps_cap,
                "planner_backend": backend,
                "reviewer_backend": review_backend,
                "llm_how_lifecycle": lifecycle,
                "plugin_worker": str(plugin_worker or "llm").strip().lower() or "llm",
                "llm_may_invent_how": bool(llm_may_invent_how),
                "stage": "B_invent" if llm_may_invent_how else ("A_plugin" if lifecycle else "catalog"),
                "sota_pursuit": bool(llm_live),
                "live_m1": bool(llm_live),
                "gpu_rounds": 0,
                "last_action": None,
                "run_state": None,
                "work_dir": str(work),
                "protocol_id": frozen_protocol.get("protocol_id"),
                "protocol_fingerprint": experiment.get("protocol_fingerprint"),
                "protocol_content_hash": experiment.get("protocol_content_hash"),
                "protocol_path": str(work / "protocol.json"),
                "seed_plan_id": frozen_plan.get("plan_id"),
                "adapter": experiment.get("adapter"),
                "catalog_id": experiment.get("catalog_id"),
                "how_catalog": experiment.get("catalog_id"),
                "dataset_id": experiment.get("dataset_id"),
                "slice_id": experiment.get("slice_id"),
                "idea_snapshot": idea_snapshot,
                "is_claim": False,
                "steps": [],
                "error": None,
                "stop_requested": False,
                "worker_pid": os.getpid(),
                "created_at": _now(),
                "updated_at": _now(),
                "note": (
                    "Unattended loop bound to a registered experiment. "
                    "One Human Gate authorizes Manager to run HOW ids already "
                    "in that experiment's catalog, and (when llm_how_lifecycle) "
                    "lets the LLM add/author/accept overlay plugins. "
                    + (
                        "Stage B: llm_may_invent_how=true — invent-fallback HOW drafts "
                        "allowed (human review → Diff plugin). Plan still no Python. "
                        if llm_may_invent_how
                        else ""
                    )
                    + "KEEP ≠ Claim."
                ),
            }
            if llm_may_invent_how:
                # Distinct Stage B fingerprint note on the frozen protocol copy.
                notes = str(frozen_protocol.get("notes") or "")
                tag = " [Live B invent gate 2026-08-30; KEEP≠Claim; no Plan Python]"
                if tag.strip() not in notes:
                    frozen_protocol["notes"] = (notes + tag).strip()
                fp = str(frozen_protocol.get("fingerprint_id") or "FP-RGBT-DFINE-V26")
                if not fp.endswith("-LIVE-B"):
                    frozen_protocol["fingerprint_id"] = f"{fp}-LIVE-B"
                stop = dict(frozen_protocol.get("stop_rules") or {})
                stop["max_rounds"] = max(int(stop.get("max_rounds") or 12), 12)
                frozen_protocol["stop_rules"] = stop
                _dump(work / "protocol.json", frozen_protocol)
                doc["protocol_fingerprint"] = frozen_protocol.get("fingerprint_id")
                doc["protocol_id"] = frozen_protocol.get("protocol_id")
            _dump(work / "campaign.json", doc)
            from scientist_lab.core.how_pending import pending_path, set_llm_may_invent_how

            set_llm_may_invent_how(pending_path(work), bool(llm_may_invent_how))

            self._launch_worker(campaign_id, lease=lease, background=background)
            lease = None
            return self.get(campaign_id)
        finally:
            if lease is not None:
                lease.release()

    def get(self, campaign_id: str) -> dict[str, Any]:
        path = self.root / campaign_id / "campaign.json"
        if not path.is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        payload = _load(path)
        work = self.root / campaign_id
        protocol_path = work / "protocol.json"
        # Frozen campaign constitution snapshot for read-only UI (not editable).
        payload["protocol"] = _load(protocol_path) if protocol_path.is_file() else {}
        payload["progress"] = self._progress(campaign_id)
        payload["worker_alive"] = self._thread_alive(campaign_id)
        from scientist_lab.services.campaign_notebook import build_campaign_notebook

        payload["notebook"] = build_campaign_notebook(
            work,
            payload,
            project_root=self.project_root,
        )
        from scientist_lab.core.how_pending import load_store, pending_path
        from scientist_lab.core.campaign_steer import load_store as load_steer, public_store, steer_path

        payload["how_pending"] = load_store(pending_path(work))
        payload["steer"] = public_store(load_steer(steer_path(work)))
        return payload

    def set_campaign_steer(
        self,
        campaign_id: str,
        *,
        action: str,
        text: str | None = None,
        why: str | None = None,
    ) -> dict[str, Any]:
        work = self.root / campaign_id
        if not (work / "campaign.json").is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        from scientist_lab.core.campaign_steer import (
            CampaignSteerError,
            clear_steer,
            set_steer,
            steer_path,
        )

        dest = steer_path(work)
        kind = str(action or "").strip().lower()
        try:
            if kind in {"set", "human_set"}:
                if not str(text or "").strip():
                    raise CampaignSteerError("steer requires text")
                set_steer(dest, text=str(text), why=str(why or text))
            elif kind == "clear":
                clear_steer(dest, why=str(why or "human cleared steer"))
            else:
                raise CampaignSteerError(f"unknown steer action: {action}")
        except CampaignSteerError as exc:
            raise AutonomousCampaignError(str(exc)) from exc
        return self.get(campaign_id)

    def chat_scout_intent(
        self,
        campaign_id: str,
        message: str,
        *,
        live: bool = False,
        locale: str | None = None,
        action: str | None = None,
    ) -> dict[str, Any]:
        work = self.root / campaign_id
        if not (work / "campaign.json").is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        from scientist_lab.core.how_pending import pending_path
        from scientist_lab.core.scout_dialogue import handle_scout_chat

        campaign = self.get(campaign_id)
        protocol_path = work / "protocol.json"
        protocol = _load(protocol_path) if protocol_path.is_file() else {}
        result = handle_scout_chat(
            pending_path(work),
            message,
            evidence=_scout_evidence(campaign, protocol),
            protocol=protocol,
            live=bool(live),
            locale=locale,
            provenance_dir=work / "literature",
            action=action,
        )
        payload = self.get(campaign_id)
        payload["scout_reply"] = result.get("reply")
        payload["scout_refused"] = bool(result.get("refused"))
        payload["scout_action"] = result.get("action")
        return payload

    def localize_scout(
        self,
        campaign_id: str,
        *,
        locale: str,
        live: bool = False,
    ) -> dict[str, Any]:
        work = self.root / campaign_id
        if not (work / "campaign.json").is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        from scientist_lab.core.how_pending import localize_scout_store, pending_path

        localize_scout_store(pending_path(work), locale=locale, live=bool(live))
        return self.get(campaign_id)

    def decide_scout_intent(
        self,
        campaign_id: str,
        *,
        action: str,
        query: str | None = None,
        why: str | None = None,
    ) -> dict[str, Any]:
        work = self.root / campaign_id
        if not (work / "campaign.json").is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        from scientist_lab.core.how_pending import (
            HowPendingError,
            INTENT_ACTIVE,
            accept_proposed_intent,
            append_scout_dialogue,
            clear_scout_intent,
            pending_path,
            reject_proposed_intent,
            set_scout_intent,
        )

        dest = pending_path(work)
        kind = str(action or "").strip().lower()
        try:
            if kind in {"human_set", "set"}:
                if not str(query or "").strip():
                    raise HowPendingError("human scout_intent requires query")
                set_scout_intent(
                    dest,
                    source="human",
                    query=str(query),
                    why=str(why or query),
                    status=INTENT_ACTIVE,
                    drafted_by="human",
                )
                append_scout_dialogue(
                    dest, role="human", text=str(query), intent_action="human_set"
                )
                append_scout_dialogue(
                    dest,
                    role="assistant",
                    text=f"已记下人指定的检索方向。下一枪 scout 使用：{query}",
                    intent_action="human_set",
                )
            elif kind == "accept":
                accept_proposed_intent(dest)
                append_scout_dialogue(
                    dest, role="human", text="接受这个查询", intent_action="accept"
                )
                append_scout_dialogue(
                    dest,
                    role="assistant",
                    text="已接受查询草稿。下一枪 scout 使用该 query。",
                    intent_action="accept",
                )
            elif kind == "reject":
                reject_proposed_intent(dest)
                append_scout_dialogue(
                    dest, role="human", text="拒绝这个查询", intent_action="reject"
                )
                append_scout_dialogue(
                    dest,
                    role="assistant",
                    text="已拒绝 LLM 查询草稿。下一枪将用 fallback。",
                    intent_action="reject",
                )
            elif kind == "clear":
                clear_scout_intent(dest, why=str(why or "human cleared scout_intent"))
                append_scout_dialogue(
                    dest, role="human", text="清空检索", intent_action="clear"
                )
                append_scout_dialogue(
                    dest,
                    role="assistant",
                    text="已清空检索意图。下一枪用 fallback。",
                    intent_action="clear",
                )
            else:
                raise HowPendingError("action must be set, accept, reject, or clear")
        except HowPendingError as exc:
            raise AutonomousCampaignError(str(exc)) from exc
        return self.get(campaign_id)

    def list(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if self.root.is_dir():
            from scientist_lab.services.campaign_notebook import build_campaign_story

            for path in self.root.glob("*/campaign.json"):
                row = _load(path)
                work = path.parent
                protocol = _load(work / "protocol.json") if (work / "protocol.json").is_file() else {}
                plan = _load(work / "plan.json") if (work / "plan.json").is_file() else {}
                pending = _load(work / "how_pending.json") if (work / "how_pending.json").is_file() else {}
                story = build_campaign_story(
                    row, protocol=protocol, seed_plan=plan, how_pending=pending
                )
                last_round = (story.get("rounds") or [{}])[-1] if story.get("rounds") else {}
                now = dict(story.get("now") or {})
                items.append(
                    {
                        "campaign_id": row.get("campaign_id"),
                        "experiment_id": row.get("experiment_id"),
                        "catalog_id": row.get("catalog_id"),
                        "planner_backend": row.get("planner_backend"),
                        "seed_how_id": story.get("experiment_truth", {}).get("identity", {}).get(
                            "seed_how_id"
                        ),
                        "truth_headline": (
                            (story.get("experiment_truth") or {}).get("truth_summary") or [None]
                        )[0],
                        "experiment_title": row.get("experiment_title"),
                        "status": row.get("status"),
                        "execute": row.get("execute"),
                        "gpu_rounds": row.get("gpu_rounds"),
                        "last_action": row.get("last_action"),
                        "run_state": row.get("run_state"),
                        "created_at": row.get("created_at"),
                        "updated_at": row.get("updated_at"),
                        "ok": row.get("ok"),
                        "headline": story.get("headline"),
                        "question": story.get("question"),
                        "did": story.get("did"),
                        "primary_metric": story.get("primary_metric"),
                        # Prefer live now-board HOW (in-flight plan), not stale scoreboard.
                        "last_how": now.get("current_how_id")
                        or (last_round.get("how_id") if isinstance(last_round, dict) else None),
                        "last_how_label": now.get("current_how_label")
                        or (last_round.get("how_label") if isinstance(last_round, dict) else None),
                        "now_round_label": now.get("round_label"),
                        "now_next_kind": now.get("next_kind"),
                    }
                )
            items.sort(
                key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""),
                reverse=True,
            )
        catalog = self.experiments.list()
        return {
            "items": items,
            "count": len(items),
            "experiments": catalog.get("items") or [],
            "experiment_count": catalog.get("count") or 0,
            "is_claim": False,
        }

    def _thread_alive(self, campaign_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(campaign_id)
        return thread is not None and thread.is_alive()

    def _await_worker_stop(self, campaign_id: str, *, timeout_s: float = 2.0) -> bool:
        """Wait briefly for the daemon worker to exit after pause/NEED_HUMAN.

        Disk status often flips to paused a moment before ``finally`` pops the
        thread registry; a short join avoids false resume failures.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                thread = self._threads.get(campaign_id)
            if thread is None:
                return True
            if not thread.is_alive():
                with self._lock:
                    self._threads.pop(campaign_id, None)
                return True
            thread.join(timeout=min(0.2, max(0.0, deadline - time.monotonic())))
        alive = self._thread_alive(campaign_id)
        if not alive:
            with self._lock:
                self._threads.pop(campaign_id, None)
        return not alive

    def reclaim_orphaned_campaigns(self) -> list[str]:
        """Mark execute=true campaigns whose worker PID is gone as paused.

        API restart kills the daemon thread but left campaign.json as running,
        which occupied the GPU mutex with no training job. Does not auto-resume.
        Leaves waiting_gpu alone if a scientist-exec-* container is still live.
        """
        reclaimed: list[str] = []
        if not self.root.is_dir():
            return reclaimed
        docker_busy = False
        try:
            from scientist_lab.services.live_gpu_mutex import _live_docker_execs

            docker_busy = bool(_live_docker_execs())
        except Exception:  # noqa: BLE001 — occupancy must not crash reclaim
            docker_busy = False
        for path in self.root.glob("*/campaign.json"):
            try:
                row = _load(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not row.get("execute"):
                continue
            status = str(row.get("status") or "")
            if status not in LIVE_CAMPAIGN_STATUSES:
                continue
            cid = str(row.get("campaign_id") or path.parent.name)
            if self._thread_alive(cid):
                continue
            # In-process workers stamp worker_pid=server PID. If the thread is
            # gone but the PID is still this process, treat as orphan.
            try:
                pid = int(row.get("worker_pid") or 0)
            except (TypeError, ValueError):
                pid = 0
            same_process_orphan = pid == os.getpid() and pid > 0
            if campaign_worker_alive(row) and not same_process_orphan:
                continue
            # Docker still training: campaign.json may lag on NEED_GATE /
            # running until the blocking execute step returns. Do not pause
            # mid-flight just because worker_pid looks dead to the poller.
            last_action = str(row.get("last_action") or "")
            mid_gpu = last_action in {
                OrchestrationAction.NEED_EXECUTION.value,
                OrchestrationAction.NEED_GATE.value,
            }
            if docker_busy and (status == "waiting_gpu" or mid_gpu):
                continue
            row["status"] = "paused"
            row["stop_requested"] = True
            row["ok"] = True
            row["orphan_reclaimed"] = True
            row.pop("worker_pid", None)
            row["updated_at"] = _now()
            if not row.get("error"):
                row["error"] = (
                    "worker lost after process restart; GPU lock released. Not auto-resumed."
                )
            _dump(path, _disk_doc(row))
            reclaimed.append(cid)
        return reclaimed

    def request_stop(self, campaign_id: str) -> dict[str, Any]:
        doc = self.get(campaign_id)
        doc["stop_requested"] = True
        doc["updated_at"] = _now()
        status = str(doc.get("status") or "")
        if status in {"queued", "running", "waiting_gpu", "pause_requested"}:
            if self._thread_alive(campaign_id) or (
                campaign_worker_alive(doc) and int(doc.get("worker_pid") or 0) != os.getpid()
            ):
                doc["status"] = "pause_requested"
            else:
                doc["status"] = "paused"
                doc["ok"] = True
                doc["orphan_reclaimed"] = True
                if not doc.get("error"):
                    doc["error"] = (
                        "worker not running; paused immediately so the GPU lock is released."
                    )
        _dump(self.root / campaign_id / "campaign.json", _disk_doc(doc))
        return self.get(campaign_id)

    def resume(
        self,
        campaign_id: str,
        *,
        live_ready: bool | None = None,
        llm_ready: bool | None = None,
        background: bool = True,
    ) -> dict[str, Any]:
        """Explicit human resume after pause / orphan reclaim / recoverable NEED_HUMAN.

        Does not auto-run. completed/failed + NEED_HUMAN (Planner fail-closed) and
        failed + ReviewRefused (Reviewer contract refuse after evidence) are
        resumable so a fixed contract can continue. Mid-GPU harvest reattaches.
        """
        path = self.root / campaign_id / "campaign.json"
        if not path.is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        spec = _load(path)
        status = str(spec.get("status") or "")
        last_action = str(spec.get("last_action") or "")
        # Ghost running: thread dead but campaign.json still says running (same-process
        # worker_pid = server PID). Treat NEED_HUMAN as recoverable.
        if (
            status == "running"
            and last_action == "NEED_HUMAN"
            and not self._thread_alive(campaign_id)
        ):
            spec["status"] = "paused"
            spec["stop_requested"] = True
            spec["orphan_reclaimed"] = True
            spec.pop("worker_pid", None)
            _dump(path, spec)
            status = "paused"
        # Mid-GPU orphan: waiter thread died; scientist-exec may still be running
        # or already finished writing outputs. Resume reattaches (harvest), never
        # invents a second containers.run for the same round.
        mid_gpu = last_action in {
            OrchestrationAction.NEED_EXECUTION.value,
            OrchestrationAction.NEED_GATE.value,
        }
        if (
            status in {"waiting_gpu", "running"}
            and mid_gpu
            and not self._thread_alive(campaign_id)
        ):
            try:
                pid = int(spec.get("worker_pid") or 0)
            except (TypeError, ValueError):
                pid = 0
            same_process_orphan = pid == os.getpid() and pid > 0
            foreign_dead = not campaign_worker_alive(spec)
            if same_process_orphan or foreign_dead:
                spec["status"] = "paused"
                spec["stop_requested"] = True
                spec["orphan_reclaimed"] = True
                spec["pending_harvest"] = True
                spec.pop("worker_pid", None)
                if not spec.get("error") or "waiter" in str(spec.get("error") or "").lower():
                    spec["error"] = (
                        "GPU waiter lost; resume will reattach/harvest existing "
                        "scientist-exec outputs (no new GPU job)."
                    )
                _dump(path, spec)
                status = "paused"
        # Planner fail-closed / schema coercion bugs land as failed+NEED_HUMAN.
        recoverable_need_human = last_action == "NEED_HUMAN" and status in {
            "completed",
            "failed",
            "blocked",
            "waiting_human",
        }
        # Reviewer contract refuse (e.g. FDPN cite false positive) lands as
        # failed with last_action still NEED_EVIDENCE_CHECK / NEED_REVIEW.
        err_text = str(spec.get("error") or "")
        recoverable_review_refuse = status == "failed" and err_text.startswith(
            "ReviewRefused:"
        ) and last_action in {
            OrchestrationAction.NEED_EVIDENCE_CHECK.value,
            OrchestrationAction.NEED_REVIEW.value,
            OrchestrationAction.NEED_HUMAN.value,
        }
        recoverable_how_draft = status == "failed" and (
            err_text.startswith("HowPendingError:")
            or "invent fallback must not include Python" in err_text
        )
        # Windows path overflow while archiving nested run_plan_roundN_from_* ids.
        recoverable_path_overflow = status == "failed" and last_action in {
            OrchestrationAction.NEXT_ROUND.value,
            OrchestrationAction.NEED_HUMAN.value,
        } and (
            "OSError" in err_text
            or "Invalid argument" in err_text
            or "Filename too long" in err_text
            or "path too long" in err_text.lower()
        )
        # Planner wrote schema-illegal extras (e.g. expected_effect.reference_last);
        # after code sanitize, resume can rebuild the next plan.
        recoverable_schema = status == "failed" and (
            err_text.startswith("SchemaValidationError:")
            or "expected_effect: Additional properties" in err_text
        )
        recoverable_harvest = bool(spec.get("pending_harvest")) and status == "paused"
        if (
            status not in {"paused", "pause_requested"}
            and not recoverable_need_human
            and not recoverable_review_refuse
            and not recoverable_how_draft
            and not recoverable_path_overflow
            and not recoverable_schema
            and not recoverable_harvest
        ):
            raise AutonomousCampaignError(
                f"cannot resume campaign in status={status!r}; "
                "only paused, NEED_HUMAN after Planner fail-closed, "
                "ReviewRefused after evidence/review, "
                "HOW draft soft-fail, "
                "path-overflow after NEXT_ROUND, "
                "SchemaValidationError after plan sanitize, "
                "or mid-GPU harvest reattach"
            )
        if self._thread_alive(campaign_id):
            # Disk often says paused before the daemon leaves ``_threads``.
            self._await_worker_stop(campaign_id, timeout_s=2.0)
        if self._thread_alive(campaign_id):
            # Soft signal — not fail-closed. UI should poll and retry.
            return {
                "ok": False,
                "fail_closed": False,
                "pause_in_progress": True,
                "metrics_forged": False,
                "campaign_id": campaign_id,
                "status": status,
                "worker_alive": True,
                "error": (
                    "campaign worker still finishing pause or GPU round; "
                    "wait until status is paused and GPU is idle, then retry"
                ),
            }
        if status == "pause_requested":
            spec["status"] = "paused"
        execute = bool(spec.get("execute"))
        if not spec.get("confirm_human_gate"):
            raise PermissionError(
                "续跑需要实验已在开始时通过 Human Gate（confirm_human_gate）"
            )
        if spec.get("llm_live") and llm_ready is False:
            return {
                "ok": False,
                "fail_closed": True,
                "metrics_forged": False,
                "campaign_id": campaign_id,
                "error": "LLM 未就绪（缺 Key / 未允许联网）；拒绝伪造成功",
                "status": "paused",
            }
        if execute and live_ready is False:
            return {
                "ok": False,
                "fail_closed": True,
                "metrics_forged": False,
                "campaign_id": campaign_id,
                "error": "cuda doctor live_ready=false; refusing GPU resume",
                "status": "paused",
                "gpu": False,
            }

        lease: LiveExecuteLease | None = None
        try:
            if execute:
                self.reclaim_orphaned_campaigns()
                # Mid-GPU harvest reattaches to our scientist-exec-*; treating that
                # container as a foreign lock blocks the only valid resume path.
                inspect_docker = self._live_runner is None and not recoverable_harvest
                lease = acquire_live_execute(
                    self.project_root,
                    owner_id=campaign_id,
                    inspect_docker=inspect_docker,
                )
            spec["stop_requested"] = False
            spec["status"] = "queued"
            spec["updated_at"] = _now()
            spec["resumed_at"] = _now()
            resume_count = int(spec.get("resume_count") or 0) + 1
            spec["resume_count"] = resume_count
            if spec.get("orphan_reclaimed"):
                spec.setdefault("orphan_reclaimed_at", spec.get("updated_at"))
            spec["error"] = None
            # Sticky fail_closed from a prior NEED_HUMAN must not look like
            # "续跑失败" on a successful resume response.
            spec["fail_closed"] = False
            spec["ok"] = True
            spec.pop("pending_harvest", None)
            if spec.get("llm_live"):
                spec["sota_pursuit"] = True
            _dump(path, spec)
            self._launch_worker(campaign_id, lease=lease, background=background)
            lease = None
            return self.get(campaign_id)
        finally:
            if lease is not None:
                lease.release()

    def extend_round_budget(
        self,
        campaign_id: str,
        *,
        add_rounds: int = 5,
        confirm_protocol_amendment: bool = False,
        resume: bool = True,
        live_ready: bool | None = None,
        llm_ready: bool | None = None,
        background: bool = True,
    ) -> dict[str, Any]:
        """Human Gate: raise campaign + protocol round budgets, optionally resume.

        Needed when Manager stops on max_extra_rounds / stop_rules.max_rounds.
        Does not invent HOW. KEEP ≠ Claim.
        """
        path = self.root / campaign_id / "campaign.json"
        if not path.is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        add = int(add_rounds)
        if add < 1:
            raise AutonomousCampaignError("add_rounds must be >= 1")
        if add > 20:
            raise AutonomousCampaignError("add_rounds cap is 20 per extend")
        if not confirm_protocol_amendment:
            raise PermissionError(
                "提高额度会改 protocol.stop_rules.max_rounds，"
                "需要 confirm_protocol_amendment（Human Gate / Protocol Amendment）"
            )
        if self._thread_alive(campaign_id):
            raise AutonomousCampaignError(
                "campaign worker still running; stop or wait before extending"
            )

        spec = _load(path)
        status = str(spec.get("status") or "")
        if status not in {"completed", "paused", "failed", "blocked", "pause_requested"}:
            raise AutonomousCampaignError(
                f"cannot extend campaign in status={status!r}; "
                "only completed/paused/failed after a round-budget stop"
            )
        if not spec.get("confirm_human_gate"):
            raise PermissionError(
                "续跑需要实验已在开始时通过 Human Gate（confirm_human_gate）"
            )

        current_extra = int(spec.get("max_extra_rounds") or 1)
        new_extra = current_extra + add
        if new_extra > MAX_EXTRA_ROUNDS_HARD_CAP:
            raise AutonomousCampaignError(
                f"max_extra_rounds would exceed hard cap {MAX_EXTRA_ROUNDS_HARD_CAP}"
            )

        work = self.root / campaign_id
        protocol_path = work / "protocol.json"
        protocol = load_json(protocol_path)
        stop = dict(protocol.get("stop_rules") or {})
        gpu_rounds = int(spec.get("gpu_rounds") or 0)
        old_max = stop.get("max_rounds")
        try:
            old_max_i = int(old_max) if old_max is not None else gpu_rounds
        except (TypeError, ValueError):
            old_max_i = gpu_rounds
        new_max = max(old_max_i + add, gpu_rounds + add)
        stop["max_rounds"] = new_max
        protocol["stop_rules"] = stop
        try:
            protocol["protocol_version"] = int(protocol.get("protocol_version") or 1) + 1
        except (TypeError, ValueError):
            protocol["protocol_version"] = 2
        amendments = list(protocol.get("amendments") or [])
        amendments.append(
            {
                "at": _now(),
                "kind": "max_rounds",
                "add_rounds": add,
                "max_rounds_before": old_max_i,
                "max_rounds_after": new_max,
                "max_extra_rounds_before": current_extra,
                "max_extra_rounds_after": new_extra,
                "note": "Human Gate Protocol Amendment: raise round budget only",
            }
        )
        protocol["amendments"] = amendments[-20:]
        _dump(protocol_path, protocol)

        spec["max_extra_rounds"] = new_extra
        spec["status"] = "paused"
        spec["stop_requested"] = True
        spec["ok"] = True
        spec["fail_closed"] = False
        spec["error"] = None
        spec["updated_at"] = _now()
        spec["round_budget_extended_at"] = _now()
        spec["round_budget_extensions"] = int(spec.get("round_budget_extensions") or 0) + 1
        _dump(path, _disk_doc(spec))

        if not resume:
            return self.get(campaign_id)
        return self.resume(
            campaign_id,
            live_ready=live_ready,
            llm_ready=llm_ready,
            background=background,
        )

    def _launch_worker(
        self,
        campaign_id: str,
        *,
        lease: LiveExecuteLease | None,
        background: bool,
    ) -> None:
        if background:
            thread = threading.Thread(
                target=self._worker,
                args=(campaign_id,),
                daemon=True,
                name=f"autonomous-{campaign_id}",
            )
            with self._lock:
                self._threads[campaign_id] = thread
                if lease is not None:
                    self._leases[campaign_id] = lease
                    lease = None
            thread.start()
        else:
            with self._lock:
                if lease is not None:
                    self._leases[campaign_id] = lease
                    lease = None
            self._worker(campaign_id)
        if lease is not None:
            lease.release()

    def run_sota_pursuit(self, campaign_id: str) -> dict[str, Any]:
        work = self.root / campaign_id
        path = work / "campaign.json"
        if not path.is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        spec = _load(path)
        from scientist_lab.services.sota_pursuit import run_sota_pursuit_tick

        pursuit = run_sota_pursuit_tick(
            work,
            spec,
            draft_runner=self._run_how_draft_arm,
            lifecycle_runner=self._run_how_lifecycle,
        )
        spec = _load(path)
        slog = list(spec.get("sota_pursuit_log") or [])
        slog.append(
            {
                "action": pursuit.get("action"),
                "progressed": pursuit.get("progressed"),
                "reason": pursuit.get("reason"),
                "board": pursuit.get("board"),
                "manual": True,
            }
        )
        spec["sota_pursuit_log"] = slog[-20:]
        spec["sota_board"] = pursuit.get("board")
        if pursuit.get("live_m1_brief"):
            spec["live_m1_brief"] = pursuit.get("live_m1_brief")
        spec["sota_pursuit"] = True
        spec["updated_at"] = _now()
        _dump(path, _disk_doc(spec))
        payload = self.get(campaign_id)
        payload["sota_pursuit"] = pursuit
        return payload

    def decide_how_candidate(
        self,
        campaign_id: str,
        candidate_id: str,
        *,
        decision: str,
        confirm_human_gate: bool = False,
        note: str | None = None,
        actor: str = "human",
    ) -> dict[str, Any]:
        work = self.root / campaign_id
        if not (work / "campaign.json").is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        from scientist_lab.core.how_draft_arm import release_invent_for_lifecycle
        from scientist_lab.core.how_pending import HowPendingError, decide_candidate, pending_path

        action = str(decision or "").strip().lower()
        try:
            if action == "release":
                release_invent_for_lifecycle(
                    pending_path(work),
                    candidate_id,
                    confirm_human_gate=confirm_human_gate,
                    actor=actor,
                    note=note,
                )
            else:
                decide_candidate(
                    pending_path(work),
                    candidate_id,
                    decision=decision,
                    actor=actor,
                    note=note,
                    confirm_human_gate=confirm_human_gate,
                )
        except HowPendingError as exc:
            raise AutonomousCampaignError(str(exc)) from exc

        # After human clears invent/HOW gate, continue lifecycle + resume worker.
        path = work / "campaign.json"
        spec = _load(path)
        if (
            str(spec.get("status") or "") == "waiting_human"
            and action in {"register", "release", "reject"}
            and bool(spec.get("llm_how_lifecycle"))
            and bool(spec.get("confirm_human_gate"))
        ):
            if action in {"register", "release"}:
                tick = self._run_how_lifecycle(work, spec)
                history = list(spec.get("how_lifecycle") or [])
                history.append(tick)
                spec["how_lifecycle"] = history[-20:]
                spec["how_lifecycle_last"] = tick
                if tick.get("draft_arm"):
                    dhist = list(spec.get("how_draft_arm") or [])
                    dhist.append(tick.get("draft_arm"))
                    spec["how_draft_arm"] = dhist[-20:]
                    spec["how_draft_arm_last"] = tick.get("draft_arm")
                if tick.get("action") == "invent_awaiting_human":
                    spec["status"] = "waiting_human"
                    spec["last_action"] = OrchestrationAction.NEED_HUMAN.value
                    spec["error"] = str(
                        tick.get("reason")
                        or "invent-fallback HOW draft awaiting human review"
                    )
                    spec["updated_at"] = _now()
                    _dump(path, _disk_doc(spec))
                    return self.get(campaign_id)
            spec["status"] = "paused"
            spec["stop_requested"] = True
            spec["error"] = None
            spec["updated_at"] = _now()
            spec.pop("worker_pid", None)
            _dump(path, _disk_doc(spec))
            try:
                return self.resume(campaign_id, background=True)
            except AutonomousCampaignError:
                return self.get(campaign_id)

        return self.get(campaign_id)

    def author_how_candidate(
        self,
        campaign_id: str,
        candidate_id: str,
        *,
        live: bool = False,
        plugin_source: str | None = None,
        unified_diff: str | None = None,
        confirm_human_gate: bool = False,
        plugin_worker: str | None = None,
    ) -> dict[str, Any]:
        if not confirm_human_gate:
            raise AutonomousCampaignError(
                "HOW plugin authoring requires confirm_human_gate"
            )
        work = self.root / campaign_id
        if not (work / "campaign.json").is_file():
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        from scientist_lab.core.how_pending import HowPendingError, pending_path
        from scientist_lab.core.how_plugin_author import (
            HowPluginAuthorError,
            author_how_patch,
        )
        from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
        from scientist_lab.llm.gateway import resolve_gateway_provider

        try:
            provider = None
            if live and not plugin_source and not unified_diff:
                provider = resolve_gateway_provider(live=True)
            worker_name = None
            if not plugin_source and not unified_diff:
                worker_name = str(plugin_worker or "llm").strip().lower() or "llm"
            authored = author_how_patch(
                pending_path(work),
                candidate_id,
                project_root=self.project_root,
                provider=provider,
                worker_name=worker_name,
                unified_diff=unified_diff,
                plugin_source=plugin_source,
                live=live,
                sandbox_root=work / "_how_plugin_sandboxes",
                protocol=load_json(work / "protocol.json") if (work / "protocol.json").is_file() else None,
            )
        except (
            HowPendingError,
            HowPluginAuthorError,
            MissingAPIKeyError,
            RealProviderNotEnabledError,
        ) as orig:
            raise AutonomousCampaignError(str(orig)) from orig
        payload = self.get(campaign_id)
        payload["plugin_author"] = authored
        return payload

    def _run_how_draft_arm(self, work: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
        from scientist_lab.core.how_draft_arm import run_how_draft_arm
        from scientist_lab.core.how_pending import HowPendingError
        from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
        from scientist_lab.llm.gateway import GatewayError, resolve_gateway_provider

        provider = None
        live = bool(spec.get("llm_live"))
        if live:
            try:
                provider = resolve_gateway_provider(live=True)
            except (MissingAPIKeyError, RealProviderNotEnabledError, GatewayError, ValueError):
                live = False
                provider = None
        round_id = f"round_{int(spec.get('gpu_rounds') or 0) + 1}"
        try:
            return run_how_draft_arm(
                work / "how_pending.json",
                confirm_human_gate=bool(spec.get("confirm_human_gate")),
                live=live and provider is not None,
                provider=provider,
                round_id=round_id,
                allow_invent_fallback=True,
                protocol=load_json(work / "protocol.json") if (work / "protocol.json").is_file() else None,
            )
        except HowPendingError as exc:
            return {
                "progressed": False,
                "ok": True,
                "action": "draft_arm_skipped",
                "reason": str(exc),
                "gpu": False,
                "can_enter_claim_gate": False,
            }

    def _run_how_lifecycle(self, work: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
        from scientist_lab.core.how_lifecycle import run_how_lifecycle_tick
        from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
        from scientist_lab.llm.gateway import GatewayError, resolve_gateway_provider

        draft = self._run_how_draft_arm(work, spec)
        if draft.get("action") == "invent_awaiting_human":
            return draft
        provider = None
        live = bool(spec.get("llm_live"))
        if live:
            try:
                provider = resolve_gateway_provider(live=True)
            except (MissingAPIKeyError, RealProviderNotEnabledError, GatewayError, ValueError):
                live = False
                provider = None
        plugin_worker = str(spec.get("plugin_worker") or "llm").strip().lower() or "llm"
        tick = run_how_lifecycle_tick(
            work / "how_pending.json",
            project_root=self.project_root,
            confirm_human_gate=True,
            live=live and provider is not None,
            provider=provider,
            plugin_worker=plugin_worker,
            sandbox_root=work / "_how_plugin_sandboxes",
            protocol=load_json(work / "protocol.json") if (work / "protocol.json").is_file() else None,
        )
        if draft.get("progressed") and not tick.get("progressed"):
            return {
                **tick,
                "progressed": True,
                "action": draft.get("action") or tick.get("action"),
                "reason": draft.get("reason") or tick.get("reason"),
                "draft_arm": draft,
            }
        if draft.get("progressed"):
            tick = dict(tick)
            tick["draft_arm"] = draft
        return tick

    def _optional_r0_baseline(self) -> dict[str, Any]:
        candidates = [
            self.project_root / "outputs" / "v26_r0" / "aps_lowlight.json",
            self.project_root / "outputs" / "v26_r0" / "metrics.json",
            self.project_root / "outputs" / "v26_r0" / "result.json",
            self.project_root / "outputs" / "v26_r0" / "run" / "metrics.json",
            self.project_root / "docs" / "research" / "v26" / "R0_BASELINE_FREEZE.json",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            nested = dict(raw.get("metrics") or {})
            bound = dict(raw.get("bound_metrics") or raw.get("metrics_bound") or {})
            metrics = {**bound, **nested, **raw}
            aps = metrics.get("APS_lowlight")
            if isinstance(aps, (int, float)) and not isinstance(aps, bool):
                return {
                    "APS_lowlight": float(aps),
                    "mAP50_95_lowlight": metrics.get("mAP50_95_lowlight"),
                    "AP50_lowlight": metrics.get("AP50_lowlight"),
                    "source": str(path),
                }
        return {}

    def _progress(self, campaign_id: str) -> dict[str, Any]:
        work = self.root / campaign_id
        run_doc = work / "experiment_run.json"
        result = work / "result.json"
        payload: dict[str, Any] = {}
        if run_doc.is_file():
            payload["experiment_run"] = _load(run_doc)
        if result.is_file():
            payload["result"] = _load(result)
        try:
            from scientist_lab.services.training_monitor import snapshot_cli_run

            payload["training"] = snapshot_cli_run(work)
        except Exception:  # noqa: BLE001 — progress is best-effort
            payload["training"] = None
        return payload

    def _worker(self, campaign_id: str) -> None:
        work = self.root / campaign_id
        path = work / "campaign.json"
        try:
            spec = _load(path)
            spec["status"] = "running"
            spec["updated_at"] = _now()
            _dump(path, _stamp_worker(spec))
            protocol = load_json(work / "protocol.json")
            plan = load_json(work / "plan.json")
            baseline_path = work / "baseline_metrics.json"
            baseline = _load(baseline_path) if baseline_path.is_file() else None
            mgr = Manager(
                work,
                protocol=protocol,
                execute=bool(spec.get("execute")),
                require_live_ready=bool(spec.get("execute")) and self._live_runner is None,
                live_runner=self._live_runner,
                baseline_metrics=baseline,
                max_extra_rounds=int(spec.get("max_extra_rounds") or 1),
                planner_backend=str(spec.get("planner_backend") or "rules"),
                reviewer_backend=str(spec.get("reviewer_backend") or "rules"),
                llm_live=bool(spec.get("llm_live")),
                confirm_human_gate=bool(spec.get("confirm_human_gate")),
            )
            run_path = work / "experiment_run.json"
            if not run_path.is_file():
                mgr.initialize_run(
                    round_index=int(plan.get("round_index") or 0),
                    plan_id=plan.get("plan_id"),
                )
            max_steps = max(1, int(spec.get("max_steps") or 48))
            gpu_rounds = int(spec.get("gpu_rounds") or 0)
            last = None
            for _ in range(max_steps):
                spec = _load(path)
                if spec.get("stop_requested"):
                    spec["status"] = "paused"
                    spec["ok"] = True
                    spec["updated_at"] = _now()
                    spec["last_action"] = (last.action if last else spec.get("last_action"))
                    spec.pop("worker_pid", None)
                    _dump(path, spec)
                    return
                # Gate APPROVED → next step blocks on GPU. Stamp waiting_gpu
                # before the long execute so the UI is not stuck on NEED_GATE.
                if (
                    str(spec.get("last_action") or "")
                    == OrchestrationAction.NEED_GATE.value
                    and str(spec.get("run_state") or "") == RunState.APPROVED.value
                    and bool(spec.get("execute"))
                ):
                    spec["status"] = "waiting_gpu"
                    spec["last_action"] = OrchestrationAction.NEED_EXECUTION.value
                    spec["updated_at"] = _now()
                    _dump(path, _stamp_worker(spec))
                last = mgr.step()
                if last.action == OrchestrationAction.NEED_EXECUTION.value:
                    if last.report.get("runner_called"):
                        gpu_rounds += 1
                    # Harvest/reattach finished this GPU slot — clear sticky flag.
                    spec.pop("pending_harvest", None)
                spec["gpu_rounds"] = gpu_rounds
                spec["last_action"] = last.action
                spec["run_state"] = last.state.run_state.value
                spec["updated_at"] = _now()
                spec["status"] = (
                    "waiting_gpu"
                    if last.action == OrchestrationAction.NEED_EXECUTION.value
                    and spec.get("execute")
                    and last.state.run_state == RunState.RUNNING
                    else "running"
                )
                step_row = last.to_dict()
                steps = list(spec.get("steps") or [])
                steps.append(step_row)
                spec["steps"] = steps[-40:]
                if last.action == OrchestrationAction.NEED_MEMORY.value and spec.get("sota_pursuit"):
                    from scientist_lab.services.sota_pursuit import run_sota_pursuit_tick

                    pursuit = run_sota_pursuit_tick(
                        work,
                        spec,
                        draft_runner=self._run_how_draft_arm,
                        lifecycle_runner=self._run_how_lifecycle,
                    )
                    slog = list(spec.get("sota_pursuit_log") or [])
                    slog.append(
                        {
                            "action": pursuit.get("action"),
                            "progressed": pursuit.get("progressed"),
                            "reason": pursuit.get("reason"),
                            "board": pursuit.get("board"),
                        }
                    )
                    spec["sota_pursuit_log"] = slog[-20:]
                    spec["sota_board"] = pursuit.get("board")
                    if pursuit.get("live_m1_brief"):
                        spec["live_m1_brief"] = pursuit.get("live_m1_brief")
                    if pursuit.get("draft_arm"):
                        history = list(spec.get("how_draft_arm") or [])
                        history.append(pursuit.get("draft_arm"))
                        spec["how_draft_arm"] = history[-20:]
                        spec["how_draft_arm_last"] = pursuit.get("draft_arm")
                    spec["updated_at"] = _now()
                    if pursuit.get("action") == "invent_awaiting_human":
                        spec["status"] = "waiting_human"
                        spec["last_action"] = OrchestrationAction.NEED_HUMAN.value
                        spec["error"] = str(
                            pursuit.get("reason")
                            or "invent-fallback HOW draft awaiting human review"
                        )
                        spec["ok"] = True
                        spec.pop("worker_pid", None)
                        _dump(path, spec)
                        return
                    if pursuit.get("progressed"):
                        _dump(path, _stamp_worker(spec))
                        continue
                if last.idle or last.action in _TERMINAL:
                    if (
                        last.action == OrchestrationAction.NEED_HUMAN.value
                        and spec.get("llm_how_lifecycle")
                        and spec.get("confirm_human_gate")
                    ):
                        tick = self._run_how_lifecycle(work, spec)
                        history = list(spec.get("how_lifecycle") or [])
                        history.append(tick)
                        spec["how_lifecycle"] = history[-20:]
                        spec["how_lifecycle_last"] = tick
                        if tick.get("draft_arm"):
                            dhist = list(spec.get("how_draft_arm") or [])
                            dhist.append(tick.get("draft_arm"))
                            spec["how_draft_arm"] = dhist[-20:]
                            spec["how_draft_arm_last"] = tick.get("draft_arm")
                        spec["updated_at"] = _now()
                        if tick.get("action") == "invent_awaiting_human":
                            spec["status"] = "waiting_human"
                            spec["error"] = str(
                                tick.get("reason")
                                or "invent-fallback HOW draft awaiting human review"
                            )
                            spec["ok"] = True
                            spec.pop("worker_pid", None)
                            _dump(path, spec)
                            return
                        if tick.get("progressed"):
                            spec["status"] = "running"
                            spec["error"] = None
                            _dump(path, _stamp_worker(spec))
                            continue
                    blocked = last.state.run_state in {RunState.BLOCKED, RunState.FAILED}
                    spec["status"] = "blocked" if blocked else "completed"
                    spec["ok"] = not blocked
                    spec["metrics_forged"] = False
                    if last.reasons:
                        spec["error"] = "; ".join(last.reasons) if blocked else spec.get("error")
                    spec.pop("worker_pid", None)
                    _dump(path, spec)
                    # Best-effort: slim this campaign's redundant GPU weights only.
                    if not blocked:
                        try:
                            from scientist_lab.artifacts.gc import slim_tree_weights

                            gc_report = slim_tree_weights(work)
                            spec["artifact_gc"] = {
                                "scope": str(work),
                                "bytes_reclaimed_gb": gc_report.get("bytes_reclaimed_gb"),
                                "deleted_files": gc_report.get("deleted_files"),
                            }
                            _dump(path, spec)
                        except Exception as gc_exc:  # noqa: BLE001
                            spec["artifact_gc_error"] = f"{type(gc_exc).__name__}: {gc_exc}"
                            _dump(path, spec)
                    return
                _dump(path, _stamp_worker(spec))
            spec["status"] = "blocked"
            spec["ok"] = False
            spec["error"] = "max_steps reached before idle/stop"
            spec["updated_at"] = _now()
            spec.pop("worker_pid", None)
            _dump(path, spec)
        except Exception as exc:  # noqa: BLE001 — campaign must fail closed, not forge
            spec = _load(path) if path.is_file() else {"campaign_id": campaign_id}
            spec["status"] = "failed"
            spec["ok"] = False
            spec["fail_closed"] = True
            spec["metrics_forged"] = False
            spec["error"] = f"{type(exc).__name__}: {exc}"
            spec["updated_at"] = _now()
            spec.pop("worker_pid", None)
            _dump(path, spec)
        finally:
            with self._lock:
                self._threads.pop(campaign_id, None)
                held = self._leases.pop(campaign_id, None)
            if held is not None:
                held.release()
