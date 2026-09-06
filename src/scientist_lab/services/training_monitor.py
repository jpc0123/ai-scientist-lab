"""Live training monitor: progress, epoch, failures for the web console."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EPOCH_RE = re.compile(
    r"(?:\"epoch\"\s*:\s*|Epoch\s*[:=]\s*|epoch\s+)(\d+)",
    re.IGNORECASE,
)
_EPOCH_BRACKET_RE = re.compile(r"Epoch:\s*\[(\d+)\s*/\s*(\d+)\]", re.IGNORECASE)
_EPOCH_STEP_RE = re.compile(
    r"Epoch:\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]",
    re.IGNORECASE,
)
_LOSS_RE = re.compile(r"\bloss:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_ETA_RE = re.compile(r"\beta:\s*([0-9:]+)", re.IGNORECASE)
_MAP_RE = re.compile(
    r"(?:mAP50[_-]?95|coco_eval_bbox|test_coco_eval_bbox)[^\d\[]{0,40}"
    r"(?:\[?\s*)(\d+\.?\d*(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"(?:RuntimeError|AttributeError|Error|Exception|Traceback|"
    r"\[worker\] failed|\[worker\] cancelled)[^\n]{0,240}",
    re.IGNORECASE,
)

ACTIVE = {"queued", "preparing", "running", "collecting", "created", "received", "validating"}
FAILED = {
    "failed",
    "cancelled",
    "timed_out",
    "validation_failed",
    "prepare_failed",
    "artifact_failed",
    "interrupted",
}
# Host waiter / API restart can leave SQLite as queued|running while the
# scientist-exec container already exited and execution.json was harvested.
_ZOMBIE_LOG_STALE_SECONDS = 180
_ZOMBIE_QUEUED_GRACE_SECONDS = 120


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _tail_text(path: Path, max_chars: int = 24_000) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def parse_training_signals(log_text: str) -> dict[str, Any]:
    """Extract latest epoch / mAP / error snippet from combined training logs."""
    if not log_text:
        return {
            "epoch": None,
            "epochs_total": None,
            "step": None,
            "steps_total": None,
            "loss": None,
            "eta": None,
            "mAP50_95": None,
            "error_snippet": None,
            "started_training": False,
            "last_lines": "",
            "status_line": "",
        }
    epochs: list[int] = []
    epochs_total = None
    step = None
    steps_total = None
    loss = None
    eta = None
    status_line = ""
    for m in _EPOCH_STEP_RE.finditer(log_text):
        epochs.append(int(m.group(1)))
        epochs_total = int(m.group(2))
        step = int(m.group(3))
        steps_total = int(m.group(4))
        status_line = m.group(0)
        # Prefer the matching line's loss/eta when present nearby.
    if not epochs:
        for m in _EPOCH_BRACKET_RE.finditer(log_text):
            epochs.append(int(m.group(1)))
            try:
                epochs_total = int(m.group(2))
            except (TypeError, ValueError):
                pass
    if not epochs:
        for m in _EPOCH_RE.finditer(log_text):
            epochs.append(int(m.group(1)))
    # Scan from the end for a compact DFINE progress line.
    for ln in reversed([x for x in log_text.splitlines() if x.strip()]):
        if "Epoch:" not in ln:
            continue
        status_line = ln.strip()
        lm = _LOSS_RE.search(ln)
        if lm:
            try:
                loss = float(lm.group(1))
            except ValueError:
                pass
        em = _ETA_RE.search(ln)
        if em:
            eta = em.group(1)
        sm = _EPOCH_STEP_RE.search(ln)
        if sm:
            step = int(sm.group(3))
            steps_total = int(sm.group(4))
            if epochs_total is None:
                epochs_total = int(sm.group(2))
        break
    maps: list[float] = []
    for m in _MAP_RE.finditer(log_text):
        try:
            maps.append(float(m.group(1)))
        except ValueError:
            continue
    err = None
    for m in _ERROR_RE.finditer(log_text):
        err = m.group(0).strip()
    # Compact UI tail: keep short status markers, not 2KB DFINE metric dumps.
    compact: list[str] = []
    for ln in log_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if (
            s.startswith("Epoch:")
            or s.startswith("[worker]")
            or "Start training" in s
            or "mAP50" in s
            or "Error" in s
            or "Traceback" in s
            or "cancelled" in s.lower()
        ):
            if s.startswith("Epoch:") and len(s) > 180:
                # Keep epoch/step/loss/eta only.
                parts = []
                em = _EPOCH_STEP_RE.search(s) or _EPOCH_BRACKET_RE.search(s)
                if em:
                    parts.append(em.group(0))
                lm = _LOSS_RE.search(s)
                if lm:
                    parts.append(f"loss: {lm.group(1)}")
                et = _ETA_RE.search(s)
                if et:
                    parts.append(f"eta: {et.group(1)}")
                s = "  ".join(parts) if parts else s[:180]
            compact.append(s)
    return {
        "epoch": epochs[-1] if epochs else None,
        "epochs_total": epochs_total,
        "step": step,
        "steps_total": steps_total,
        "loss": loss,
        "eta": eta,
        "mAP50_95": maps[-1] if maps else None,
        "error_snippet": err,
        "started_training": "Start training" in log_text or bool(epochs),
        "last_lines": "\n".join(compact[-8:]),
        "status_line": status_line[:220] if status_line else "",
    }


def parse_dfine_log_txt(path: Path) -> dict[str, Any]:
    """Parse DFINE log.txt JSON-lines for epoch/mAP."""
    if not path.is_file():
        return {"epoch": None, "mAP50_95": None, "epochs_seen": 0}
    epoch = None
    best_map = None
    last_map = None
    count = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" not in row:
                continue
            count += 1
            epoch = int(row["epoch"])
            bbox = row.get("test_coco_eval_bbox") or row.get("coco_eval_bbox")
            if isinstance(bbox, list) and bbox:
                try:
                    last_map = float(bbox[0])
                    if best_map is None or last_map > best_map:
                        best_map = last_map
                except (TypeError, ValueError):
                    pass
    except OSError:
        return {"epoch": None, "mAP50_95": None, "epochs_seen": 0}
    return {
        "epoch": epoch,
        "mAP50_95": last_map,
        "best_mAP50_95": best_map,
        "epochs_seen": count,
    }


def discover_worker_log(runtime_dir: Path, job_id: str | None) -> Path | None:
    if not job_id:
        return None
    path = runtime_dir / "scientist-worker" / "jobs" / job_id / "logs" / "combined.log"
    return path if path.is_file() else None


def resolve_execution_output_dir(
    outputs_dir: Path | str | None,
    execution_id: str | None,
    *,
    project_id: str | None = None,
) -> Path | None:
    """Locate outputs/<project>/<execution_id> for monitor / refresh reconcile."""
    if not outputs_dir or not execution_id:
        return None
    root = Path(outputs_dir)
    if not root.is_dir():
        return None
    eid = str(execution_id)
    if project_id:
        direct = root / str(project_id) / eid
        if direct.is_dir():
            return direct
    matches = sorted(root.glob(f"*/{eid}"))
    if matches:
        return matches[0]
    direct = root / eid
    return direct if direct.is_dir() else None


def _parse_iso_age_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        t0 = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - t0).total_seconds())


def inspect_scientist_exec(execution_id: str | None) -> dict[str, Any]:
    """Best-effort docker state for scientist-exec-<id> (running or exited)."""
    if not execution_id:
        return {"alive": False, "status": None, "exit_code": None, "name": None}
    eid = str(execution_id)
    suffix = eid[5:] if eid.startswith("exec_") else eid
    name = f"scientist-exec-{suffix}"
    try:
        proc = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Status}}|{{.State.ExitCode}}",
                name,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"alive": False, "status": None, "exit_code": None, "name": name}
    if proc.returncode != 0:
        return {"alive": False, "status": None, "exit_code": None, "name": name}
    text = (proc.stdout or "").strip()
    if "|" not in text:
        return {"alive": False, "status": None, "exit_code": None, "name": name}
    status_s, code_s = text.split("|", 1)
    status_s = status_s.strip().lower()
    try:
        exit_code = int(code_s.strip())
    except (TypeError, ValueError):
        exit_code = None
    alive = status_s in {"running", "created", "restarting", "paused"}
    return {
        "alive": alive,
        "status": status_s or None,
        "exit_code": exit_code,
        "name": name,
    }


def read_disk_execution_terminal(
    output_dir: Path | str | None,
) -> dict[str, Any] | None:
    """If bind-mounted outputs already harvested a terminal status, return it."""
    if output_dir is None:
        return None
    root = Path(output_dir)
    if not root.is_dir():
        return None
    execution = _safe_json(root / "execution.json") or {}
    metrics = _safe_json(root / "metrics.json") or {}
    metrics_status = str(metrics.get("status") or "").lower()
    exec_status = str(execution.get("status") or "").lower()
    return_code = execution.get("return_code")
    finished_at = execution.get("finished_at") or execution.get("updated_at")

    if exec_status == "completed" or metrics_status == "completed" or return_code == 0:
        return {
            "status": "completed",
            "completed_at": finished_at,
            "return_code": 0 if return_code is None else return_code,
            "reason": "disk_execution_completed",
        }
    if exec_status in FAILED or metrics_status in FAILED:
        return {
            "status": exec_status if exec_status in FAILED else metrics_status,
            "completed_at": finished_at,
            "return_code": return_code,
            "reason": "disk_execution_failed",
            "error_message": (
                (metrics.get("error") if isinstance(metrics.get("error"), str) else None)
                or f"disk status={exec_status or metrics_status}"
            ),
        }
    if return_code not in (None, 0):
        return {
            "status": "failed",
            "completed_at": finished_at,
            "return_code": return_code,
            "reason": "disk_nonzero_return_code",
            "error_message": f"container return_code={return_code}",
        }
    return None


def reconcile_active_execution_status(
    *,
    status: str,
    execution_id: str | None = None,
    output_dir: Path | str | None = None,
    started_at: str | None = None,
    log_path: Path | str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Demote ghost active rows when container/disk prove the run is dead or done.

    Does not invent metrics. Prefer disk harvest → docker inspect → log staleness.
    Remote worker jobs (job_id set) are left alone unless disk already terminal.
    """
    current = str(status or "unknown").lower()
    if current not in ACTIVE:
        return {"status": current, "changed": False}

    disk = read_disk_execution_terminal(output_dir)
    if disk is not None:
        return {
            "status": disk["status"],
            "changed": True,
            "completed_at": disk.get("completed_at"),
            "error_message": disk.get("error_message"),
            "demote_reason": disk.get("reason"),
            "is_active": disk["status"] in ACTIVE,
            "is_failed": disk["status"] in FAILED,
            "is_completed": disk["status"] == "completed",
        }

    # Remote worker path: without disk terminal evidence, keep SQLite status.
    if job_id:
        return {"status": current, "changed": False}

    docker = inspect_scientist_exec(execution_id)
    if docker.get("alive"):
        return {"status": "running" if current != "running" else current, "changed": current != "running"}

    exit_code = docker.get("exit_code")
    docker_status = docker.get("status")
    if docker_status == "exited":
        if exit_code == 0:
            # Exited cleanly but harvest sidecar missing — still not "live".
            return {
                "status": "interrupted",
                "changed": True,
                "completed_at": _now_iso(),
                "error_message": (
                    "scientist-exec exited 0 but SQLite still active; "
                    "host waiter likely died before finalize"
                ),
                "demote_reason": "docker_exited_zero_no_sidecar",
                "is_active": False,
                "is_failed": True,
                "is_completed": False,
            }
        if exit_code not in (None, 0):
            return {
                "status": "failed",
                "changed": True,
                "completed_at": _now_iso(),
                "error_message": f"scientist-exec exited with code {exit_code}",
                "demote_reason": "docker_exited_nonzero",
                "is_active": False,
                "is_failed": True,
                "is_completed": False,
            }

    log_mtime = None
    if log_path is not None:
        lp = Path(log_path)
        if lp.is_file():
            try:
                log_mtime = lp.stat().st_mtime
            except OSError:
                log_mtime = None
    log_fresh = bool(log_mtime) and (time.time() - float(log_mtime)) < _ZOMBIE_LOG_STALE_SECONDS
    if log_fresh:
        return {"status": current, "changed": False}

    # No training log yet → waiting for docker/GPU start, not a mid-run ghost.
    # (APPROVED/queued packs legitimately sit here until scientist-exec appears.)
    if log_mtime is None:
        return {"status": current, "changed": False}

    age = _parse_iso_age_seconds(started_at)
    # Brand-new queued rows may not have a container yet.
    if current in {"queued", "created", "received"} and (
        age is None or age < _ZOMBIE_QUEUED_GRACE_SECONDS
    ):
        return {"status": current, "changed": False}

    return {
        "status": "interrupted",
        "changed": True,
        "completed_at": _now_iso(),
        "error_message": (
            "no live scientist-exec and stale training log; "
            "removed from active monitor (API/waiter restart ghost)"
        ),
        "demote_reason": "zombie_no_container_stale_log",
        "is_active": False,
        "is_failed": True,
        "is_completed": False,
    }


def _live_scientist_exec_names() -> list[str]:
    try:
        listed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.strip()
        for line in (listed.stdout or "").splitlines()
        if line.strip().startswith("scientist-exec-")
    ]


def discover_live_exec_log(search_root: Path | str) -> Path | None:
    """Map a running scientist-exec-* container to host combined.log."""
    names = _live_scientist_exec_names()
    if not names:
        return None
    suffix = names[0].removeprefix("scientist-exec-")
    exec_id = suffix if suffix.startswith("exec_") else f"exec_{suffix}"
    hint = Path(search_root)
    outputs = None
    for parent in [hint, *hint.parents]:
        if parent.name == "outputs" and parent.is_dir():
            outputs = parent
            break
        cand = parent / "outputs"
        if cand.is_dir():
            outputs = cand
            break
    if outputs is None:
        return None
    matches = sorted(outputs.glob(f"*/{exec_id}/combined.log"))
    if matches:
        return matches[0]
    direct = outputs / exec_id / "combined.log"
    return direct if direct.is_file() else None


def _live_scientist_exec_logs(*, tail: int = 40, max_chars: int = 12_000) -> str:
    """Best-effort docker logs for scientist-exec-* CLI GPU containers."""
    names = _live_scientist_exec_names()
    if not names:
        return ""
    try:
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(int(tail)), names[0]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    text = text.strip()
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def fetch_docker_job_logs(job_id: str, *, tail: int = 120, max_chars: int = 24_000) -> str:
    """Read live container stdout when worker combined.log stops updating."""
    if not job_id:
        return ""
    name = f"scientist-worker-{job_id}"
    try:
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(int(tail)), name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    text = text.strip()
    if not text:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def prefer_fresher_log(worker_log: str, docker_log: str) -> str:
    """Prefer docker logs when they report a newer Epoch than frozen combined.log."""
    if not docker_log.strip():
        return worker_log
    if not worker_log.strip():
        return docker_log
    w = parse_training_signals(worker_log)
    d = parse_training_signals(docker_log)
    w_ep = w.get("epoch")
    d_ep = d.get("epoch")
    if d_ep is None:
        return worker_log
    if w_ep is None or int(d_ep) > int(w_ep):
        return docker_log
    w_step = w.get("step")
    d_step = d.get("step")
    if (
        w_ep is not None
        and int(d_ep) == int(w_ep)
        and d_step is not None
        and (w_step is None or int(d_step) >= int(w_step))
    ):
        return docker_log
    return worker_log


def discover_dfine_log(runtime_dir: Path, job_id: str | None, output_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if job_id:
        base = runtime_dir / "scientist-worker" / "jobs" / job_id / "output"
        candidates.extend(base.glob("**/log.txt"))
    if output_dir is not None:
        candidates.extend(Path(output_dir).glob("**/log.txt"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if path.is_file():
            return path
    return None


def load_gate_campaigns(outputs_dir: Path) -> list[dict[str, Any]]:
    """Surface active Gate J / J3 / J3b campaign meta for the monitor."""
    campaigns: list[dict[str, Any]] = []
    roots = [
        (
            "GATE_K2C",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_k_seed_sensitivity",
            "K2C_run_meta.json",
            "GATE_K2C_AUTOCHAIN.log",
        ),
        (
            "GATE_K3A",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_k_seed_sensitivity",
            "K3A_run_meta.json",
            "GATE_K3A_AUTOCHAIN.log",
        ),
        (
            "GATE_K2B",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_k_seed_sensitivity",
            "K2B_run_meta.json",
            "GATE_K2B_AUTOCHAIN.log",
        ),
        (
            "GATE_K2A",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_k_seed_sensitivity",
            "K2A_run_meta.json",
            "GATE_K2A_AUTOCHAIN.log",
        ),
        (
            "GATE_J3B",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_j_seed_diagnosis",
            "J3B_run_meta.json",
            "GATE_J3B_AUTOCHAIN.log",
        ),
        (
            "GATE_J3",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_j_seed_diagnosis",
            "J3_run_meta.json",
            "GATE_J3_AUTOCHAIN.log",
        ),
        (
            "GATE_J",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_j_seed_diagnosis",
            "run_meta.json",
            "GATE_J_AUTOCHAIN.log",
        ),
        (
            "GATE_I",
            outputs_dir / "experiments" / "v25_real_rgbt" / "gate_i_a2_formal_multiseed",
            "run_meta.json",
            "GATE_I_AUTOCHAIN.log",
        ),
    ]
    for name, folder, meta_name, log_name in roots:
        if not folder.is_dir():
            continue
        meta = _safe_json(folder / meta_name) or {}
        log_tail = _tail_text(folder / log_name, max_chars=4000)
        if not meta and not log_tail:
            continue
        status = str(meta.get("status") or "")
        note = None
        if name == "GATE_J3B":
            note = (
                "战役级 resume：复用已完成的 rep1，未完成的 rep2 会新建 execution 从头训。"
                "中断不会从 mid-run last.pth 续 epoch（确定性对照需要同起点）。"
            )
        campaigns.append(
            {
                "campaign": name,
                "path": str(folder),
                "status": status or ("unknown" if meta else "log_only"),
                "meta": meta,
                "note": note,
                "log_tail": log_tail[-800:] if log_tail else "",
                "failed": bool(
                    "error" in status.lower()
                    or "failed" in status.lower()
                    or "j3_autochain_error" in log_tail
                    or "j3b_autochain_error" in log_tail
                    or (
                        any("_EXIT=1" in ln for ln in log_tail.splitlines()[-5:])
                        if log_tail
                        else False
                    )
                ),
                "updated_at": meta.get("updated_at") or meta.get("completed_at"),
            }
        )
    # Prefer unique campaigns; keep J3 before J when both share folder
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in campaigns:
        key = c["campaign"]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def enrich_execution_row(
    *,
    attempt: dict[str, Any],
    live: dict[str, Any] | None,
    log_text: str,
    dfine: dict[str, Any] | None,
    job_id: str | None,
    output_dir: Path | str | None = None,
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    status = str(
        (live or {}).get("status")
        or attempt.get("status")
        or "unknown"
    ).lower()
    progress = (live or {}).get("progress")
    stage = (live or {}).get("stage") or (live or {}).get("message")
    error_message = (
        (live or {}).get("error_message")
        or ((attempt.get("error_json") or {}) if isinstance(attempt.get("error_json"), dict) else {}).get(
            "message"
        )
    )
    signals = parse_training_signals(log_text)
    # DFINE log.txt records completed epochs; combined.log shows the in-progress
    # Epoch: [N/M] line. Prefer the higher of the two while training.
    dfine_epoch = (dfine or {}).get("epoch")
    signal_epoch = signals.get("epoch")
    if dfine_epoch is not None and signal_epoch is not None:
        epoch = max(int(dfine_epoch), int(signal_epoch))
    else:
        epoch = signal_epoch if signal_epoch is not None else dfine_epoch
    map_val = (dfine or {}).get("mAP50_95")
    if map_val is None:
        map_val = signals.get("mAP50_95")
    best_map = (dfine or {}).get("best_mAP50_95")

    # Heuristic progress from epochs when worker only reports coarse stage progress.
    epochs_total = signals.get("epochs_total")
    params = {}
    result = attempt.get("result_json") or {}
    if isinstance(result, dict):
        contract = result.get("contract") or {}
        if isinstance(contract, dict):
            params = contract.get("parameters") or {}
    if epochs_total is None and isinstance(params, dict) and params.get("epochs") is not None:
        try:
            epochs_total = int(params["epochs"])
        except (TypeError, ValueError):
            epochs_total = None
    step = signals.get("step")
    steps_total = signals.get("steps_total")
    if status in {"queued", "preparing", "created", "received"} and (
        signals.get("started_training") or signal_epoch is not None
    ):
        status = "running"
        stage = stage or "training"

    reconciled = reconcile_active_execution_status(
        status=status,
        execution_id=attempt.get("execution_id"),
        output_dir=output_dir,
        started_at=attempt.get("started_at") or attempt.get("created_at"),
        log_path=log_path,
        job_id=job_id,
    )
    if reconciled.get("changed"):
        status = str(reconciled.get("status") or status)
        if reconciled.get("error_message"):
            error_message = reconciled.get("error_message") or error_message
        stage = stage or reconciled.get("demote_reason")

    # Stale combined.log can mix an old Epoch:[0] step with a newer dfine epoch.
    if (
        signal_epoch is not None
        and dfine_epoch is not None
        and int(signal_epoch) < int(dfine_epoch)
    ):
        step = None
        steps_total = None
    epoch_progress = None
    if epoch is not None and epochs_total and epochs_total > 0:
        # DFINE logs 0-based epochs. Prefer intra-epoch step for smoother bars.
        if status == "completed":
            epoch_progress = 1.0
        elif (
            step is not None
            and steps_total
            and int(steps_total) > 0
            and status in ACTIVE
        ):
            epoch_progress = min(
                0.99,
                max(
                    0.0,
                    (float(epoch) + (float(step) / float(steps_total)))
                    / float(epochs_total),
                ),
            )
        else:
            epoch_progress = min(
                0.99, max(0.0, (float(epoch) + 1.0) / float(epochs_total))
            )
            if signals.get("started_training") and int(epoch) == 0:
                epoch_progress = max(epoch_progress, 0.02)

    display_progress = epoch_progress if epoch_progress is not None else progress
    if status == "completed":
        display_progress = 1.0
    elif display_progress is None and status in ACTIVE:
        display_progress = 0.05

    error_snippet = error_message or signals.get("error_snippet")
    is_failed = status in FAILED or bool(error_snippet and status not in ACTIVE and status != "completed")

    # For completed runs, surface total epochs as finished (20/20 not 19/20).
    display_epoch = epoch
    if (
        status == "completed"
        and epochs_total is not None
        and epoch is not None
        and int(epoch) == int(epochs_total) - 1
    ):
        display_epoch = int(epochs_total)

    if status == "completed":
        display_progress = 1.0
        # Prefer DFINE completed epoch count when available.
        training_epochs = None
        if isinstance(dfine, dict) and dfine.get("epochs_seen") is not None:
            try:
                training_epochs = int(dfine["epochs_seen"])
            except (TypeError, ValueError):
                training_epochs = None
        if training_epochs is not None and epochs_total is not None:
            display_epoch = training_epochs

    completed_at = (
        reconciled.get("completed_at")
        if reconciled.get("changed")
        else None
    ) or attempt.get("completed_at") or (live or {}).get("finished_at")

    return {
        "execution_id": attempt.get("execution_id"),
        "node_id": attempt.get("node_id"),
        "project_id": attempt.get("project_id"),
        "runner_profile": attempt.get("runner_profile"),
        "status": status,
        "job_id": job_id,
        "progress": display_progress,
        "worker_progress": progress,
        "stage": stage,
        "epoch": display_epoch,
        "epochs_total": epochs_total,
        "step": step,
        "steps_total": steps_total,
        "loss": signals.get("loss"),
        "eta": signals.get("eta") if status in ACTIVE else None,
        "status_line": (signals.get("status_line") or "") if status in ACTIVE else "",
        "mAP50_95": map_val,
        "best_mAP50_95": best_map,
        "started_training": bool(signals.get("started_training") or (dfine or {}).get("epochs_seen")),
        "error_message": error_snippet,
        "is_active": status in ACTIVE,
        "is_failed": is_failed,
        "is_completed": status == "completed",
        "started_at": attempt.get("started_at") or attempt.get("created_at"),
        "completed_at": completed_at,
        "created_at": attempt.get("created_at"),
        "demote_reason": reconciled.get("demote_reason") if reconciled.get("changed") else None,
        # Keep UI payloads small: long DFINE lines freeze the training monitor page.
        "log_tail": (
            (signals.get("last_lines") or "")[-900:]
            if status in ACTIVE or is_failed
            else ""
        ),
        "href": f"/executions/{attempt.get('execution_id')}",
        "updated_at": _now_iso(),
        "resume_note": (
            "中断/取消后再次启动会新建 execution，并从 epoch 0 重训；"
            "不会自动加载上次 mid-run 的 last.pth（Gate J3b 确定性对照要求同起点）。"
            if status in ACTIVE or is_failed
            else None
        ),
    }


_CLI_MARKERS = (
    "execution.json",
    "metrics.json",
    "combined.log",
    "loop_report.json",
    "manager_status.json",
    "aps_lowlight.json",
)
_LOG_FRESH_SECONDS = 180


def _cli_train_dir(pack: Path) -> Path:
    for cand in (pack / "round" / "run", pack / "run"):
        if cand.is_dir():
            return cand
    return pack


def _looks_like_cli_run(path: Path) -> bool:
    if not path.is_dir():
        return False
    train = _cli_train_dir(path)
    return any((path / name).is_file() or (train / name).is_file() for name in _CLI_MARKERS)


def discover_cli_campaign_dirs(project_root: Path | str) -> list[Path]:
    """Find v2.6 CLI/GPU packs under outputs/ and .run/. Does not invent metrics."""
    root = Path(project_root)
    found: list[Path] = []
    seen: set[str] = set()
    for base in (root / "outputs", root / ".run"):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not child.name.lower().startswith("v26"):
                continue
            if child.name in seen or not _looks_like_cli_run(child):
                continue
            seen.add(child.name)
            found.append(child)
    return found


def snapshot_cli_run(pack: Path | str) -> dict[str, Any]:
    """Read-only epoch/step/status snapshot of a CLI output or Manager pack."""
    pack_path = Path(pack)
    train = _cli_train_dir(pack_path)
    execution = _safe_json(train / "execution.json") or _safe_json(pack_path / "execution.json") or {}
    metrics_file = _safe_json(train / "metrics.json") or _safe_json(pack_path / "metrics.json") or {}
    nested = metrics_file.get("metrics") if isinstance(metrics_file.get("metrics"), dict) else {}
    metrics = dict(nested or {})
    if not metrics:
        metrics = {k: v for k, v in metrics_file.items() if k in {"APS_lowlight", "APS", "mAP50_95", "AP_small"}}
    for extra in (pack_path / "aps_lowlight.json", train / "aps_lowlight.json"):
        slice_eval = _safe_json(extra) or {}
        if slice_eval.get("APS_lowlight") is not None:
            metrics.setdefault("APS_lowlight", slice_eval.get("APS_lowlight"))
            metrics.setdefault("mAP50_95_lowlight", slice_eval.get("mAP50_95_lowlight"))
    manager = (
        _safe_json(pack_path / "round" / "manager_status.json")
        or _safe_json(pack_path / "manager_status.json")
        or {}
    )
    experiment_run = (
        _safe_json(pack_path / "experiment_run.json")
        or _safe_json(pack_path / "round" / "experiment_run.json")
        or {}
    )
    loop_report = _safe_json(pack_path / "loop_report.json") or _safe_json(pack_path / "round" / "loop_report.json") or {}
    log_path = train / "combined.log"
    if not log_path.is_file():
        log_path = pack_path / "combined.log"
    if not log_path.is_file():
        live_log = discover_live_exec_log(pack_path)
        if live_log is not None:
            log_path = live_log
    log_text = _tail_text(log_path, max_chars=24_000) if log_path.is_file() else ""
    pack_parts = {part.lower() for part in pack_path.resolve().parts}
    pack_finished = bool(
        execution.get("finished_at")
        or execution.get("return_code") == 0
        or str(metrics_file.get("status") or "").lower() == "completed"
    )
    if not pack_finished and ("outputs" in pack_parts or ".run" in pack_parts):
        docker_text = _live_scientist_exec_logs()
        if docker_text:
            log_text = prefer_fresher_log(log_text, docker_text) if log_text else docker_text
    signals = parse_training_signals(log_text)
    dfine = parse_dfine_log_txt(train / "log.txt") if (train / "log.txt").is_file() else None

    started_at = execution.get("started_at") or experiment_run.get("updated_at")
    finished_at = execution.get("finished_at")
    return_code = execution.get("return_code")
    metrics_status = str(metrics_file.get("status") or "").lower()
    run_state = str(experiment_run.get("run_state") or "").upper()
    evidence_status = str(experiment_run.get("evidence_status") or "").upper()
    mtime = log_path.stat().st_mtime if log_path.is_file() else 0.0
    log_fresh = bool(mtime) and (time.time() - mtime) < _LOG_FRESH_SECONDS

    stuck_at = None
    if finished_at or metrics_status == "completed" or return_code == 0:
        status = "completed"
    elif return_code not in (None, 0) or metrics_status in FAILED or run_state in {"FAILED", "BLOCKED"}:
        status = "failed"
        stuck_at = "training"
    elif log_fresh and (signals.get("started_training") or log_text):
        status = "running"
        stuck_at = "training"
    elif run_state in {"APPROVED", "RUNNING", "EXECUTING", "COLLECTING"} and evidence_status in {
        "PENDING",
        "",
    }:
        status = "running"
        stuck_at = "training" if log_text else "docker"
    elif run_state in {"GATED"} or "HUMAN" in str(manager.get("run_state") or ""):
        status = "preparing"
        stuck_at = "gate"
    elif str(loop_report.get("stage") or manager.get("stage") or "").lower() in {"llm", "plan", "review"}:
        status = "preparing"
        stuck_at = "llm"
    elif started_at and not finished_at:
        status = "queued"
        stuck_at = "docker" if not log_text else "training"
    else:
        status = "unknown"

    reconciled = reconcile_active_execution_status(
        status=status,
        execution_id=str(
            execution.get("execution_id")
            or experiment_run.get("run_id")
            or pack_path.name
        ),
        output_dir=train if (train / "execution.json").is_file() or (train / "metrics.json").is_file() else pack_path,
        started_at=started_at,
        log_path=log_path if log_path.is_file() else None,
        job_id=None,
    )
    if reconciled.get("changed"):
        status = str(reconciled.get("status") or status)
        if status not in ACTIVE:
            stuck_at = reconciled.get("demote_reason") or stuck_at
            finished_at = finished_at or reconciled.get("completed_at")

    epoch = signals.get("epoch")
    if dfine and dfine.get("epoch") is not None:
        epoch = max(int(epoch or 0), int(dfine["epoch"])) if epoch is not None else dfine.get("epoch")
    epochs_total = signals.get("epochs_total")
    training = metrics_file.get("training") if isinstance(metrics_file.get("training"), dict) else {}
    if epochs_total is None and training.get("epochs_requested") is not None:
        try:
            epochs_total = int(training["epochs_requested"])
        except (TypeError, ValueError):
            epochs_total = None
    if status == "completed" and training.get("epochs_completed") is not None:
        try:
            epoch = int(training["epochs_completed"])
        except (TypeError, ValueError):
            pass

    progress = None
    if status == "completed":
        progress = 1.0
    elif epoch is not None and epochs_total:
        progress = min(0.99, max(0.0, (float(epoch) + 1.0) / float(epochs_total)))
    elif status in ACTIVE:
        progress = 0.05

    execution_id = (
        execution.get("execution_id")
        or experiment_run.get("run_id")
        or metrics_file.get("node_id")
        or pack_path.name
    )
    elapsed_s = None
    if started_at and finished_at:
        try:
            t0 = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(finished_at).replace("Z", "+00:00"))
            elapsed_s = max(0.0, (t1 - t0).total_seconds())
        except (TypeError, ValueError):
            elapsed_s = None
    elif started_at and status in ACTIVE:
        try:
            t0 = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            elapsed_s = max(0.0, (datetime.now(timezone.utc) - t0).total_seconds())
        except (TypeError, ValueError):
            elapsed_s = None

    log_tail = (signals.get("last_lines") or log_text or "")[-900:]
    href_id = pack_path.name
    return {
        "execution_id": execution_id,
        "node_id": experiment_run.get("plan_id") or metrics_file.get("node_id") or pack_path.name,
        "project_id": metrics_file.get("project_id") or "project_rgbt_cuda_001",
        "status": status,
        "job_id": None,
        "progress": progress,
        "stage": stuck_at or status,
        "epoch": epoch,
        "epochs_total": epochs_total,
        "step": signals.get("step"),
        "steps_total": signals.get("steps_total"),
        "loss": signals.get("loss"),
        "eta": signals.get("eta"),
        "status_line": signals.get("status_line") or "",
        "mAP50_95": (dfine or {}).get("mAP50_95") or metrics.get("mAP50_95"),
        "best_mAP50_95": (dfine or {}).get("best_mAP50_95"),
        "APS_lowlight": metrics.get("APS_lowlight"),
        "started_training": bool(signals.get("started_training") or training.get("trained")),
        "error_message": None if status == "completed" else (signals.get("error_snippet") or manager.get("error")),
        "is_active": status in ACTIVE,
        "is_failed": status in FAILED,
        "is_completed": status == "completed",
        "started_at": started_at,
        "completed_at": finished_at,
        "log_tail": log_tail,
        "href": f"/loop/{href_id}",
        "log_path": str(log_path) if log_path.is_file() else None,
        "pack_path": str(pack_path),
        "stuck_at": stuck_at,
        "elapsed_seconds": elapsed_s,
        "metrics": metrics,
        "source": "cli_output",
        "updated_at": _now_iso(),
        "resume_note": None,
    }


def load_cli_monitor_rows(project_root: Path | str) -> list[dict[str, Any]]:
    return [snapshot_cli_run(path) for path in discover_cli_campaign_dirs(project_root)]


def load_v26_campaigns(project_root: Path | str) -> list[dict[str, Any]]:
    """Surface V26.4 R0 + V26.5 packs even when no SQLite execution exists."""
    root = Path(project_root)
    campaigns: list[dict[str, Any]] = []
    r0 = root / "outputs" / "v26_r0"
    if r0.is_dir() and _looks_like_cli_run(r0):
        row = snapshot_cli_run(r0)
        campaigns.append(
            {
                "campaign": "V26_R0",
                "status": row.get("status") or "unknown",
                "failed": bool(row.get("is_failed")),
                "updated_at": row.get("completed_at") or row.get("started_at"),
                "log_tail": row.get("log_tail") or "",
                "note": "V26.4 R0 比较锚。APS_lowlight 来自 low_light_subset_v1 val，不是全集 APS/mAP。",
                "meta": {
                    "execution_id": row.get("execution_id"),
                    "APS_lowlight": row.get("APS_lowlight"),
                    "epoch": row.get("epoch"),
                    "epochs_total": row.get("epochs_total"),
                    "href": "/loop/v26_r0",
                    "log_path": row.get("log_path"),
                },
            }
        )
    v26_live = [
        path for path in discover_cli_campaign_dirs(root) if path.name.lower() not in {"v26_r0"}
    ]
    if v26_live:
        for pack in v26_live:
            row = snapshot_cli_run(pack)
            label = pack.name.upper()
            campaigns.append(
                {
                    "campaign": label,
                    "status": row.get("status") or "unknown",
                    "failed": bool(row.get("is_failed")),
                    "updated_at": row.get("completed_at") or row.get("started_at") or row.get("updated_at"),
                    "log_tail": row.get("log_tail") or "",
                    "note": f"V26.5 pack `{pack.name}`。卡点={row.get('stuck_at') or '无'}。KEEP ≠ Claim。不是 G2。",
                    "meta": {
                        "execution_id": row.get("execution_id"),
                        "pack": pack.name,
                        "APS_lowlight": row.get("APS_lowlight"),
                        "epoch": row.get("epoch"),
                        "epochs_total": row.get("epochs_total"),
                        "step": row.get("step"),
                        "steps_total": row.get("steps_total"),
                        "eta": row.get("eta"),
                        "href": f"/loop/{pack.name}",
                        "log_path": row.get("log_path"),
                        "stuck_at": row.get("stuck_at"),
                        "elapsed_seconds": row.get("elapsed_seconds"),
                    },
                }
            )
    else:
        campaigns.append(
            {
                "campaign": "V26_5",
                "status": "not_started",
                "failed": False,
                "updated_at": None,
                "log_tail": "",
                "note": "V26.5 第 1 轮还没有 outputs/ 或 .run/ 目录。当前没有 Docker 训练容器。",
                "meta": {"href": "/loop/v26_5_round1", "stuck_at": "not_started"},
            }
        )
    return campaigns
