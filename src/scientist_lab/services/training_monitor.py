"""Live training monitor: progress, epoch, failures for the web console."""

from __future__ import annotations

import json
import re
import subprocess
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
FAILED = {"failed", "cancelled", "timed_out", "validation_failed", "prepare_failed", "artifact_failed"}


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
        "eta": signals.get("eta"),
        "status_line": signals.get("status_line") or "",
        "mAP50_95": map_val,
        "best_mAP50_95": best_map,
        "started_training": bool(signals.get("started_training") or (dfine or {}).get("epochs_seen")),
        "error_message": error_snippet,
        "is_active": status in ACTIVE,
        "is_failed": is_failed,
        "is_completed": status == "completed",
        "started_at": attempt.get("started_at") or attempt.get("created_at"),
        "completed_at": attempt.get("completed_at") or (live or {}).get("finished_at"),
        "created_at": attempt.get("created_at"),
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
