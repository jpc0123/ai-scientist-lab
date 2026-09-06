"""Reattach to an existing scientist-exec-* GPU job and harvest artifacts.

Host waiters (API thread / CLI) can die while the container keeps writing
bind-mounted outputs. Resume must wait + collect — never containers.run again.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.domain.models import utc_now_iso


@dataclass(frozen=True)
class LiveExecBinding:
    container_name: str
    execution_id: str
    output_dir: Path
    status: str  # running | exited | dead | created | ...
    exit_code: int | None = None


def execution_id_from_container_name(name: str) -> str:
    token = str(name or "").strip().lstrip("/")
    suffix = token.removeprefix("scientist-exec-")
    if suffix.startswith("exec_"):
        return suffix
    return f"exec_{suffix}" if suffix else ""


def _docker_json(args: list[str], *, timeout: float = 8.0) -> Any:
    try:
        proc = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def list_scientist_exec_names(*, all_containers: bool = True) -> list[str]:
    fmt = "{{.Names}}"
    cmd = ["ps", "-a", "--format", fmt] if all_containers else ["ps", "--format", fmt]
    try:
        proc = subprocess.run(
            ["docker", *cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.strip().lstrip("/")
        for line in (proc.stdout or "").splitlines()
        if line.strip().lstrip("/").startswith("scientist-exec-")
    ]


def resolve_output_dir(
    outputs_root: Path,
    execution_id: str,
    *,
    project_id: str | None = None,
) -> Path | None:
    root = Path(outputs_root)
    if not root.is_dir() or not execution_id:
        return None
    if project_id:
        direct = root / project_id / execution_id
        if direct.is_dir():
            return direct
    matches = sorted(root.glob(f"*/{execution_id}"))
    if matches:
        return matches[0]
    direct = root / execution_id
    return direct if direct.is_dir() else None


def discover_live_exec_bindings(
    outputs_root: Path,
    *,
    project_id: str | None = None,
    all_containers: bool = True,
) -> list[LiveExecBinding]:
    """Map scientist-exec-* containers to host output dirs."""
    out: list[LiveExecBinding] = []
    for name in list_scientist_exec_names(all_containers=all_containers):
        exec_id = execution_id_from_container_name(name)
        output_dir = resolve_output_dir(outputs_root, exec_id, project_id=project_id)
        if output_dir is None:
            continue
        status, code = inspect_container(name)
        out.append(
            LiveExecBinding(
                container_name=name,
                execution_id=exec_id,
                output_dir=output_dir,
                status=status or "unknown",
                exit_code=code,
            )
        )
    return out


def inspect_container(name: str) -> tuple[str | None, int | None]:
    data = _docker_json(
        ["inspect", "--format", "{{json .State}}", name],
        timeout=8,
    )
    if not isinstance(data, Mapping):
        return None, None
    status = str(data.get("Status") or "") or None
    code = data.get("ExitCode")
    try:
        exit_code = int(code) if code is not None else None
    except (TypeError, ValueError):
        exit_code = None
    return status, exit_code


def wait_existing_container(
    name: str,
    *,
    timeout_seconds: float = 86_400.0,
    poll_interval_seconds: float = 2.0,
) -> int:
    """Block until container exits. Does not create or remove containers."""
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        status, code = inspect_container(name)
        if status in {"exited", "dead"}:
            if code is not None:
                return int(code)
            break
        if status is None:
            # Gone from docker — treat as finished unknown.
            return -1
        if time.monotonic() > deadline:
            raise TimeoutError(f"reattach wait timed out for {name}")
        time.sleep(max(0.2, float(poll_interval_seconds)))
    wait_raw = _docker_json(["wait", name], timeout=max(30.0, timeout_seconds))
    if isinstance(wait_raw, Mapping) and wait_raw.get("StatusCode") is not None:
        return int(wait_raw["StatusCode"])
    if isinstance(wait_raw, int):
        return wait_raw
    # docker wait prints a bare integer line when not using json format
    try:
        proc = subprocess.run(
            ["docker", "wait", name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(30.0, timeout_seconds),
            check=False,
        )
        return int((proc.stdout or "").strip() or "-1")
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1


def write_execution_sidecar(
    output_dir: Path,
    *,
    execution_id: str,
    container_name: str,
    container_id: str | None = None,
    return_code: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "execution.json"
    payload = {
        "execution_id": execution_id,
        "container_name": container_name,
        "container_id": container_id,
        "return_code": return_code,
        "finished_at": utc_now_iso() if return_code is not None else None,
        "harvested": True,
        "updated_at": utc_now_iso(),
    }
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                payload = {**old, **{k: v for k, v in payload.items() if v is not None}}
        except (OSError, json.JSONDecodeError):
            pass
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Durable pointer for campaign reattach without scanning docker.
    pointer = output_dir / "live_execution.json"
    pointer.write_text(
        json.dumps(
            {
                "execution_id": execution_id,
                "container_name": container_name,
                "container_id": container_id,
                "output_directory": str(output_dir),
                "updated_at": utc_now_iso(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def metrics_ready(output_dir: Path) -> bool:
    return (Path(output_dir) / "metrics.json").is_file()


def contract_identity(contract: Mapping[str, Any] | None) -> dict[str, Any]:
    """Stable keys that bind a GPU artifact dir to one ExperimentContract / round."""
    if not isinstance(contract, Mapping):
        return {}
    mat = (
        contract.get("materialization")
        if isinstance(contract.get("materialization"), Mapping)
        else {}
    )
    how = mat.get("how") if isinstance(mat.get("how"), Mapping) else {}
    params = dict(contract.get("parameters") or {})
    if not params:
        params = dict(contract.get("legacy_parameters") or {})
    if not params:
        params = dict(mat.get("legacy_parameters") or {})
    if not params and how:
        params = dict(how.get("legacy_parameters") or {})
        if how.get("fusion_method") is not None:
            params.setdefault("fusion_method", how.get("fusion_method"))
        if how.get("input_mode") is not None:
            params.setdefault("input_mode", how.get("input_mode"))
        if how.get("neck_type") is not None:
            params.setdefault("neck", {"type": how.get("neck_type")})
    neck = params.get("neck") if isinstance(params.get("neck"), Mapping) else {}
    node = str(
        contract.get("node_id")
        or contract.get("run_id")
        or contract.get("plan_id")
        or ""
    ).strip()
    epochs = params.get("epochs")
    try:
        epochs_i = int(epochs) if epochs is not None else None
    except (TypeError, ValueError):
        epochs_i = None
    return {
        "node_id": node,
        "seed": contract.get("seed"),
        "input_mode": params.get("input_mode"),
        "fusion_method": params.get("fusion_method"),
        "neck_type": (neck or {}).get("type") or params.get("neck_type"),
        "epochs": epochs_i,
        "project_id": contract.get("project_id"),
    }


def load_contract_identity(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    for name in ("contract.json", "_legacy_fast_eval_contract.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, Mapping):
            ident = contract_identity(data)
            if ident.get("node_id") or ident.get("fusion_method") is not None:
                # Prefer on-disk train knobs over a stale/mismatched contract stamp.
                # (r3 LONGTRAIN bug: node_id matched P3 plan but artifacts were F0@2ep.)
                artifact = load_artifact_identity(root)
                if artifact.get("fusion_method") is not None or artifact.get("epochs") is not None:
                    for key in ("fusion_method", "input_mode", "epochs", "neck_type"):
                        if artifact.get(key) is not None:
                            ident[key] = artifact[key]
                return ident
    artifact = load_artifact_identity(root)
    if artifact:
        return artifact
    return {}


def load_artifact_identity(output_dir: Path) -> dict[str, Any]:
    """Read fusion/epochs from executed artifacts (config / model_summary), not plan prose."""
    root = Path(output_dir)
    out: dict[str, Any] = {}
    for name in ("config.json", "model_summary.json", "dfine_subset.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        if data.get("fusion_method") is not None:
            out.setdefault("fusion_method", data.get("fusion_method"))
        if data.get("input_mode") is not None:
            out.setdefault("input_mode", data.get("input_mode"))
        if data.get("epochs") is not None:
            try:
                out.setdefault("epochs", int(data.get("epochs")))
            except (TypeError, ValueError):
                pass
        if data.get("epochs_requested") is not None and "epochs" not in out:
            try:
                out["epochs"] = int(data.get("epochs_requested"))
            except (TypeError, ValueError):
                pass
        neck = data.get("neck") if isinstance(data.get("neck"), Mapping) else {}
        if (neck or {}).get("type") or data.get("neck_type"):
            out.setdefault("neck_type", (neck or {}).get("type") or data.get("neck_type"))
    return out


def _identity_value_equal(key: str, expected: Any, actual: Any) -> bool:
    if key == "epochs":
        try:
            return int(expected) == int(actual)
        except (TypeError, ValueError):
            return False
    ev = str(expected)
    av = str(actual or "")
    # Plugin tokens are case-insensitive (plugin:P2 vs plugin:p2).
    if key == "fusion_method":
        return ev.lower() == av.lower()
    return ev == av


def identities_match(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
) -> bool:
    """Bind harvest to one round: node_id *and* any declared HOW knobs.

    Matching only on node_id previously allowed a plan that requested
    fusion_method=plugin:P2 to harvest a same-node container that actually
    trained plugin:P3A (plan→metrics mis-attribution).
    """
    if not expected:
        return False
    exp = {k: v for k, v in dict(expected).items() if v not in (None, "")}
    act = {k: v for k, v in dict(actual or {}).items() if v not in (None, "")}
    if not exp:
        return False
    node = exp.get("node_id")
    if node and str(act.get("node_id") or "") != str(node):
        return False
    for key in ("seed", "input_mode", "fusion_method", "neck_type", "epochs", "project_id"):
        if key in exp and not _identity_value_equal(key, exp[key], act.get(key)):
            return False
    if node:
        return True
    return any(k in exp for k in ("fusion_method", "neck_type", "seed", "epochs"))


def pick_binding_for_harvest(
    bindings: list[LiveExecBinding],
    *,
    prefer_running: bool = True,
    expected: Mapping[str, Any] | None = None,
) -> LiveExecBinding | None:
    if not bindings:
        return None
    matched = bindings
    if expected:
        matched = [
            b
            for b in bindings
            if identities_match(expected, load_contract_identity(b.output_dir))
        ]
        if not matched:
            return None
    running = [b for b in matched if b.status == "running"]
    if prefer_running and running:
        return running[0]
    exited = [b for b in matched if b.status in {"exited", "dead"}]
    if exited:
        return exited[0]
    return matched[0]


def harvest_binding(
    binding: LiveExecBinding,
    *,
    wait: bool = True,
    timeout_seconds: float = 86_400.0,
    remove_container: bool = False,
) -> dict[str, Any]:
    """Wait (optional) + ensure execution.json + return orchestrator-shaped payload."""
    code = binding.exit_code
    status = binding.status
    if wait and status == "running":
        code = wait_existing_container(
            binding.container_name,
            timeout_seconds=timeout_seconds,
        )
        status = "exited"
    elif status in {"exited", "dead"} and code is None:
        _, code = inspect_container(binding.container_name)

    write_execution_sidecar(
        binding.output_dir,
        execution_id=binding.execution_id,
        container_name=binding.container_name,
        return_code=code,
    )

    ok = code == 0 and metrics_ready(binding.output_dir)
    if code == 0 and not metrics_ready(binding.output_dir):
        # Brief grace for final metric flush after exit.
        for _ in range(10):
            time.sleep(0.5)
            if metrics_ready(binding.output_dir):
                ok = True
                break

    if remove_container and status in {"exited", "dead"}:
        subprocess.run(
            ["docker", "rm", "-f", binding.container_name],
            capture_output=True,
            check=False,
            timeout=30,
        )

    run_status = "completed" if ok else "failed"
    return {
        "orchestrator": "exec_reattach",
        "exploratory_only": True,
        "formal_success": False,
        "dry_run": False,
        "reattached": True,
        "status": run_status,
        "run": {
            "execution_id": binding.execution_id,
            "status": run_status,
            "return_code": code,
            "output_directory": str(binding.output_dir),
            "container_name": binding.container_name,
        },
    }


def _read_live_pointer(dest: Path) -> dict[str, Any] | None:
    path = Path(dest) / "live_execution.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def try_harvest_existing(
    *,
    outputs_root: Path,
    dest: Path,
    project_id: str | None = None,
    wait: bool = True,
    timeout_seconds: float = 86_400.0,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Harvest only when artifacts belong to *this* contract / round.

    Orphan waiter recovery: dest may hold live_execution.json + matching legacy
    contract for the in-flight run. New rounds must not reuse an unrelated
    exited scientist-exec-* or the newest metrics under outputs/.
    """
    dest = Path(dest)
    expected_id = contract_identity(expected) if expected else dict(expected or {})
    if expected and not expected_id:
        expected_id = dict(expected)

    # Same-round dest already has metrics (partial sync after container exit).
    if metrics_ready(dest) and (dest / "execution.json").is_file():
        dest_ident = load_contract_identity(dest)
        if identities_match(expected_id, dest_ident):
            return {
                "orchestrator": "exec_reattach",
                "exploratory_only": True,
                "formal_success": False,
                "dry_run": False,
                "reattached": True,
                "status": "completed",
                "run": {
                    "execution_id": None,
                    "status": "completed",
                    "output_directory": str(dest),
                    "source": "dest_artifacts",
                },
            }

    # Prefer the durable pointer written when *this* dest started a container.
    pointer = _read_live_pointer(dest)
    pointer_exec = str((pointer or {}).get("execution_id") or "").strip()
    bindings = discover_live_exec_bindings(
        Path(outputs_root),
        project_id=project_id,
        all_containers=True,
    )
    if pointer_exec:
        pointed = [b for b in bindings if b.execution_id == pointer_exec]
        binding = pick_binding_for_harvest(pointed, expected=expected_id or None)
        if binding is None and pointed:
            # Pointer is for this dest's job; allow harvest if dest legacy matches.
            if identities_match(expected_id, load_contract_identity(dest)):
                binding = pick_binding_for_harvest(pointed, expected=None)
        if binding is not None:
            payload = harvest_binding(
                binding,
                wait=wait,
                timeout_seconds=timeout_seconds,
                remove_container=False,
            )
            from scientist_lab.adapters.dfine.cuda_runner import _sync_output_dir

            _sync_output_dir(payload, dest)
            return payload

    # Otherwise only an explicitly matching scientist-exec for this node_id/HOW.
    binding = pick_binding_for_harvest(bindings, expected=expected_id or None)
    if binding is None:
        return None

    payload = harvest_binding(
        binding,
        wait=wait,
        timeout_seconds=timeout_seconds,
        remove_container=False,
    )
    from scientist_lab.adapters.dfine.cuda_runner import _sync_output_dir

    _sync_output_dir(payload, dest)
    return payload
