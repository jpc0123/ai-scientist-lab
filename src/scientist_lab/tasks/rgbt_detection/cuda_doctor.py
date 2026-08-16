"""CUDA + DFINE runtime doctor (v2.3.2 + v2.3.7 GPU depth).

Offline-safe: never fails CI when GPU/Docker image is absent — reports readiness.
Live Fast Eval still requires explicit accept_v23_real gates.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from scientist_lab.tasks.rgbt_detection.vendor_audit import (
    audit_dfine_vendor_pin,
    cuda_dockerfile_present,
)


CUDA_ENV_KEY = "rgbt-detection-v2-cuda"
CUDA_IMAGE = "scientist-rgbt-detection:v2-cuda"
EXAMPLE_CONTRACT = "examples/rgbt_remote_cuda_dfine_rgb_contract.json"
FORMAL_TRIAD_CONTRACTS = (
    "examples/rgbt_formal_cuda_rgb_contract.json",
    "examples/rgbt_formal_cuda_thermal_contract.json",
    "examples/rgbt_formal_cuda_fusion_contract.json",
)
CUDA_PROTOCOL = "examples/rgbt_protocol_cuda.json"


def _run_cmd(args: list[str], *, timeout: float = 8.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "executable not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _parse_nvidia_csv_row(line: str) -> dict[str, str]:
    """Parse `name, driver_version, memory.total` CSV row from nvidia-smi."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 3:
        return {
            "name": parts[0],
            "driver_version": parts[1],
            "memory_total": parts[2],
        }
    if len(parts) == 1:
        return {"name": parts[0], "driver_version": "", "memory_total": ""}
    return {"name": line, "driver_version": "", "memory_total": ""}


def probe_nvidia_smi() -> dict[str, Any]:
    code, out, err = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if code != 0:
        return {
            "ok": False,
            "available": False,
            "gpu_count": 0,
            "gpus": [],
            "detail": err or out or f"exit={code}",
        }
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    gpus = [_parse_nvidia_csv_row(ln) for ln in lines]
    return {
        "ok": True,
        "available": bool(gpus),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "detail": (
            f"{gpus[0]['name']} x{len(gpus)}" if gpus else "no gpu rows"
        ),
    }

def probe_docker_daemon() -> dict[str, Any]:
    if shutil.which("docker") is None:
        return {"ok": False, "available": False, "detail": "docker not on PATH"}
    code, out, err = _run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"])
    if code != 0:
        return {
            "ok": False,
            "available": False,
            "detail": err or out or f"exit={code}",
        }
    return {"ok": True, "available": True, "detail": out or "docker ok"}


def probe_docker_nvidia_runtime() -> dict[str, Any]:
    """Detect whether the Docker daemon advertises an nvidia runtime."""
    if shutil.which("docker") is None:
        return {
            "ok": False,
            "available": False,
            "detail": "docker not on PATH",
        }
    code, out, err = _run_cmd(
        ["docker", "info", "--format", "{{json .Runtimes}}"]
    )
    if code != 0:
        return {
            "ok": False,
            "available": False,
            "detail": err or out or f"exit={code}",
        }
    runtimes_raw = out or "{}"
    names: list[str] = []
    try:
        parsed = json.loads(runtimes_raw)
        if isinstance(parsed, dict):
            names = sorted(str(k) for k in parsed.keys())
    except json.JSONDecodeError:
        names = re.findall(r'"([A-Za-z0-9_-]+)"\s*:', runtimes_raw)
        if not names:
            names = [
                tok.strip()
                for tok in re.split(r"[\s,{}]+", runtimes_raw)
                if tok.strip() and tok.strip() not in {"null", "true", "false"}
            ]
    has_nvidia = any(n.lower() == "nvidia" for n in names)
    return {
        "ok": has_nvidia,
        "available": has_nvidia,
        "runtimes": names,
        "detail": (
            "nvidia runtime present"
            if has_nvidia
            else f"nvidia runtime missing (have: {', '.join(names) or 'none'})"
        ),
        "install_hint": (
            "Install NVIDIA Container Toolkit and restart Docker "
            "(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)"
        ),
    }


def probe_cuda_image(*, image: str = CUDA_IMAGE) -> dict[str, Any]:
    code, out, err = _run_cmd(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"]
    )
    if code != 0:
        return {
            "ok": False,
            "present": False,
            "image": image,
            "detail": err or out or "image missing",
            "build_hint": (
                f"docker build -t {image} "
                f"-f dockerfiles/rgbt-detection-v2-cuda/Dockerfile ."
            ),
        }
    return {
        "ok": True,
        "present": True,
        "image": image,
        "id": out[:19] if out else "",
        "detail": "present",
    }


def summarize_gpu_readiness(
    *,
    nvidia: dict[str, Any] | None,
    docker_nvidia: dict[str, Any] | None,
    cuda_image: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compact GPU readiness block for demos / System Doctor."""
    nvidia = nvidia or {}
    docker_nvidia = docker_nvidia or {}
    cuda_image = cuda_image or {}
    blockers: list[str] = []
    if not nvidia.get("ok"):
        blockers.append("nvidia-smi unavailable")
    if not docker_nvidia.get("ok"):
        blockers.append("docker nvidia runtime missing")
    if not cuda_image.get("ok"):
        blockers.append("CUDA DFINE image missing")
    return {
        "gpu_count": int(nvidia.get("gpu_count") or 0),
        "gpus": list(nvidia.get("gpus") or []),
        "nvidia_smi_ok": bool(nvidia.get("ok")),
        "docker_nvidia_runtime_ok": bool(docker_nvidia.get("ok")),
        "cuda_image_ok": bool(cuda_image.get("ok")),
        "ready_for_gpu_container": (
            bool(nvidia.get("ok"))
            and bool(docker_nvidia.get("ok"))
            and bool(cuda_image.get("ok"))
        ),
        "blockers": blockers,
    }


def build_dfine_cuda_doctor(
    project_root: Path | str,
    *,
    image_registry: dict[str, str] | None = None,
    probe_runtime: bool = True,
) -> dict[str, Any]:
    """Compose a readiness report for CUDA Vendor DFINE Fast Eval."""
    root = Path(project_root).resolve()
    registry = dict(image_registry or {})
    expected_image = registry.get(CUDA_ENV_KEY, CUDA_IMAGE)

    vendor = audit_dfine_vendor_pin(root)
    dockerfile = cuda_dockerfile_present(root)
    contract = root / EXAMPLE_CONTRACT
    protocol = root / CUDA_PROTOCOL
    triad_paths = [root / rel for rel in FORMAL_TRIAD_CONTRACTS]
    triad_ok = all(p.is_file() for p in triad_paths)

    checks: list[dict[str, Any]] = [
        {
            "id": "vendor_pin",
            "ok": bool(vendor.get("ok")),
            "level": "ok" if vendor.get("ok") else "error",
            "detail": vendor.get("vendor_root") or "missing",
        },
        {
            "id": "cuda_dockerfile",
            "ok": dockerfile,
            "level": "ok" if dockerfile else "error",
            "detail": str(
                root / "dockerfiles" / "rgbt-detection-v2-cuda" / "Dockerfile"
            ),
        },
        {
            "id": "image_registry_key",
            "ok": CUDA_ENV_KEY in registry
            and registry.get(CUDA_ENV_KEY) == expected_image,
            "level": "ok" if CUDA_ENV_KEY in registry else "warning",
            "detail": registry.get(CUDA_ENV_KEY, "missing"),
        },
        {
            "id": "example_contract",
            "ok": contract.is_file(),
            "level": "ok" if contract.is_file() else "error",
            "detail": str(contract),
        },
        {
            "id": "cuda_protocol",
            "ok": protocol.is_file(),
            "level": "ok" if protocol.is_file() else "warning",
            "detail": str(protocol),
        },
        {
            "id": "formal_triad_contracts",
            "ok": triad_ok,
            "level": "ok" if triad_ok else "warning",
            "detail": (
                f"{sum(1 for p in triad_paths if p.is_file())}/3 present"
            ),
        },
    ]

    runtime: dict[str, Any] = {
        "probed": False,
        "docker": None,
        "cuda_image": None,
        "nvidia_smi": None,
        "docker_nvidia_runtime": None,
        "gpu": None,
    }
    if probe_runtime:
        runtime["probed"] = True
        docker = probe_docker_daemon()
        runtime["docker"] = docker
        checks.append(
            {
                "id": "docker_daemon",
                "ok": bool(docker.get("ok")),
                "level": "ok" if docker.get("ok") else "warning",
                "detail": docker.get("detail"),
            }
        )
        if docker.get("ok"):
            image = probe_cuda_image(image=expected_image)
            runtime["cuda_image"] = image
            checks.append(
                {
                    "id": "cuda_image",
                    "ok": bool(image.get("ok")),
                    "level": "ok" if image.get("ok") else "warning",
                    "detail": image.get("detail"),
                    "build_hint": image.get("build_hint"),
                }
            )
            nvidia_rt = probe_docker_nvidia_runtime()
            runtime["docker_nvidia_runtime"] = nvidia_rt
            checks.append(
                {
                    "id": "docker_nvidia_runtime",
                    "ok": bool(nvidia_rt.get("ok")),
                    "level": "ok" if nvidia_rt.get("ok") else "warning",
                    "detail": nvidia_rt.get("detail"),
                    "install_hint": nvidia_rt.get("install_hint"),
                }
            )
        else:
            checks.append(
                {
                    "id": "cuda_image",
                    "ok": False,
                    "level": "warning",
                    "detail": "skipped (docker unavailable)",
                }
            )
            checks.append(
                {
                    "id": "docker_nvidia_runtime",
                    "ok": False,
                    "level": "warning",
                    "detail": "skipped (docker unavailable)",
                }
            )
        nvidia = probe_nvidia_smi()
        runtime["nvidia_smi"] = nvidia
        checks.append(
            {
                "id": "nvidia_smi",
                "ok": bool(nvidia.get("ok")),
                "level": "ok" if nvidia.get("ok") else "warning",
                "detail": nvidia.get("detail"),
                "gpu_count": nvidia.get("gpu_count"),
            }
        )
        runtime["gpu"] = summarize_gpu_readiness(
            nvidia=nvidia,
            docker_nvidia=runtime.get("docker_nvidia_runtime"),
            cuda_image=runtime.get("cuda_image"),
        )

    errors = sum(1 for c in checks if c.get("level") == "error")
    warnings = sum(1 for c in checks if c.get("level") == "warning")
    if errors:
        overall = "error"
    elif warnings:
        overall = "warning"
    else:
        overall = "ok"

    # Core live gate stays: vendor + dockerfile + docker + image + nvidia-smi.
    # docker nvidia runtime is reported separately (preferred, not always required
    # when runners use --gpus / device_requests).
    live_ready = (
        bool(vendor.get("ok"))
        and dockerfile
        and bool((runtime.get("docker") or {}).get("ok"))
        and bool((runtime.get("cuda_image") or {}).get("ok"))
        and bool((runtime.get("nvidia_smi") or {}).get("ok"))
    )
    gpu_block = runtime.get("gpu") or {}

    return {
        "ok": errors == 0,
        "overall": overall,
        "live_ready": live_ready,
        "gpu_container_preferred": bool(gpu_block.get("ready_for_gpu_container")),
        "environment_key": CUDA_ENV_KEY,
        "image": expected_image,
        "example_contract": str(contract),
        "checks": checks,
        "runtime": runtime,
        "vendor": vendor,
        "docs": {
            "runthrough": "docs/dfine-cuda-runthrough.md",
            "plan": "docs/plans/第二十三步制作方案.md",
        },
        "next_steps": [
            "Offline demo: python scripts/demo_dfine_cuda_offline.py",
            "Doctor: scientist-lab dfine-cuda-doctor",
            "Build image if missing: see cuda_image.build_hint",
            "Set RUN_REAL_DFINE_TESTS=1 and RUN_REAL_CUDA=1 for accept_v23_real",
            "Live pipeline: scientist-lab dfine-real-acceptance --execute",
            "Guide: docs/dfine-cuda-runthrough.md",
        ],
        "gpu_required_for_live": True,
        "network_used": False,
        "doctor_version": "v2.3.7",
    }


def dfine_cuda_doctor_json(
    project_root: Path | str,
    *,
    image_registry: dict[str, str] | None = None,
    probe_runtime: bool = True,
) -> str:
    report = build_dfine_cuda_doctor(
        project_root,
        image_registry=image_registry,
        probe_runtime=probe_runtime,
    )
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"
