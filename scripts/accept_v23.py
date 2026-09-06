"""v2.3 offline acceptance: Vendor DFINE pin + staging + contract surface (zero GPU).

Does NOT run Docker, CUDA training, or formal DFINE benchmarks.

Usage:
  python scripts/accept_v23.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab import API_VERSION, __version__
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.evidence.claim_matrix import (
    evaluate_formal_dfine_claim,
    evaluate_formal_dfine_path_claim,
)
from scientist_lab.evidence.models import EvidenceRecord
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.baselines.dfine_s import DFineSBaselineAdapter
from scientist_lab.tasks.rgbt_detection.vendor_audit import (
    BASELINE_IMPLEMENTATION,
    PINNED_COMMIT,
    audit_dfine_vendor_pin,
    cuda_dockerfile_present,
    resolve_dfine_vendor_root,
    stage_dfine_vendor,
)


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v23"
    accept_root.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    # 0) Version freeze target (v2.3.8 seal)
    checks.append(
        _check(
            "package_version_2_3_0",
            __version__ == "2.3.0" and API_VERSION == "v2.3.0",
            f"pkg={__version__} api={API_VERSION}",
        )
    )

    # 1) Vendor pin
    audit = audit_dfine_vendor_pin(ROOT)
    checks.append(
        _check(
            "vendor_pin",
            audit.get("ok") is True,
            f"commit={PINNED_COMMIT[:12]} root={audit.get('vendor_root')}",
        )
    )

    # 2) Staging API smoke with a tiny fake vendor tree (avoid copying full DFINE)
    fake_root = accept_root / "fake_project"
    fake_dfine = fake_root / "third_party" / "DFINE"
    fake_dfine.mkdir(parents=True, exist_ok=True)
    (fake_dfine / "train.py").write_text("# accept_v23 fake vendor\n", encoding="utf-8")
    (fake_root / "third_party" / "VENDOR.md").write_text(
        f"Pinned commit: `{PINNED_COMMIT}`\n",
        encoding="utf-8",
    )
    stage_dir = accept_root / "workspace_stage"
    if stage_dir.exists():
        import shutil

        shutil.rmtree(stage_dir)
    staged = stage_dfine_vendor(stage_dir, project_root=fake_root)
    checks.append(
        _check(
            "vendor_stage_api",
            staged.get("ok") is True
            and Path(str(staged.get("dest") or "")).joinpath("train.py").is_file(),
            f"copied={staged.get('copied')}",
        )
    )

    # 3) Adapter contract surface
    adapter = DFineSBaselineAdapter()
    contract_path = ROOT / "examples" / "rgbt_remote_cuda_dfine_rgb_contract.json"
    contract = ExperimentContract.model_validate_json(
        contract_path.read_text(encoding="utf-8")
    )
    adapter.validate_parameters(dict(contract.parameters or {}))
    native = adapter.build_native_config(contract)
    checks.append(
        _check(
            "dfine_s_adapter_surface",
            native.get("baseline_key") == "dfine_s"
            and native.get("baseline_implementation") == BASELINE_IMPLEMENTATION
            and native.get("vendor_commit") == PINNED_COMMIT
            and contract.environment_key == "rgbt-detection-v2-cuda",
            f"impl={native.get('baseline_implementation')}",
        )
    )

    # 4) CUDA Dockerfile + image registry key (files only; no docker build)
    settings = Settings(project_root=ROOT).resolve()
    checks.append(
        _check(
            "cuda_dockerfile_and_registry",
            cuda_dockerfile_present(ROOT)
            and "rgbt-detection-v2-cuda" in settings.image_registry
            and settings.image_registry["rgbt-detection-v2-cuda"]
            == "scientist-rgbt-detection:v2-cuda",
            settings.image_registry.get("rgbt-detection-v2-cuda", ""),
        )
    )

    # 5) Config builder offline (skip if dataset missing)
    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    real_app = ROOT / "experiment_apps" / "rgbt_detection_real"
    if data_root.is_dir() and real_app.is_dir():
        sys.path.insert(0, str(real_app))
        from dfine_config_builder import write_dfine_fast_config
        from dfine_dataset_stage import count_categories, stage_coco_for_dfine

        cfg_dir = accept_root / "cfg"
        cfg_dir.mkdir(exist_ok=True)
        stage = stage_coco_for_dfine(
            data_root, cfg_dir / "stage", input_mode="rgb"
        )
        n_cls = count_categories(stage["train_ann"])
        cfg = write_dfine_fast_config(
            dfine_root=resolve_dfine_vendor_root(ROOT),
            config_path=cfg_dir / "fast.yml",
            stage_paths=stage,
            output_dir=cfg_dir / "out",
            epochs=1,
            batch_size=2,
            num_workers=0,
            image_size=160,
            learning_rate=2e-4,
            num_classes=n_cls,
            seed=42,
        )
        text = cfg.read_text(encoding="utf-8")
        checks.append(
            _check(
                "dfine_fast_config",
                "pretrained: False" in text and "num_classes:" in text,
                str(cfg),
            )
        )
    else:
        checks.append(
            _check(
                "dfine_fast_config",
                False,
                "rgbt_fast_eval_v1 or rgbt_detection_real missing",
            )
        )

    # 6) Formal DFINE claim remains blocked on stand-in evidence
    stand_in = EvidenceRecord(
        evidence_id="ev_accept_v23_standin",
        project_id="project_accept_v23",
        evidence_type="paired_comparison",
        evidence_strength="weak",
        limitations=["stand-in backend; not formal DFINE"],
    )
    claim = evaluate_formal_dfine_claim(
        project_id="project_accept_v23", records=[stand_in]
    )
    checks.append(
        _check(
            "formal_dfine_claim_blocked_on_standin",
            claim.support_status == "blocked",
            claim.support_status,
        )
    )

    # 7) Unit regression
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_dfine_vendor.py",
            "tests/unit/test_dfine_v231.py",
            "tests/unit/test_dfine_v232.py",
            "tests/unit/test_dfine_v233.py",
            "tests/unit/test_dfine_v234.py",
            "tests/unit/test_dfine_v235.py",
            "tests/unit/test_dfine_v236.py",
            "tests/unit/test_dfine_v237.py",
            "tests/unit/test_dfine_v238.py",
            "-q",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    checks.append(
        _check(
            "unit_dfine_v231_v238",
            proc.returncode == 0,
            (proc.stdout or proc.stderr or "")[-400:],
        )
    )

    # 8) Orchestrator dry-run (v2.3.2)
    orch_svc = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=accept_root / "orch.db",
            runtime_dir=accept_root / "runtime",
            outputs_dir=accept_root / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    dry = orch_svc.dfine_cuda_fast_eval(dry_run=True, probe_runtime=False)
    checks.append(
        _check(
            "dfine_cuda_fast_eval_dry_run",
            dry.get("dry_run") is True
            and dry.get("formal_success") is False
            and dry.get("exploratory_only") is True,
            dry.get("status", ""),
        )
    )

    # 9) Vendor evidence annotation offline (v2.3.3)
    from scientist_lab.tasks.rgbt_detection.dfine_evidence import (
        build_dfine_fast_eval_evidence,
    )

    vendor_ev = build_dfine_fast_eval_evidence(
        project_id="project_accept_v23",
        execution_id="exec_accept_vendor",
        metrics={"mAP50_95": 0.1},
        contract={
            "project_id": "project_accept_v23",
            "execution_mode": "fast_eval",
            "parameters": {"dfine_backend": "dfine"},
            "task_config": {"claim_level": "exploratory_comparison"},
        },
        execution_metadata={
            "standin_or_vendor": "vendor",
            "baseline_implementation": BASELINE_IMPLEMENTATION,
        },
    )
    vendor_claim = evaluate_formal_dfine_claim(
        project_id="project_accept_v23", records=[vendor_ev]
    )
    checks.append(
        _check(
            "vendor_evidence_non_standin_claim",
            vendor_ev.metric_summary.get("standin_or_vendor") == "vendor"
            and vendor_claim.support_status == "unsupported"
            and vendor_ev.evidence_strength == "weak",
            vendor_claim.support_status,
        )
    )

    # 10) CUDA formal triad dry-run (v2.3.4)
    triad = orch_svc.dfine_cuda_formal_triad(dry_run=True, probe_runtime=False)
    checks.append(
        _check(
            "dfine_cuda_formal_triad_dry_run",
            triad.get("dry_run") is True
            and triad.get("formal_success") is False
            and triad.get("formal_superiority_claimed") is False
            and (triad.get("validation") or {}).get("ok") is True
            and set((triad.get("would_run") or {}).keys())
            == {"rgb", "thermal", "fusion"},
            triad.get("status", ""),
        )
    )

    # 11) Formal path Claim Gate (v2.3.5)
    from scientist_lab.evidence.formal_dfine_gate import assess_formal_dfine_path_gate

    triad_records = [
        EvidenceRecord(
            evidence_id=f"ev_accept_{role}",
            project_id="project_accept_v23",
            evidence_type="single_execution",
            source_node_ids=[f"rgbt_formal_cuda_node_00{i}"],
            protocol_id="protocol_rgbt_cuda_001",
            claim_level="exploratory_comparison",
            evidence_strength="weak",
            limitations=["Vendor DFINE Fast Eval only."],
            metric_summary={
                "standin_or_vendor": "vendor",
                "baseline_implementation": BASELINE_IMPLEMENTATION,
                "triad_role": role,
                "evaluation_scope": "fast_eval_subset",
            },
        )
        for i, role in enumerate(("rgb", "thermal", "fusion"), start=1)
    ]
    gate = assess_formal_dfine_path_gate(triad_records)
    path_claim = evaluate_formal_dfine_path_claim(
        project_id="project_accept_v23", records=triad_records
    )
    sup_claim = evaluate_formal_dfine_claim(
        project_id="project_accept_v23", records=triad_records
    )
    gate_pkg = orch_svc.dfine_formal_path_gate("project_accept_v23")
    checks.append(
        _check(
            "formal_dfine_path_gate_triad",
            gate.get("formal_path_open") is True
            and gate.get("formal_superiority_eligible") is False
            and path_claim.support_status == "supported"
            and path_claim.claim_id == "claim_formal_dfine_path"
            and sup_claim.support_status == "unsupported"
            and gate_pkg.get("formal_success") is False,
            f"path={path_claim.support_status} sup={sup_claim.support_status}",
        )
    )

    # 12) Real acceptance contract dry-run + default SKIP (v2.3.6)
    import os as _os

    from scientist_lab.tasks.rgbt_detection.dfine_real_acceptance import (
        PIPELINE_ID,
        real_dfine_acceptance_gates,
    )

    real_plan = orch_svc.dfine_real_acceptance(
        dry_run=True, include_formal_triad=True, probe_runtime=False
    )
    skip_ready, skip_reason, _skip_gates = real_dfine_acceptance_gates(
        ROOT, environ={}
    )
    cleaned_env = {
        k: v
        for k, v in _os.environ.items()
        if k
        not in {
            "RUN_REAL_DFINE_TESTS",
            "RUN_REAL_CUDA",
            "ACCEPT_V23_REAL_TRIAD",
        }
    }
    real_script = subprocess.run(
        [sys.executable, "scripts/accept_v23_real.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=cleaned_env,
    )
    checks.append(
        _check(
            "real_acceptance_contract_and_default_skip",
            real_plan.get("dry_run") is True
            and real_plan.get("pipeline") == PIPELINE_ID
            and real_plan.get("formal_success") is False
            and real_plan.get("formal_superiority_claimed") is False
            and skip_ready is False
            and real_script.returncode == 0
            and '"overall": "skipped"' in (real_script.stdout or ""),
            f"skip_reason={skip_reason} rc={real_script.returncode}",
        )
    )

    # 13) Offline demo + doctor v2.3.7 surface
    demo = subprocess.run(
        [sys.executable, "scripts/demo_dfine_cuda_offline.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    doctor_off = orch_svc.dfine_cuda_doctor(probe_runtime=False)
    sys_doc = orch_svc.system_doctor()
    sys_ids = {c.get("id") for c in (sys_doc.get("checks") or [])}
    checks.append(
        _check(
            "demo_and_doctor_v237",
            demo.returncode == 0
            and '"overall": "passed"' in (demo.stdout or "")
            and doctor_off.get("doctor_version") == "v2.3.7"
            and doctor_off.get("ok") is True
            and "dfine_cuda" in sys_ids
            and (ROOT / "docs" / "dfine-cuda-runthrough.md").is_file(),
            f"demo_rc={demo.returncode} doctor={doctor_off.get('doctor_version')}",
        )
    )

    # 14) Seal artifacts (v2.3.8)
    manifest = ROOT / "docs" / "acceptance" / "v2.3" / "version_manifest.json"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    web_pkg = json.loads(
        (ROOT / "web" / "package.json").read_text(encoding="utf-8")
    )
    checks.append(
        _check(
            "seal_artifacts_v230",
            manifest.is_file()
            and '"version": "v2.3.0"' in manifest.read_text(encoding="utf-8")
            and "## [2.3.0]" in changelog
            and web_pkg.get("version") == "2.3.0"
            and (ROOT / "docs" / "acceptance" / "v2.3" / "README.md").is_file(),
            f"web={web_pkg.get('version')} manifest={manifest.is_file()}",
        )
    )

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    report = {
        "suite": "accept_v23",
        "version": "v2.3.0",
        "overall": "passed" if passed == total else "failed",
        "passed": passed,
        "total": total,
        "network_used": False,
        "gpu_used": False,
        "checks": checks,
        "definition": (
            "Offline acceptance for Vendor DFINE readiness "
            "(pin + stage + adapter + CUDA Dockerfile/registry + claim gate). "
            "Not a formal CUDA/DFINE scientific run."
        ),
    }
    report_path = accept_root / "accept_v23_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (accept_root / "README.md").write_text(
        """# Scientist Lab v2.3.0 Acceptance (offline)

```text
python scripts/accept_v23.py
```

Zero GPU / zero network. Demo: `scripts/demo_dfine_cuda_offline.py`.
Live CUDA DFINE: `scripts/accept_v23_real.py` (default SKIP).
Guide: `docs/dfine-cuda-runthrough.md`.
Tag target: `v2.3.0` (create only after explicit user request).
""",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
