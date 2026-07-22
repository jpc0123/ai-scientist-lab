"""Export frozen OpenAPI schema for /api/v1 (v1.7.1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.api.app import create_app
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def main() -> int:
    accept_root = ROOT / "outputs" / "_openapi_export"
    settings = Settings(
        project_root=ROOT,
        db_path=accept_root / "export.db",
        runtime_dir=accept_root / "runtime",
        outputs_dir=accept_root / "outputs",
        experiment_app_dir=ROOT / "experiment_app",
    ).resolve()
    service = ExperimentService(settings=settings)
    app = create_app(service=service)
    schema = app.openapi()
    out_dir = ROOT / "docs" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "openapi_v1.json"
    out_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path} ({len(schema.get('paths') or {})} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
