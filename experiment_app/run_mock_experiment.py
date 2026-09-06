from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_manifest(output_dir: Path, *, include_metrics: bool) -> None:
    artifacts = []
    if include_metrics:
        artifacts.append(
            {"type": "metrics", "path": "metrics.json", "required": True}
        )
    artifacts.extend(
        [
            {"type": "log", "path": "combined.log", "required": False},
            {"type": "execution", "path": "execution.json", "required": False},
        ]
    )
    write_json(
        output_dir / "artifact_manifest.json",
        {"schema_version": "1.0", "artifacts": artifacts},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    fail_mode = str(config.get("fail_mode", "none")).lower()
    sleep_seconds = float(config.get("sleep_seconds", 2))

    if fail_mode == "hang":
        # Used to exercise timeout / cancel paths.
        time.sleep(max(sleep_seconds, 3600))

    time.sleep(max(0.0, sleep_seconds))

    if fail_mode == "exception":
        raise RuntimeError("模拟实验主动失败 (simulated experiment exception)")

    if fail_mode == "missing_metrics":
        write_manifest(output_dir, include_metrics=False)
        print(json.dumps({"ok": False, "fail_mode": fail_mode}), flush=True)
        return

    learning_rate = float(config.get("learning_rate", 0.001))
    accuracy = 0.80 + min(learning_rate * 10, 0.05)

    metrics = {
        "schema_version": "1.0",
        "primary_metric": "accuracy",
        "metrics": {
            "accuracy": accuracy,
            "loss": round(1.0 - accuracy, 6),
        },
        "training": {
            "seed": args.seed,
            "learning_rate": learning_rate,
            "fail_mode": fail_mode,
        },
        "status": "completed",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_manifest(output_dir, include_metrics=True)
    print(json.dumps({"ok": True, "accuracy": accuracy}), flush=True)


if __name__ == "__main__":
    main()
