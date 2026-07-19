from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, content: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def write_training_curve(path: Path, losses: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "loss"])
        for index, loss in enumerate(losses, start=1):
            writer.writerow([index, float(loss)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(config_path)

    contract_path = output_dir / "contract.json"
    contract = read_json(contract_path) if contract_path.exists() else {}

    seed = int(
        args.seed
        if args.seed is not None
        else config.get("seed", contract.get("seed", 42))
    )

    learning_rate = float(config.get("learning_rate", 0.001))
    epochs = int(config.get("epochs", 30))
    hidden_units = int(config.get("hidden_units", 64))
    batch_size = int(config.get("batch_size", 64))
    test_size = float(config.get("test_size", 0.2))

    random.seed(seed)
    np.random.seed(seed)

    print("Starting real Digits MLP experiment", flush=True)
    print(f"seed={seed}", flush=True)
    print(f"learning_rate={learning_rate}", flush=True)
    print(f"epochs={epochs}", flush=True)
    print(f"hidden_units={hidden_units}", flush=True)
    print(f"batch_size={batch_size}", flush=True)

    started_at = time.time()

    dataset = load_digits()
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int64)
    features = features / 16.0

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(hidden_units,),
                    learning_rate_init=learning_rate,
                    max_iter=epochs,
                    batch_size=batch_size,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=5,
                    random_state=seed,
                ),
            ),
        ]
    )

    print(
        f"Training samples={len(train_x)}, test samples={len(test_x)}",
        flush=True,
    )

    model.fit(train_x, train_y)

    predicted_labels = model.predict(test_x)
    predicted_probabilities = model.predict_proba(test_x)

    accuracy = accuracy_score(test_y, predicted_labels)
    macro_f1 = f1_score(test_y, predicted_labels, average="macro")
    evaluation_log_loss = log_loss(test_y, predicted_probabilities)

    classifier = model.named_steps["classifier"]
    duration_seconds = time.time() - started_at

    metrics = {
        "schema_version": "1.0",
        "project_id": contract.get("project_id"),
        "node_id": contract.get("node_id"),
        "primary_metric": "accuracy",
        "metrics": {
            "accuracy": round(float(accuracy), 6),
            "macro_f1": round(float(macro_f1), 6),
            "log_loss": round(float(evaluation_log_loss), 6),
        },
        "training": {
            "seed": seed,
            "epochs_requested": epochs,
            "iterations_completed": int(classifier.n_iter_),
            "hidden_units": hidden_units,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": int(len(train_x)),
            "test_samples": int(len(test_x)),
            "duration_seconds": round(duration_seconds, 3),
        },
        "status": "completed",
    }

    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "model.joblib"
    curve_path = output_dir / "training_curve.csv"
    confusion_path = output_dir / "confusion_matrix.csv"

    write_json(metrics_path, metrics)
    joblib.dump(model, model_path)
    write_training_curve(
        curve_path,
        [float(value) for value in classifier.loss_curve_],
    )
    np.savetxt(
        confusion_path,
        confusion_matrix(test_y, predicted_labels),
        delimiter=",",
        fmt="%d",
    )

    artifact_manifest = {
        "schema_version": "1.0",
        "artifacts": [
            {"type": "metrics", "path": "metrics.json", "required": True},
            {"type": "model", "path": "model.joblib", "required": True},
            {
                "type": "training_curve",
                "path": "training_curve.csv",
                "required": True,
            },
            {
                "type": "confusion_matrix",
                "path": "confusion_matrix.csv",
                "required": True,
            },
            {"type": "execution", "path": "execution.json", "required": False},
            {"type": "log", "path": "combined.log", "required": False},
        ],
    }
    write_json(output_dir / "artifact_manifest.json", artifact_manifest)

    print(f"accuracy={accuracy:.6f}", flush=True)
    print(f"macro_f1={macro_f1:.6f}", flush=True)
    print(f"log_loss={evaluation_log_loss:.6f}", flush=True)
    print(f"duration_seconds={duration_seconds:.3f}", flush=True)
    print("Real Digits MLP experiment completed", flush=True)


if __name__ == "__main__":
    main()
