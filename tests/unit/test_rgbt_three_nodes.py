from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def test_three_node_contracts_share_controls():
    rgb = _load("rgbt_rgb_smoke_contract.json")
    thermal = _load("rgbt_thermal_smoke_contract.json")
    fusion = _load("rgbt_fusion_smoke_contract.json")

    assert rgb["node_id"] == "rgbt_node_001"
    assert thermal["node_id"] == "rgbt_node_002"
    assert fusion["node_id"] == "rgbt_node_003"

    shared_keys = [
        "epochs",
        "batch_size",
        "learning_rate",
        "smoke_image_width",
        "smoke_image_height",
        "max_train_images",
        "max_val_images",
        "model",
    ]
    for key in shared_keys:
        assert rgb["parameters"][key] == thermal["parameters"][key] == fusion["parameters"][key]

    assert rgb["seed"] == thermal["seed"] == fusion["seed"] == 42
    assert rgb["parameters"]["input_mode"] == "rgb"
    assert thermal["parameters"]["input_mode"] == "thermal"
    assert fusion["parameters"]["input_mode"] == "rgbt"
    assert fusion["parameters"]["fusion_method"] == "early_concat"
    assert rgb["parameters"]["fusion_method"] == thermal["parameters"]["fusion_method"] == "none"

    for contract in (rgb, thermal, fusion):
        assert contract["execution_mode"] == "smoke_train"
        assert contract["task_config"]["claim_level"] == "pipeline_validation_only"
        assert contract["dataset_reference"] == "dataset:rgbt_debug_v1"
