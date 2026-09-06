"""Gate B helper: one CUDA (or CPU) train step for A3 gated multiscale fusion.

Usage (from repo root, with REAL_APP on PYTHONPATH or cwd=app):

  python experiment_apps/rgbt_detection_real/gate_b_a3_one_batch.py

Success criteria:
  - finite loss
  - RGB backbone, thermal backbone, and fusion all receive gradients
  - optimizer.step() completes
  - no shape mismatch
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
if str(ROOT / "third_party" / "DFINE") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party" / "DFINE"))


def main() -> int:
    import torch
    from apply_gated_fusion import apply_gated_multiscale_fusion
    from dfine_config_builder import write_dfine_fast_config
    from dfine_dataset_stage import count_categories, stage_coco_for_dfine
    from models.fusion_factory import FusionConfig

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    if not data_root.is_dir():
        print(f"FAIL missing dataset: {data_root}")
        return 2

    out = APP / "_gate_b_a3_scratch"
    out.mkdir(parents=True, exist_ok=True)
    stage = stage_coco_for_dfine(
        data_root,
        out / "stage",
        input_mode="rgbt",
        fusion_method="gated_multiscale",
    )
    n_cls = count_categories(stage["train_ann"])
    dfine_root = ROOT / "third_party" / "DFINE"
    cfg_path = write_dfine_fast_config(
        dfine_root=dfine_root,
        config_path=out / "cfg.yml",
        stage_paths=stage,
        output_dir=out / "run",
        epochs=1,
        batch_size=2,
        num_workers=0,
        image_size=160,
        learning_rate=2e-4,
        num_classes=n_cls,
        seed=42,
        scale_queries_to_tokens=True,
        budget_record_path=out / "budget.json",
    )

    from src.core import YAMLConfig  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_cfg = YAMLConfig(str(cfg_path))
    yaml_cfg.yaml_cfg["use_amp"] = False
    yaml_cfg.yaml_cfg["sync_bn"] = False

    fusion_cfg = FusionConfig(
        type="gated_multiscale",
        levels=("p3", "p4", "p5"),
        share_backbone=False,
        residual=True,
    )
    summary = apply_gated_multiscale_fusion(
        yaml_cfg,
        fusion_cfg,
        thermal_train_img=Path(stage["thermal_train_img"]),
        thermal_val_img=Path(stage["thermal_val_img"]),
    )
    model = yaml_cfg.model.to(device)
    criterion = yaml_cfg.criterion.to(device)
    optimizer = yaml_cfg.optimizer
    loader = yaml_cfg.train_dataloader

    model.train()
    batch = next(iter(loader))
    samples, targets = batch
    samples = samples.to(device)
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

    if samples.shape[1] != 6:
        print(f"FAIL expected 6ch input, got {tuple(samples.shape)}")
        return 3

    optimizer.zero_grad(set_to_none=True)
    metas = {"epoch": 0, "step": 0, "global_step": 0, "epoch_step": 1}
    outputs = model(samples, targets=targets)
    loss_dict = criterion(outputs, targets, **metas)
    loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
    if not torch.isfinite(loss):
        print(f"FAIL non-finite loss: {loss}")
        return 4
    loss.backward()
    optimizer.step()
    # Refresh grads check after step still used pre-step grads — re-run one fwd for clarity.
    optimizer.zero_grad(set_to_none=True)
    outputs = model(samples, targets=targets)
    loss_dict = criterion(outputs, targets, **metas)
    loss = sum(loss_dict.values())
    loss.backward()

    backbone = model.backbone
    rgb_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in backbone.rgb_backbone.parameters()
        if p.requires_grad
    )
    thr_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in backbone.thermal_backbone.parameters()
        if p.requires_grad
    )
    fus_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in backbone.fusion.parameters()
        if p.requires_grad
    )
    if not (rgb_ok and thr_ok and fus_ok):
        print(f"FAIL gradients rgb={rgb_ok} thermal={thr_ok} fusion={fus_ok}")
        return 5
    optimizer.step()

    print("PASS Gate B A3 one-batch")
    print(
        {
            "device": str(device),
            "loss": float(loss.detach().cpu()),
            "input_shape": list(samples.shape),
            "fusion": summary,
            "cuda": device.type == "cuda",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
