"""Gate B helper: one CUDA train step for A4 = early_concat + FDPN neck.

Success criteria:
  - 3ch input (early_concat staging)
  - encoder is FDPN
  - finite loss
  - FDPN receives gradients
  - optimizer.step() completes
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
    from apply_fdpn_neck import apply_neck
    from dfine_config_builder import write_dfine_fast_config
    from dfine_dataset_stage import count_categories, stage_coco_for_dfine
    from models.fdpn import FDPN
    from models.neck_factory import NeckConfig

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    if not data_root.is_dir():
        print(f"FAIL missing dataset: {data_root}")
        return 2

    out = APP / "_gate_b_a4_scratch"
    out.mkdir(parents=True, exist_ok=True)
    stage = stage_coco_for_dfine(
        data_root,
        out / "stage",
        input_mode="rgbt",
        fusion_method="early_concat",
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

    neck_summary = apply_neck(yaml_cfg, NeckConfig(type="fdpn"))
    model = yaml_cfg.model.to(device)
    criterion = yaml_cfg.criterion.to(device)
    optimizer = yaml_cfg.optimizer
    loader = yaml_cfg.train_dataloader

    if not isinstance(model.encoder, FDPN):
        print(f"FAIL encoder is {type(model.encoder).__name__}, expected FDPN")
        return 3

    model.train()
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

    if samples.shape[1] != 3:
        print(f"FAIL expected 3ch early_concat input, got {tuple(samples.shape)}")
        return 4

    metas = {"epoch": 0, "step": 0, "global_step": 0, "epoch_step": 1}
    optimizer.zero_grad(set_to_none=True)
    outputs = model(samples, targets=targets)
    loss_dict = criterion(outputs, targets, **metas)
    loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
    if not torch.isfinite(loss):
        print(f"FAIL non-finite loss: {loss}")
        return 5
    loss.backward()

    neck_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.encoder.parameters()
        if p.requires_grad
    )
    if not neck_ok:
        print("FAIL FDPN has no finite gradients")
        return 6
    optimizer.step()

    print("PASS Gate B A4 one-batch")
    print(
        {
            "device": str(device),
            "loss": float(loss.detach().cpu()),
            "input_shape": list(samples.shape),
            "neck": neck_summary,
            "cuda": device.type == "cuda",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
