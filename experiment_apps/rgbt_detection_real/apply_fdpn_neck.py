"""Attach FDPN as DFINE model.encoder (config-driven; keeps optimizer name groups)."""

from __future__ import annotations

from typing import Any

from models.fdpn import FDPN
from models.neck_factory import NeckConfig, NeckFactory


def apply_neck(
    yaml_cfg: Any,
    neck_config: NeckConfig,
) -> dict[str, Any]:
    """Replace or keep encoder according to ``neck_config.type``.

    Must run before optimizer/EMA are first accessed so FDPN params are trained.
    Module remains under attribute name ``encoder`` for DFINE AdamW regex groups.
    """
    model = yaml_cfg.model
    if not hasattr(model, "encoder"):
        raise TypeError(f"model has no encoder attribute: {type(model)!r}")

    eval_size = None
    if hasattr(yaml_cfg, "yaml_cfg"):
        raw = yaml_cfg.yaml_cfg.get("eval_spatial_size")
        if raw is not None:
            eval_size = tuple(int(x) for x in raw)

    existing = model.encoder
    if isinstance(existing, FDPN) and neck_config.type == "fdpn":
        raise RuntimeError("FDPN neck already applied")

    cfg = neck_config
    if neck_config.type == "fdpn" and hasattr(existing, "in_channels"):
        # Align I/O geometry with the HybridEncoder we replace.
        data = neck_config.to_dict()
        data["in_channels"] = [int(c) for c in existing.in_channels]
        if hasattr(existing, "feat_strides"):
            data["feat_strides"] = [int(s) for s in existing.feat_strides]
        if hasattr(existing, "hidden_dim"):
            # Preserve Dual/Hybrid out width unless user set a different out_channels.
            if int(neck_config.out_channels) == 256 and int(existing.hidden_dim) != 256:
                data["out_channels"] = int(existing.hidden_dim)
                data["hidden_dim"] = int(existing.hidden_dim)
            elif neck_config.hidden_dim is None:
                data["hidden_dim"] = int(existing.hidden_dim)
                data["out_channels"] = int(existing.hidden_dim)
        cfg = NeckConfig.from_mapping(data)

    new_encoder = NeckFactory.create(
        cfg,
        eval_spatial_size=eval_size,
        existing_encoder=existing,
    )
    model.encoder = new_encoder

    summary: dict[str, Any] = {
        "neck_type": cfg.type,
        "neck_config": cfg.to_dict(),
        "encoder_class": type(new_encoder).__name__,
        "encoder_params": int(sum(p.numel() for p in new_encoder.parameters())),
        "replaced_encoder_class": type(existing).__name__,
    }
    if isinstance(new_encoder, FDPN):
        summary["out_channels"] = list(new_encoder.out_channels)
        summary["out_strides"] = list(new_encoder.out_strides)
    return summary
