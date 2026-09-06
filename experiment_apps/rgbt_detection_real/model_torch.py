"""Torch mini-detector stand-in for DFINE-S (v0.8.1).

Replace with vendored DFINE-S under the same baseline_key=dfine_s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def channels_for_mode(input_mode: str, fusion_method: str) -> int:
    mode = (input_mode or "rgb").strip().lower()
    if mode == "rgb":
        return 3
    if mode == "thermal":
        return 3
    if mode == "rgbt":
        return 4 if (fusion_method or "early_concat") == "early_concat" else 3
    raise ValueError(f"unsupported input_mode: {input_mode}")


class MiniDetHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.cls = nn.Linear(32, num_classes)
        self.box = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x).flatten(1)
        return self.cls(feat), self.box(feat)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def create_model(
    *,
    in_channels: int,
    seed: int,
    device: torch.device,
    num_classes: int = 10,
) -> MiniDetHead:
    torch.manual_seed(seed)
    model = MiniDetHead(in_channels=in_channels, num_classes=num_classes)
    return model.to(device)


def save_checkpoint(model: MiniDetHead, path: Path, *, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_meta = dict(meta)
    payload_meta.setdefault("num_classes", model.num_classes)
    torch.save({"model": model.state_dict(), "meta": payload_meta}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[MiniDetHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = dict(payload.get("meta") or {})
    model = MiniDetHead(
        in_channels=int(meta.get("in_channels", 3)),
        num_classes=int(meta.get("num_classes", 10)),
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, meta
