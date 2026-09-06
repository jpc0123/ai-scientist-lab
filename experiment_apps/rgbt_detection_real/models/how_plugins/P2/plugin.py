"""HOW plugin P2. Implements FeatureFusion with a small gate MLP on concatenated RGB/thermal channels."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class _GateMLP(nn.Module):
    """Tiny 2-layer MLP that outputs a per-channel sigmoid gate."""

    def __init__(self, channels: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, thermal], dim=1)
        return torch.sigmoid(self.net(x))


class P2GatedFusion(FeatureFusion):
    """Per-level gated fusion using a small MLP on concatenated RGB/thermal features."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        self.gates = nn.ModuleList([_GateMLP(c) for c in chans])

    def forward(
        self, rgb_feats: Sequence[torch.Tensor], thermal_feats: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if len(rgb_feats) != len(thermal_feats):
            raise ValueError("rgb and thermal feature lists must have same length")
        if len(rgb_feats) != len(self.channels):
            raise ValueError("number of feature levels must match channels")
        out: list[torch.Tensor] = []
        for gate, r, t in zip(self.gates, rgb_feats, thermal_feats):
            g = gate(r, t)
            fused = g * r + (1 - g) * t
            if self.residual:
                fused = fused + r + t
            out.append(fused)
        return out


def build_plugin(cfg) -> P2GatedFusion:
    """Factory used by the HOW runner."""
    channels = cfg.get("channels", [64, 128, 256, 512])
    residual = cfg.get("residual", True)
    return P2GatedFusion(channels=channels, residual=residual)


def build_fusion(channels=(256, 512, 1024), residual=True, **kwargs):
    """Lab contract export; wraps LLM alias build_plugin."""
    try:
        return build_plugin(channels=channels, residual=residual, **kwargs)
    except TypeError:
        return build_plugin({"channels": list(channels), "residual": residual})
