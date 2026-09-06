"""Feature-level RGB-T fusion interfaces and gated multiscale implementation."""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Abstract multiscale fusion: RGB features × Thermal features → fused list."""

    @abstractmethod
    def forward(
        self,
        rgb_features: list[torch.Tensor],
        thermal_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        raise NotImplementedError


class GatedFeatureFusion(nn.Module):
    """Single-scale gated blend with residual projection (stable, ablatable)."""

    def __init__(self, channels: int, *, residual: bool = True) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        self.channels = int(channels)
        self.residual = bool(residual)
        self.gate = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(self.channels * 2, self.channels, kernel_size=1)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        thermal_feature: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_feature.shape != thermal_feature.shape:
            raise ValueError(
                "RGB and thermal feature shapes must match: "
                f"{rgb_feature.shape} vs {thermal_feature.shape}"
            )
        if rgb_feature.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {rgb_feature.shape[1]}"
            )
        joined = torch.cat([rgb_feature, thermal_feature], dim=1)
        gate = self.gate(joined)
        if getattr(self, "diag_enabled", False):
            # Detached capture for mechanism diagnosis; no autograd impact.
            self._last_gate = gate.detach()
            self._last_rgb = rgb_feature.detach()
            self._last_thermal = thermal_feature.detach()
        mixed = gate * rgb_feature + (1.0 - gate) * thermal_feature
        if self.residual:
            return mixed + self.project(joined)
        return mixed


class GatedMultiscaleFusion(FeatureFusion):
    """Apply independent GatedFeatureFusion at each pyramid level."""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        residual: bool = True,
        levels: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.levels = list(levels) if levels is not None else [
            f"p{i + 3}" for i in range(len(chans))
        ]
        if len(self.levels) != len(chans):
            raise ValueError(
                f"levels length {len(self.levels)} != channels length {len(chans)}"
            )
        self.fusions = nn.ModuleList(
            [GatedFeatureFusion(c, residual=residual) for c in chans]
        )

    def forward(
        self,
        rgb_features: list[torch.Tensor],
        thermal_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(rgb_features) != len(thermal_features):
            raise ValueError(
                "rgb/thermal feature list lengths must match: "
                f"{len(rgb_features)} vs {len(thermal_features)}"
            )
        if len(rgb_features) != len(self.fusions):
            raise ValueError(
                f"expected {len(self.fusions)} feature levels, got {len(rgb_features)}"
            )
        return [
            fusion(rgb, thr)
            for fusion, rgb, thr in zip(self.fusions, rgb_features, thermal_features)
        ]
