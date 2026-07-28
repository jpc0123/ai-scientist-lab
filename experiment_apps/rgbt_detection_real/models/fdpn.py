"""Frequency-Dynamic Pyramid Network (FDPN) — DFINE-compatible encoder/neck.

I/O contract matches HybridEncoder for DFINETransformer:
  in:  list×3 channels [256,512,1024] (or configured), strides [8,16,32]
  out: list×3 channels [hidden_dim]*3
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FrequencyDynamicGate(nn.Module):
    """Channel + depthwise spatial gate (lightweight frequency-style dynamic fusion)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.pw(self.dw(x)))
        return x * gate


class FDPNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = _ConvBNAct(channels * 2, channels, k=1)
        self.refine = nn.Sequential(
            _ConvBNAct(channels, channels, k=3),
            FrequencyDynamicGate(channels),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.refine(self.fuse(torch.cat([a, b], dim=1)))


class FDPN(nn.Module):
    """Dynamic pyramid neck used as DFINE ``model.encoder`` replacement."""

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024),
        hidden_dim: int = 256,
        feat_strides: Sequence[int] = (8, 16, 32),
        levels: Sequence[str] = ("p3", "p4", "p5"),
        eval_spatial_size: tuple[int, int] | list[int] | None = None,
        residual_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = [int(c) for c in in_channels]
        self.hidden_dim = int(hidden_dim)
        self.feat_strides = [int(s) for s in feat_strides]
        self.levels = [str(x) for x in levels]
        self.eval_spatial_size = (
            tuple(int(x) for x in eval_spatial_size) if eval_spatial_size else None
        )
        self.residual_scale = (
            float(residual_scale) if residual_scale is not None else None
        )
        if len(self.in_channels) != 3:
            raise ValueError(f"FDPN v1 expects 3 levels, got {len(self.in_channels)}")
        if len(self.feat_strides) != len(self.in_channels):
            raise ValueError("feat_strides length must match in_channels")
        if len(self.levels) != len(self.in_channels):
            raise ValueError("levels length must match in_channels")

        self.out_channels = [self.hidden_dim] * len(self.in_channels)
        self.out_strides = list(self.feat_strides)

        self.input_proj = nn.ModuleList(
            [_ConvBNAct(c, self.hidden_dim, k=1) for c in self.in_channels]
        )
        # Top-down (P5→P4, P4→P3)
        self.lateral = nn.ModuleList(
            [_ConvBNAct(self.hidden_dim, self.hidden_dim, k=1) for _ in range(2)]
        )
        self.fpn = nn.ModuleList([FDPNBlock(self.hidden_dim) for _ in range(2)])
        # Bottom-up (P3→P4, P4→P5)
        self.downsample = nn.ModuleList(
            [_ConvBNAct(self.hidden_dim, self.hidden_dim, k=3, s=2) for _ in range(2)]
        )
        self.pan = nn.ModuleList([FDPNBlock(self.hidden_dim) for _ in range(2)])
        self.out_gate = nn.ModuleList(
            [FrequencyDynamicGate(self.hidden_dim) for _ in range(3)]
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature maps, got {len(feats)}"
            )
        for i, (feat, c) in enumerate(zip(feats, self.in_channels)):
            if feat.shape[1] != c:
                raise ValueError(
                    f"level {i} channels {feat.shape[1]} != expected {c}"
                )

        p3, p4, p5 = [proj(f) for proj, f in zip(self.input_proj, feats)]
        projected = [p3, p4, p5]

        # Top-down
        p5_td = self.lateral[0](p5)
        p4_td = self.fpn[0](F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest"), p4)
        p4_td = self.lateral[1](p4_td)
        p3_td = self.fpn[1](F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest"), p3)

        # Bottom-up
        p3_out = p3_td
        p4_out = self.pan[0](self.downsample[0](p3_out), p4_td)
        p5_out = self.pan[1](self.downsample[1](p4_out), p5_td)

        outs = [self.out_gate[0](p3_out), self.out_gate[1](p4_out), self.out_gate[2](p5_out)]
        if self.residual_scale is not None:
            scale = float(self.residual_scale)
            outs = [proj + scale * out for proj, out in zip(projected, outs)]
        if getattr(self, "diag_enabled", False):
            self._last_inputs = [t.detach() for t in feats]
            self._last_outputs = [t.detach() for t in outs]
        return outs
