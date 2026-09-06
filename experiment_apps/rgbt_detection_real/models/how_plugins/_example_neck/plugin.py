"""Example HOW neck plugin. Not registered. Not a Claim.

Replaces DFINE HybridEncoder at the encoder slot. I/O matches FDPN/HybridEncoder:
  in:  list×3 feature maps with in_channels
  out: list×3 feature maps with hidden_dim channels
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PLUGIN_KIND = "neck"


class LateralAddNeck(nn.Module):
    """Top-down lateral add. Same encoder slot as N1; not a new detector family."""

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024),
        hidden_dim: int = 256,
        feat_strides: Sequence[int] = (8, 16, 32),
        **_: object,
    ) -> None:
        super().__init__()
        chans = [int(c) for c in in_channels]
        if len(chans) < 2:
            raise ValueError("neck plugin needs at least two pyramid levels")
        self.in_channels = chans
        self.hidden_dim = int(hidden_dim)
        self.feat_strides = [int(s) for s in feat_strides][: len(chans)]
        while len(self.feat_strides) < len(chans):
            self.feat_strides.append(self.feat_strides[-1] * 2)
        self.out_channels = [self.hidden_dim] * len(chans)
        self.out_strides = list(self.feat_strides)
        self.proj = nn.ModuleList(
            nn.Conv2d(c, self.hidden_dim, kernel_size=1) for c in chans
        )
        self.smooth = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
            for _ in chans
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if not isinstance(feats, (list, tuple)) or len(feats) != len(self.in_channels):
            raise ValueError(
                f"neck expected {len(self.in_channels)} feature maps, got {type(feats)!r}"
            )
        projected = []
        for idx, (proj, feat, ch) in enumerate(
            zip(self.proj, feats, self.in_channels)
        ):
            if int(feat.shape[1]) != int(ch):
                raise ValueError(
                    f"neck level {idx} channels {int(feat.shape[1])} != {ch}"
                )
            projected.append(proj(feat))
        outs: list[torch.Tensor] = [projected[-1]]
        for idx in range(len(projected) - 2, -1, -1):
            up = F.interpolate(
                outs[0], size=projected[idx].shape[-2:], mode="nearest"
            )
            outs.insert(0, projected[idx] + up)
        return [smooth(item) for smooth, item in zip(self.smooth, outs)]


def build_neck(
    in_channels: Sequence[int] = (256, 512, 1024),
    hidden_dim: int = 256,
    feat_strides: Sequence[int] = (8, 16, 32),
    **kwargs: object,
) -> nn.Module:
    return LateralAddNeck(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        feat_strides=feat_strides,
        **kwargs,
    )
