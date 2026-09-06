"""Gate A: unit tests for FDPN neck (no CUDA required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
REAL_APP = ROOT / "experiment_apps" / "rgbt_detection_real"
sys.path.insert(0, str(REAL_APP))

from models.fdpn import FDPN, FrequencyDynamicGate  # noqa: E402
from models.neck_factory import NeckConfig, NeckFactory, parse_neck_config  # noqa: E402


def test_frequency_gate_shape():
    gate = FrequencyDynamicGate(16)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    y = gate(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None


def test_fdpn_output_shapes_and_channels():
    neck = FDPN(in_channels=(256, 512, 1024), hidden_dim=256, feat_strides=(8, 16, 32))
    feats = [
        torch.randn(2, 256, 20, 20, requires_grad=True),
        torch.randn(2, 512, 10, 10, requires_grad=True),
        torch.randn(2, 1024, 5, 5, requires_grad=True),
    ]
    outs = neck(feats)
    assert len(outs) == 3
    assert outs[0].shape == (2, 256, 20, 20)
    assert outs[1].shape == (2, 256, 10, 10)
    assert outs[2].shape == (2, 256, 5, 5)
    assert neck.out_channels == [256, 256, 256]
    assert neck.out_strides == [8, 16, 32]


def test_fdpn_gradients_flow():
    neck = FDPN(in_channels=(32, 64, 128), hidden_dim=40)
    feats = [
        torch.randn(1, 32, 16, 16, requires_grad=True),
        torch.randn(1, 64, 8, 8, requires_grad=True),
        torch.randn(1, 128, 4, 4, requires_grad=True),
    ]
    outs = neck(feats)
    loss = sum(o.mean() for o in outs)
    loss.backward()
    assert all(f.grad is not None for f in feats)
    assert any(p.grad is not None for p in neck.parameters() if p.requires_grad)


def test_fdpn_rejects_channel_mismatch():
    neck = FDPN(in_channels=(16, 32, 64), hidden_dim=16)
    feats = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 31, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    with pytest.raises(ValueError, match="channels"):
        neck(feats)


def test_neck_config_roundtrip_json():
    cfg = NeckConfig(
        type="fdpn",
        in_channels=(256, 512, 1024),
        out_channels=256,
        levels=("p3", "p4", "p5"),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
    )
    restored = NeckConfig.from_mapping(json.loads(json.dumps(cfg.to_dict())))
    assert restored == cfg
    built = NeckFactory.create(restored)
    assert isinstance(built, FDPN)


def test_parse_neck_config_defaults_and_aliases():
    assert parse_neck_config({}).type == "standard"
    assert parse_neck_config({"neck": {"type": "hybrid"}}).type == "standard"
    cfg = parse_neck_config({"neck": {"type": "fdpn", "out_channels": 128}})
    assert cfg.type == "fdpn"
    assert cfg.out_channels == 128
    assert cfg.hidden_dim == 128
    with pytest.raises(ValueError, match="Unknown neck"):
        parse_neck_config({"neck": {"type": "panet"}})


def test_neck_factory_standard_passthrough():
    class _Enc(nn.Module):
        def forward(self, x):  # pragma: no cover
            return x

    enc = _Enc()
    out = NeckFactory.create(NeckConfig(type="standard"), existing_encoder=enc)
    assert out is enc
    with pytest.raises(ValueError, match="existing HybridEncoder"):
        NeckFactory.create(NeckConfig(type="standard"))


def test_dfine_adapter_accepts_p00_neck_and_blocks_fusion_fdpn():
    from scientist_lab.tasks.rgbt_detection.baselines.dfine_s import DFineSBaselineAdapter

    adapter = DFineSBaselineAdapter()
    adapter.validate_parameters(
        {
            "input_mode": "rgbt",
            "fusion_method": "gated_multiscale",
            "fusion": {"type": "gated_multiscale"},
            "neck": {"type": "fdpn"},
            "epochs": 1,
        }
    )
    with pytest.raises(ValueError, match="not a fusion switch"):
        adapter.validate_parameters(
            {"input_mode": "rgbt", "fusion_method": "fdpn", "epochs": 1}
        )
    with pytest.raises(ValueError, match="unsupported neck"):
        adapter.validate_parameters(
            {
                "input_mode": "rgbt",
                "fusion_method": "early_concat",
                "neck": {"type": "unknown"},
                "epochs": 1,
            }
        )
