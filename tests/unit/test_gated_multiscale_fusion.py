"""Gate A: unit tests for A3 gated multiscale fusion (no CUDA required)."""

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

from claim_gate import ClaimRejected, assert_claim_allowed, resolve_protocol  # noqa: E402
from fusion_names import normalize_fusion_method  # noqa: E402
from models.dual_stream_backbone import DualStreamGatedBackbone, build_dual_stream_backbone  # noqa: E402
from models.feature_fusion import GatedFeatureFusion, GatedMultiscaleFusion  # noqa: E402
from models.fusion_factory import FusionConfig, build_feature_fusion, parse_fusion_config  # noqa: E402


class _ToyBackbone(nn.Module):
    def __init__(self, channels=(32, 64, 128)) -> None:
        super().__init__()
        self.channels = list(channels)
        self.out_channels = list(channels)
        self.stems = nn.ModuleList(
            [nn.Conv2d(3, c, kernel_size=1) for c in channels]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for i, stem in enumerate(self.stems):
            h = max(1, x.shape[-2] // (2 ** (i + 1)))
            w = max(1, x.shape[-1] // (2 ** (i + 1)))
            feat = stem(x)
            outs.append(
                torch.nn.functional.interpolate(
                    feat, size=(h, w), mode="bilinear", align_corners=False
                )
            )
        return outs


def test_gated_feature_fusion_shape_and_channels():
    fusion = GatedFeatureFusion(16, residual=True)
    rgb = torch.randn(2, 16, 8, 8, requires_grad=True)
    thr = torch.randn(2, 16, 8, 8, requires_grad=True)
    out = fusion(rgb, thr)
    assert out.shape == rgb.shape


def test_gated_feature_fusion_rejects_shape_mismatch():
    fusion = GatedFeatureFusion(8)
    rgb = torch.randn(1, 8, 4, 4)
    thr = torch.randn(1, 8, 5, 5)
    with pytest.raises(ValueError, match="shapes must match"):
        fusion(rgb, thr)


def test_gated_multiscale_level_count_and_channels():
    fusion = GatedMultiscaleFusion([16, 32, 64], residual=True, levels=["p3", "p4", "p5"])
    rgb = [
        torch.randn(1, 16, 16, 16, requires_grad=True),
        torch.randn(1, 32, 8, 8, requires_grad=True),
        torch.randn(1, 64, 4, 4, requires_grad=True),
    ]
    thr = [t.detach().clone().requires_grad_(True) for t in rgb]
    # Use independent tensors so grads are distinct.
    thr = [
        torch.randn_like(rgb[0], requires_grad=True),
        torch.randn_like(rgb[1], requires_grad=True),
        torch.randn_like(rgb[2], requires_grad=True),
    ]
    fused = fusion(rgb, thr)
    assert len(fused) == 3
    for f, r in zip(fused, rgb):
        assert f.shape == r.shape
        assert f.shape[1] == r.shape[1]


def test_gated_multiscale_gradients_reach_both_streams():
    fusion = GatedMultiscaleFusion([8, 16], residual=True)
    rgb = [
        torch.randn(2, 8, 8, 8, requires_grad=True),
        torch.randn(2, 16, 4, 4, requires_grad=True),
    ]
    thr = [
        torch.randn(2, 8, 8, 8, requires_grad=True),
        torch.randn(2, 16, 4, 4, requires_grad=True),
    ]
    fused = fusion(rgb, thr)
    loss = sum(feature.mean() for feature in fused)
    loss.backward()
    assert rgb[0].grad is not None and thr[0].grad is not None
    assert rgb[1].grad is not None and thr[1].grad is not None


def test_dual_stream_backbone_6ch_and_grads():
    rgb_bb = _ToyBackbone((16, 32, 64))
    cfg = FusionConfig(channels=(16, 32, 64), share_backbone=False, residual=True)
    dual = build_dual_stream_backbone(rgb_bb, cfg)
    assert isinstance(dual, DualStreamGatedBackbone)
    x = torch.randn(2, 6, 64, 64, requires_grad=True)
    outs = dual(x)
    assert len(outs) == 3
    loss = sum(o.mean() for o in outs)
    loss.backward()
    assert x.grad is not None
    # Both backbones participate when not shared.
    rgb_grad = any(p.grad is not None for p in dual.rgb_backbone.parameters() if p.requires_grad)
    thr_grad = any(
        p.grad is not None for p in dual.thermal_backbone.parameters() if p.requires_grad
    )
    fusion_grad = any(p.grad is not None for p in dual.fusion.parameters() if p.requires_grad)
    assert rgb_grad and thr_grad and fusion_grad


def test_fusion_config_roundtrip_json():
    cfg = FusionConfig(
        type="gated_multiscale",
        levels=("p3", "p4", "p5"),
        share_backbone=False,
        residual=True,
        channels=(256, 512, 1024),
    )
    payload = cfg.to_dict()
    text = json.dumps(payload)
    restored = FusionConfig.from_mapping(json.loads(text))
    assert restored == cfg
    built = build_feature_fusion(restored)
    assert isinstance(built, GatedMultiscaleFusion)


def test_parse_fusion_config_from_parameters():
    cfg = parse_fusion_config(
        {
            "fusion_method": "full_fusion",
            "fusion": {
                "type": "gated_multiscale",
                "levels": ["p3", "p4", "p5"],
                "share_backbone": False,
                "residual": True,
            },
        }
    )
    assert cfg is not None
    assert cfg.type == "gated_multiscale"
    assert normalize_fusion_method("full_fusion") == "gated_multiscale"
    assert parse_fusion_config({"fusion_method": "early_concat"}) is None


def test_claim_gate_rejects_smoke_performance_claims():
    protocol = resolve_protocol(
        contract={"execution_mode": "fast_eval"},
        parameters={"protocol": "smoke"},
    )
    assert protocol == "smoke"
    with pytest.raises(ClaimRejected, match="Smoke"):
        assert_claim_allowed("performance_superiority", protocol=protocol)


def test_dfine_adapter_accepts_gated_multiscale():
    from scientist_lab.tasks.rgbt_detection.baselines.dfine_s import DFineSBaselineAdapter

    adapter = DFineSBaselineAdapter()
    adapter.validate_parameters(
        {
            "input_mode": "rgbt",
            "fusion_method": "gated_multiscale",
            "fusion": {"type": "gated_multiscale", "residual": True},
            "epochs": 1,
        }
    )
    with pytest.raises(ValueError, match="not a fusion switch"):
        adapter.validate_parameters(
            {"input_mode": "rgbt", "fusion_method": "fdpn", "epochs": 1}
        )


def test_stage_gated_paired_folders(tmp_path: Path):
    from dfine_dataset_stage import stage_coco_for_dfine

    data_root = ROOT / "datasets" / "rgbt_fast_eval_v1"
    if not data_root.exists():
        pytest.skip("rgbt_fast_eval_v1 missing")
    stage = stage_coco_for_dfine(
        data_root,
        tmp_path / "stage",
        input_mode="rgbt",
        fusion_method="gated_multiscale",
    )
    assert stage["staging_mode"] == "gated_paired"
    assert Path(stage["thermal_train_img"]).is_dir()
    assert Path(stage["thermal_val_img"]).is_dir()
    assert any(Path(stage["train_img"]).glob("*"))
    assert any(Path(stage["thermal_train_img"]).glob("*"))
