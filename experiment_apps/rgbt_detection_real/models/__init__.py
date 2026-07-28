"""RGB-T fusion + neck modules for D-FINE Vendor path (config-driven)."""

from fusion_names import normalize_fusion_method

from .dual_stream_backbone import DualStreamGatedBackbone
from .fdpn import FDPN, FrequencyDynamicGate
from .feature_fusion import FeatureFusion, GatedFeatureFusion, GatedMultiscaleFusion
from .fusion_factory import FusionConfig, build_feature_fusion, parse_fusion_config
from .neck_factory import NeckConfig, NeckFactory, parse_neck_config

__all__ = [
    "DualStreamGatedBackbone",
    "FDPN",
    "FeatureFusion",
    "FrequencyDynamicGate",
    "FusionConfig",
    "GatedFeatureFusion",
    "GatedMultiscaleFusion",
    "NeckConfig",
    "NeckFactory",
    "build_feature_fusion",
    "normalize_fusion_method",
    "parse_fusion_config",
    "parse_neck_config",
]
