"""RT-DETR zoo modules for YAMLConfig registration. Not an Agent."""

from __future__ import annotations

__all__ = ["register_rtdetr_modules"]


def register_rtdetr_modules() -> None:
    """Import so @register() fills D-FINE GLOBAL_CONFIG. Requires DFINE on sys.path."""
    from vendor_rtdetr.rtdetr import RTDETR  # noqa: F401
    from vendor_rtdetr.rtdetr_criterion import RTDETRCriterion  # noqa: F401
    from vendor_rtdetr.rtdetr_decoder import RTDETRTransformer  # noqa: F401
    from vendor_rtdetr.rtdetr_postprocessor import RTDETRPostProcessor  # noqa: F401
