"""RT-DETR-S baseline key. Same fusion HOW surface as dfine_s; different detector."""

from scientist_lab.tasks.rgbt_detection.baselines.dfine_s import DFineSBaselineAdapter
from scientist_lab.tasks.rgbt_detection.baseline_adapter import register_baseline


@register_baseline
class RTDETRSBaselineAdapter(DFineSBaselineAdapter):
    baseline_key = "rtdetr_s"
