# Vendored RT-DETR zoo (lyuwenyu/RT-DETR)

- Upstream: https://github.com/lyuwenyu/RT-DETR
- Path: `rtdetr_pytorch/src/zoo/rtdetr/`
- Retrieved: 2026-08-21 (main)
- Files: `rtdetr.py`, `rtdetr_decoder.py`, `rtdetr_criterion.py`, `rtdetr_postprocessor.py`
- License: see upstream; D-FINE vendor is a fork of the same stack

Scientist Lab glue (HOW / fusion wrap / APS_lowlight) stays in
`experiment_apps/rgbt_detection_real/`. These files are the **RT-DETR decoder
family**, not a fifth Agent and not D-FINE FDR/GO-LSD.

Patches vs upstream (D-FINE YAMLConfig registry + shared utils):

- `@register` → `@register()` (D-FINE `src.core.register` is a factory)
- decoder denoising/utils imports → `src.zoo.dfine.*` (already copied from RT-DETR)
- criterion `src.misc.dist` → `src.misc.dist_utils`; class name `RTDETRCriterion`
- criterion `forward(..., **kwargs)` so D-FINE `det_engine` can pass `epoch=`
- criterion unwraps D-FINE `HungarianMatcher` `{'indices': ...}` (upstream RT-DETR matcher returned the list)
- postprocessor `axis=-1` → `dim=-1` for torch.topk
