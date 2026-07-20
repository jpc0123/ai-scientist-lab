# RGB-T real detection baseline app (v0.8.1)

This package is the **fixed-entry** experiment app for real baselines.

## Current stand-in

- `baseline_key=dfine_s`
- Implementation: `torch_mini_standin_v0_8_1` (small Conv net)
- Produces the same Scientist Lab artifact schema as planned for DFINE-S

## Replace with DFINE-S

1. Vendor DFINE under `third_party/DFINE/` (pinned commit)
2. Implement native config/result adapters in `adapters/`
3. Keep `run_detection_experiment.py` CLI unchanged
4. Keep `model_summary.baseline_key=dfine_s` and update `baseline_implementation`

## Entry

```bat
python run_detection_experiment.py --config ... --output-dir ... --data-root ... --execution-mode smoke_train
```
