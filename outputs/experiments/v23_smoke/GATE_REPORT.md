# v2.3 smoke gate report (updated 2026-07-28)

## S00 CU128 — PASS (execution-path smoke)

| Field | Value |
|------|------|
| S00 status | completed |
| S00 exit code | 0 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Compute capability | sm_120 |
| PyTorch | 2.7.1+cu128 |
| CUDA runtime | 12.8 |
| Image tag (immutable) | `scientist-rgbt-detection:v2.3-cu128-torch2.7.1-s00` |
| Image ID | `sha256:6b4dfd4c96d42f22d874d296545ce1205595b2fdc3abfbf585be574143142656` |

### Result
Scientist Lab → Worker → Docker → CUDA → Vendor D-FINE completed the prescribed Fast Eval smoke-training workflow.

### Scope
This proves **execution-path viability only**.
It does **not** prove accuracy, improvement, module effectiveness, statistical significance, or formal superiority.

Artifacts:
- `outputs/experiments/v23_smoke/S00_CU128_RUN.log`
- `outputs/experiments/v23_smoke/S00_IMAGE_ID.txt`
- `outputs/experiments/v23_smoke/S00_IMAGE_INSPECT.json`
- `runtime/scientist-worker/jobs/job_7be88c9b0956`

## Fixes locked for reuse (not 160-only hardcodes)

1. `eval_spatial_size` derived from experiment `input_size` / H×W
2. `num_queries` capped by coarse tokens **only when** Fast Eval `scale_queries_to_tokens=True`
3. COCO category remap to contiguous 0…N−1 with persisted bijective map JSON

## Image promotion plan

- Candidate: `scientist-rgbt-detection:v2-cuda-cu128`
- Immutable S00 pin: `scientist-rgbt-detection:v2.3-cu128-torch2.7.1-s00`
- Official alias after verify re-run: `scientist-rgbt-detection:v2-cuda`
- Do **not** move git tag `v2.3.0`
