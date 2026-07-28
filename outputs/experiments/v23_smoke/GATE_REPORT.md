# v2.3 smoke gate report (updated 2026-07-28)

## S00 — PASS (execution-path smoke)

| Field | Value |
|------|------|
| S00 status | completed |
| S00 exit code | 0 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Compute capability | sm_120 |
| PyTorch | 2.7.1+cu128 |
| CUDA runtime | 12.8 |
| Immutable image tag | `scientist-rgbt-detection:v2.3-cu128-torch2.7.1-s00` |
| Image ID | `sha256:6b4dfd4c96d42f22d874d296545ce1205595b2fdc3abfbf585be574143142656` |
| Official alias | `scientist-rgbt-detection:v2-cuda` (same Image ID) |
| Official re-run | `S00_OFFICIAL_V2_CUDA_RUN.log` → status=completed, exit=0, image=`v2-cuda` |

### Result
Scientist Lab → Worker → Docker → CUDA → Vendor D-FINE completed the prescribed Fast Eval smoke-training workflow.

### Scope
Proves **execution-path viability only**. Does **not** prove accuracy, improvement, module effectiveness, statistical significance, or formal superiority.

### Git
- Fix commit: `34e9736` on `feat/v2.2-real-provider-restricted-diff`
- `v2.3.0` remains frozen at `0befdd7` (not moved)

## S01 — PASS (RGB-T data-chain smoke)

| Field | Value |
|------|------|
| Experiment ID | `S01_RGBT_DATA_SMOKE` |
| Contract | `examples/rgbt_s01_rgbt_data_smoke_contract.json` |
| Status | completed |
| Exit | 0 |
| Image | `scientist-rgbt-detection:v2-cuda` |
| Mode | `input_mode=rgbt`, `fusion_method=early_concat`, AMP on, seed=0 |
| Pairing | train 24 / val 12 paired 1:1 |

## B0x — PASS (matched modality baselines, exploratory)

Protocol `protocol_rgbt_cuda_001` registered; project `project_rgbt_cuda_001` ready.
Details: `outputs/experiments/v23_smoke/B0x/SUMMARY.md`

| ID | Mode | Fusion | Status | Exit | Exec |
|----|------|--------|--------|------|------|
| B00 | rgb | none | completed | 0 | exec_c1511f6f0522 |
| B01 | thermal | none | completed | 0 | exec_133c0aa127ad |
| B02 | rgbt | early_concat | completed | 0 | exec_4ac0eb472f33 |

All three: mAP50=mAP50_95=0.0 (1-epoch tiny Fast Eval; **not** a ranking).

### Allowed
Matched RGB / Thermal / early-fusion Vendor D-FINE Fast Eval **paths** completed on official `v2-cuda`.

### Forbidden
Accuracy ranking, fusion superiority, FDPN effectiveness, formal claims.

## Fixes locked (not 160-only hardcodes)

1. `eval_spatial_size` from experiment input H×W
2. `num_queries` capped by coarse tokens **only** when Fast Eval `scale_queries_to_tokens=True`
3. COCO remap → contiguous 0…N−1 with persisted bijective map

## Next

P00 / ablations / multi-seed only after a longer training budget freeze if accuracy claims are desired.
