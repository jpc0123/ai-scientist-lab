# B0x matched Fast Eval baselines (exploratory)

## Freeze

See `FREEZE.md`. Protocol `protocol_rgbt_cuda_001` registered; project `project_rgbt_cuda_001` ready.

Image: `scientist-rgbt-detection:v2-cuda`  
Image ID: `sha256:6b4dfd4c96d42f22d874d296545ce1205595b2fdc3abfbf585be574143142656`  
Code SHA at freeze start: `284fb6d`

## Results (seed=42, 1 epoch, 160×160, batch=2)

| ID | Mode | Fusion | Status | Exit | Exec | Duration (s) | Peak GPU (MiB) | mAP50 | mAP50_95 |
|----|------|--------|--------|------|------|--------------|----------------|-------|----------|
| B00 | rgb | none | completed | 0 | exec_c1511f6f0522 | 20.39 | 361.8 | 0.0 | 0.0 |
| B01 | thermal | none | completed | 0 | exec_133c0aa127ad | 21.28 | 361.8 | 0.0 | 0.0 |
| B02 | rgbt | early_concat | completed | 0 | exec_4ac0eb472f33 | 20.66 | 361.8 | 0.0 | 0.0 |

Logs:
- `B00_RGB_ONLY.log`
- `B01_THERMAL_ONLY.log`
- `B02_RGBT_EARLY_FUSION.log`

## Allowed claim

Matched RGB / Thermal / early-fusion Vendor D-FINE Fast Eval **execution paths** all completed on official `v2-cuda` under the same short-budget protocol.

## Forbidden claims

- Accuracy ranking among B00/B01/B02
- Early fusion is better / worse
- FDPN / full method effectiveness
- Formal superiority or statistical significance

mAP=0 on all three is expected for 1-epoch tiny Fast Eval smoke and must not be interpreted as a modality comparison.

## Next

P00 (full method) and A00–A03 only after deciding whether Fast Eval budget is sufficient for exploratory comparison, or after freezing a longer training protocol. Prefer still **not** claiming superiority until multi-seed formal budget exists.
