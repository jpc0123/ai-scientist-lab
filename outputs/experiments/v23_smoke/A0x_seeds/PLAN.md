# Ablation + multi-seed plan status (2026-07-28)

## Mapping to original plan

| Plan ID | Meaning | Status |
|---------|---------|--------|
| A0 / B00 | RGB-only | runnable |
| A1 / B01 | Thermal-only | runnable |
| A2 / B02 | early_concat | runnable |
| A3 | 完整融合 beyond early_concat | **blocked** |
| P00 | 完整方法 + FDPN | **blocked** |
| B02/P00 × 3 seeds | protocol seeds 42,43,44 | **running / recording** for A0–A2 only |

## Claim scope

Multi-seed Fast Eval on tiny budget remains **exploratory**. Do not claim modality superiority or FDPN effectiveness from mAP≈0.
