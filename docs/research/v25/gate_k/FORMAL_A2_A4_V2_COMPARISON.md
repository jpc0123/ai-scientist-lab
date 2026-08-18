# Formal A2 vs A4 v2 — Comparison

**Protocol:** `protocol_rgbt_formal_a2_a4_v2`  
**Primary:** last@20 mAP50_95  
**Decision:** `strong_evidence__consistent_gain`

## Per-seed

| seed | A2 last | A4 last | Δ last | A2 best | A4 best | A2 exec | A4 exec |
|------|---------|---------|--------|---------|---------|---------|---------|
| 44 | 0.0728 | 0.1217 | +0.0489 | 0.1432 | 0.1550 | `exec_4f4151310027` | `exec_8bffa02fe333` |
| 43 | 0.1093 | 0.1381 | +0.0288 | 0.1531 | 0.1457 | `exec_d4b4feb6cfb1` | `exec_1f526b78bc5f` |
| 42 | 0.0529 | 0.1483 | +0.0954 | 0.1413 | 0.1631 | `exec_00cc3754ea3d` | `exec_e5e0067e8d1b` |

## Summary

| metric | value |
|--------|-------|
| A2 mean last | 0.0783 |
| A4 mean last | 0.1361 |
| mean Δ | +0.0577 |
| A4 last range | 0.0266 |
| positive seeds | 3/3 |

## Claim policy

- Strong: 3/3 Δ>0 and mean Δ>0 → consistent gain
- Moderate: 2/3 positive → exploratory improvement trend
- Negative/null: no consistent FDPN benefit (valid conclusion)

**Accepted 2026-08-07** as formal candidate method (`DECISION_ACCEPT_A4_LAUNCH_P01.json`).  
P01 residual_scale unlocked under `formal_a4_p01_v1`. Further A2 tuning / SOTA / significance claims remain blocked.
