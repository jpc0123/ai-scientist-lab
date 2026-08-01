# Gate J4 — Curve Diagnosis (partial, pre-J3b)

Updated: `2026-08-01T09:07:39.408765+00:00`

## Best-of-run snapshot

| run | exec | best ep | best mAP50_95 | last mAP50_95 | peak−last |
|-----|------|---------|---------------|---------------|-----------|
| seed42_original | `exec_4088755cafc6` | 19 | 0.083347 | 0.082180 | 0.0012 |
| seed43_original | `exec_a6f6dff68971` | 3 | 0.108174 | 0.037971 | 0.0702 |
| seed43_rep1 | `exec_f542d7eea047` | 4 | 0.073337 | 0.013970 | 0.0594 |
| seed44_original | `exec_373196093d79` | 15 | 0.041430 | 0.022243 | 0.0192 |
| seed43_j3_det_rep1 | `exec_dd11f6d9c60c` | 4 | 0.042954 | 0.038845 | 0.0041 |
| seed43_j3_det_rep2 | `exec_faf97ad6609d` | 11 | 0.079952 | 0.046199 | 0.0338 |

## Observations

- Gate I multiseed: seed43_original peaks early/high then collapses; seed44 stays low — level gap is curve-shape, not only best-epoch noise.
- J3 det_rep1 best=0.04295406358691742@ep4 vs det_rep2 best=0.07995186793450823@ep11 (|Δ|mAP≈0.037) with matching order_hash — residual CUDA/EMA nondeterminism (J3b target).
- seed43_original best≈0.10817447410080037 vs seed44_original best≈0.04143016439577045 — if J3b still cannot lock seed43, prefer Case A protocol revision over claiming seed-mean.
- J3b rows will be appended after rng-hunt completes; A4/P01 remain blocked.

## Next

1. Wait for J3b dual-rep completion.
2. Append `seed43_j3b_rep*` curves.
3. Only then decide Case A (protocol) vs Case C (deterministic formal rebaseline).
4. Keep A4/P01 blocked.
