# A2 Baseline Checkpoint Protocol v2 — Freeze + Stability Reaggregation

**Protocol:** `a2_baseline_checkpoint_v2`  
**Decision:** `protocol_v2_frozen__await_seed_sensitivity_review`  
**Updated:** `2026-08-02T03:17:45.169548+00:00`

## One-line conclusion

J3b/J4 show A2 train-loss trajectories are highly reproducible and overall performance levels are close, but per-epoch val mAP still has residual variance on the small validation set, making single-point best-on-val unstable; freeze last@20 as primary reporting, reaggregate A2 multiseed stability, keep A4/P01 blocked.

## Frozen reporting rule

- Primary: **last@20** `mAP50_95`
- Secondary: best-on-val (+ earliest-epoch tie-break)
- Diagnostics: best−last gap, top-3 mean, last-5 mean, best epoch
- `test_set_used_for_selection`: false

## Reuse audit

All Gate I seeds 42/43/44 already have `checkpoint/last.pt` and epoch-20 rows in `training_history.csv`. **No unconditional retrain** for this reaggregation.

## Stability table

| Seed | last@20 mAP50-95 | best mAP50-95 | best−last | best ep | top3 mean | last5 mean |
|-----:|-----------------:|--------------:|----------:|--------:|----------:|-----------:|
| 42 | 0.082180 | 0.083347 | 0.001167 | 19 | 0.079827 | 0.070291 |
| 43 | 0.037971 | 0.108174 | 0.070203 | 3 | 0.104483 | 0.065215 |
| 44 | 0.022243 | 0.041430 | 0.019188 | 15 | 0.041324 | 0.027096 |

## Dispersion

- last@20: mean=0.047465, stdev=0.031076, range=0.059938
- best-on-val: mean=0.077650, stdev=0.033735, range=0.066744
- Q1 last@20 less dispersed than best? **True**
- Q2 seed44 still low at fixed endpoint? **True**

## Interpretation

- last@20 cross-seed dispersion is only **slightly** smaller than best-on-val (range 0.060 vs 0.067). Protocol v2 removes peak-selection allergy, but does **not** by itself make A2 multiseed stable.
- Under last@20, seed43 collapses from best 0.108 → **0.038** (best−last≈0.070). That early spike was never a durable endpoint capability.
- seed44 remains low at the fixed endpoint (**0.022**). Main remaining issue looks like true A2 seed/protocol sensitivity, not only checkpoint choice.
- Therefore: freeze reporting on last@20, but **do not** freeze formal A2 baseline candidates or unblock A4/P01 yet.

## A4 / P01

Remain **blocked**. Unblock only after v2 same-seed repro is acceptable, last@20 rule frozen (done), three-seed last@20 has no engineering anomaly, and run noise << expected method delta (~0.035).

**Next:** Reporting protocol v2 is frozen. A2 three-seed last@20 still shows meaningful spread and/or seed44 low endpoint; review before any A4/P01 unblock. No unconditional retrain required for this reaggregation.
