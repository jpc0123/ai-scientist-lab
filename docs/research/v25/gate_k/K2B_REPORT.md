# Gate K2-B — warmup×2 @ original LR Comparison

**Decision:** `only_44_improve__43_not_cliff__scheduler_or_lr_schedule_review`  
**Completed:** `2026-08-03T07:12:50.098711+00:00`  
**Factor:** `warmup_duration 500 → 1000` · **LR fixed at** `0.0002` (not stacked with K2-A)

| Seed | 原 best | 新 best | 原 last | 新 last | 原 gap | 新 gap |
|-----:|--------:|--------:|--------:|--------:|-------:|-------:|
| 43 | 0.1082 | 0.0428 | 0.0380 | 0.0394 | 0.0702 | 0.0034 |
| 44 | 0.0414 | 0.0896 | 0.0222 | 0.0330 | 0.0192 | 0.0566 |

- seed43 improved: **False** (phenotype=`stable_but_capped`)
- seed44 improved: **True** (phenotype=`other`)
- seed43 late_cliff: **False** · mid_max=0.0423 · best=0.0428
- seed44 ep1-5 mean: **0.0592** · preds@0.1: **1401**

**Next:** 44 improved; 43 no longer cliffs but did not retain mid peak → schedule/LR shape review.

A4/P01 remain blocked.
