# Gate K2-A — LR×0.5 Comparison

**Decision:** `only_44_improve__go_k2b_or_scheduler`  
**Completed:** `2026-08-02T12:13:31.726293+00:00`

| Seed | 原 best | 新 best | 原 last | 新 last | 原 gap | 新 gap |
|-----:|--------:|--------:|--------:|--------:|-------:|-------:|
| 43 | 0.1082 | 0.0395 | 0.0380 | 0.0395 | 0.0702 | 0.0000 |
| 44 | 0.0414 | 0.0816 | 0.0222 | 0.0602 | 0.0192 | 0.0214 |

- seed43 improved: **False**
- seed44 improved: **True**
- seed43 late_cliff: **False** (pre_last_max=0.0389)
- seed44 preds@0.1: **3643**

**Next:** Lower LR helps early start; seed43 collapse may be overfit/schedule/data → K2-B or scheduler review.

A4/P01 remain blocked.
