# Gate J2 — Replicate Comparison

**Decision:** inconclusive_need_j3_j4

| run | exec | best ep | mAP50_95 | mAP50 |
|-----|------|---------|----------|-------|
| seed43_original | exec_a6f6dff68971 | 3 | 0.108174 | 0.226189 |
| seed43_rep1 | exec_f542d7eea047 | 4 | 0.073337 | 0.136152 |
| seed44_original | exec_373196093d79 | 15 | 0.041430 | 0.106617 |
| seed44_rep1 | exec_755b00bebeaf | 5 | 0.040383 | 0.124422 |

- seed43 classify: moderate_replicate_gap (orig 0.108 → rep 0.073, |Δ|≈0.035)
- seed44 classify: reproducible_near_original (orig 0.041 → rep 0.040, |Δ|≈0.0010)

**Next:** Run curve/detection diagnosis before protocol change.

Still blocked: A4 / P01 / pick-best-seed / drop-seed44 averaging.
