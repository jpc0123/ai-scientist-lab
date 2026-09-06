# Gate I Report — Formal A2 Multiseed

**Complete:** True  
**Engineering (all):** True  
**Formal candidates OK:** False  
**Decision:** diagnose_seed_instability

## Seeds
| seed | policy | exec | best ep | mAP50_95 | mAP50 | eng |
|------|--------|------|---------|----------|-------|-----|
| 42 | reuse_gate_h | exec_4088755cafc6 | 19 | 0.083347 | 0.189676 | True |
| 43 | fresh | exec_a6f6dff68971 | 3 | 0.108174 | 0.226189 | True |
| 44 | fresh | exec_373196093d79 | 15 | 0.041430 | 0.106617 | True |

## Aggregate
- mean mAP50_95 = **0.077650**
- stdev = 0.033735
- range = [0.041430, 0.108174]

## Interpretation
- Engineering three-seed matrix completed (CUDA, queries=300, non-zero preds).
- Seed44 (0.041) is a low outlier vs seed42 (0.083) / seed43 (0.108); spread gate failed.
- Decision diagnose_seed_instability: do **not** freeze formal baseline candidates yet; inspect seed44 curve/pairing before A4/P01.

## Claims
- May report exploratory three-seed execution completed.
- Must not claim stable A2 formal baseline, SOTA, or method ranking.
- A4/P01 remain blocked.
