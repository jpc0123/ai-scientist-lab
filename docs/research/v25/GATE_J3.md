# Gate J3 — Deterministic Diagnosis (active)

## Changes vs Gate I training
- `num_workers=0`
- `configure_determinism(seed)` + CUBLAS workspace
- DataLoader `Generator` + `worker_init_fn`
- BatchImageCollate **multiscale disabled**
- AMP off

## Success criteria
- |Δ|best mAP50_95 ≤ 0.01
- |Δ|best epoch ≤ 2
- epoch0 `order_hash` identical

## Queue
1. seed43 det_rep1 (running/queued)
2. seed43 det_rep2 (auto)
3. J3_RUN_COMPARISON + continue J4

A4/P01 still blocked.
