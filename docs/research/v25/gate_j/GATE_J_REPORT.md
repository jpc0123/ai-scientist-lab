# Gate J3 — Deterministic seed43 Comparison

**Success:** `False`  
**Decision:** `caseB_determinism_insufficient__continue_rng_hunt`

| run | exec | best ep | mAP50_95 | order_hash |
|-----|------|---------|----------|------------|
| det_rep1 | `exec_dd11f6d9c60c` | 4 | 0.042954 | `sha256:d807ca0781caf62c5a0e8ea95f8c4c2afeb68a7a86ba0d5eac1d0dff9d44057c` |
| det_rep2 | `exec_faf97ad6609d` | 11 | 0.079952 | `sha256:d807ca0781caf62c5a0e8ea95f8c4c2afeb68a7a86ba0d5eac1d0dff9d44057c` |

- |Δ|mAP50_95 = 0.036998 (pass≤0.01: False)
- |Δ|best_epoch = 7 (pass≤2: False)
- order_hash match: True

**Next:** Metrics still diverge under locks — hunt remaining nondeterministic ops/AMP/CUDA.
