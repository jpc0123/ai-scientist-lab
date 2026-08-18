# Gate J3b — RNG Hunt seed43 Comparison

**Success:** `False`  
**Decision:** `caseB_still_nondeterministic__deeper_rng_or_caseA_protocol`

Deltas vs J3: `disable_ema` + `TF32 off` + `reseed_each_epoch`

| run | exec | best ep | mAP50_95 | order_hash |
|-----|------|---------|----------|------------|
| j3b_rep1 | `exec_cceebb5a17a4` | 2 | 0.059882 | `sha256:d807ca0781caf62c5a0e8ea95f8c4c2afeb68a7a86ba0d5eac1d0dff9d44057c` |
| j3b_rep2 | `exec_e99fc5803811` | 8 | 0.050298 | `sha256:d807ca0781caf62c5a0e8ea95f8c4c2afeb68a7a86ba0d5eac1d0dff9d44057c` |

- |Δ|mAP50_95 = 0.009584 (pass≤0.01: True)
- |Δ|best_epoch = 6 (pass≤2: False)
- order_hash match: True

**Next:** Still diverge after EMA/TF32/reseed — either deeper CUDA op hunt or accept Case A and revise multiseed protocol (still block A4/P01).
