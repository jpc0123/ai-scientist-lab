# Gate J3b — RNG Hunt seed43 Comparison (reinterpreted after J4)

**Raw gate success (old criteria):** `False`  
**Old auto decision:** `caseB_still_nondeterministic__deeper_rng_or_caseA_protocol`  
**Revised interpretation (after J4):** `caseC_metric_reproducible_checkpoint_unstable`

Deltas vs J3: `disable_ema` + `TF32 off` + `reseed_each_epoch`

| run | exec | best ep | mAP50_95 | order_hash |
|-----|------|---------|----------|------------|
| j3b_rep1 | `exec_cceebb5a17a4` | 2 | 0.059882 | `sha256:d807ca0781caf62c5a0e8ea95f8c4c2afeb68a7a86ba0d5eac1d0dff9d44057c` |
| j3b_rep2 | `exec_e99fc5803811` | 8 | 0.050298 | `sha256:d807ca0781caf62c5a0e8ea95f8c4c2afeb68a7a86ba0d5eac1d0dff9d44057c` |

- `|Δ|mAP50_95` = 0.009584 (pass≤0.01: True)
- `|Δ|best_epoch` = 6 (pass≤2: False)
- order_hash match: True

## Correct reading

Do **not** treat this as “training chain still clearly nondeterministic.”

More accurate:

- same-seed **best/final-ish performance** is largely reproducible within 0.01
- **best-on-val peak timing** is unstable
- sample order is stable

J4 shows train losses nearly identical (Pearson≈0.999). Validation mAP shapes still differ (Pearson negative), with early spike vs mid-run elevated band — so Case C is about **checkpoint rule fragility + mild val-metric residual**, not a license to claim bitwise-identical trajectories.

**Next:** freeze checkpoint/repro protocol (primary last@20); keep A4/P01 blocked; defer deep CUDA hunt. See `J4_REPORT.md`.
