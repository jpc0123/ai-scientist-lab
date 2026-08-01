# Gate J — Seed Instability Diagnosis

## Status

- **J1 RUN IDENTITY AUDIT: PASS** (`identity_pass__proceed_j2_replicates`)
- Throughput probe draft deferred; instability diagnosis takes priority.

## J1 key findings

| Check | Result |
|--|--|
| Staged train/val COCO SHA | **identical** across seeds 42/43/44 |
| Train/val image name lists | **identical** (1830/500) |
| Resolved DFINE yaml (seed stripped) | **identical**; raw yaml differs **only** `seed:` |
| Checkpoint policy | **identical** (best-on-val mAP50_95) |
| Pair counts | 1830/500 all runs |
| Subset depends on train seed? | **No** (registration seed=42 frozen once) |
| Soft note | seed42 `protocol` label = `gate_h_curve_health` (reuse); training config still matched |

Artifacts: `outputs/experiments/v25_real_rgbt/gate_j_seed_diagnosis/`

## Allowed claim (now)

A2 三种子工程执行使用了**同一冻结子集成员**与**除 random_seed 外相同的 resolved 训练配置**；跨种子离散**不能**用子集重抽样解释。正式基线仍未冻结，待 J2/J3。

## J2 next

- `A2_seed44_rep1` then `A2_seed43_rep1` (do not overwrite Gate I exec dirs)
- Decision tree: seed effect vs nondeterminism vs one-off fault

## Still blocked

A4 / P01 / seed45-46 / drop-seed44 averaging / pick-best-seed baseline
