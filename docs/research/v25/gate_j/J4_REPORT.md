# Gate J4 — Curve Diagnosis (J3b seed43)

Updated: `2026-08-02T03:10:00+00:00` (approx; see `J4_J3B_CURVE_DIAGNOSIS.json`)

## Parent question

J3b failed the old gate because `|Δ|best_epoch=6` despite `|Δ|best mAP≤0.01` and matched `order_hash`.  
Is this **training-chain nondeterminism**, or **best-on-val checkpoint timing noise** on a small validation set?

## Runs

| run | exec | best ep | best mAP50_95 | last mAP50_95 |
|-----|------|---------|---------------|---------------|
| j3b_rep1 | `exec_cceebb5a17a4` | 2 | 0.059882 | 0.024336 |
| j3b_rep2 | `exec_e99fc5803811` | 8 | 0.050298 | 0.034189 |

- `|Δ|best mAP50_95` = **0.009584** (≤0.01)
- `|Δ|best_epoch` = **6**
- `order_hash` match: **True**

Artifacts: `J4_J3B_CURVE_DIAGNOSIS.json`, `J4_J3B_CURVE_COMPARISON.csv`

## Four diagnostics

### 1) Per-epoch curve correlation

| series | Pearson | Spearman | MAE |
|--------|---------|----------|-----|
| train_loss | **0.999** | ~1.0 | 0.108 |
| loss_vfl | high | high | small |
| loss_bbox | high | high | small |
| loss_giou | high | high | small |
| mAP50_95 | **-0.377** | low/neg | 0.0146 |
| mAP50 | similar pattern to mAP50_95 | — | — |
| lr | identical after warmup | — | — |

**Read:** optimization trajectories (losses) are effectively the same. Validation mAP shapes are **not** the same — so this is not a clean “identical curve, different peak only” story.

### 2) Window / aggregate mAP (not single peak)

| aggregate | rep1 | rep2 | \|Δ\| |
|-----------|------|------|------|
| best±2 centered at rep1 best (ep2) | ~0.035 | ~0.039 | ~0.004 |
| best±2 centered at rep2 best (ep8) | ~0.020 | ~0.040 | ~0.020 |
| last-5 epoch mean | ~0.0237 | ~0.0366 | **0.0130** |
| top-3 epoch mean | ~0.0437 | ~0.0476 | **0.0039** |

Top-3 means are very close; last-5 means differ by ~0.013 (still near the 0.01 band, slightly above).

### 3) Top-k epoch overlap

- rep1 top-5 epochs: **2, 1, 5, 20, 17**
- rep2 top-5 epochs: **8, 3, 5, 4, 7**
- overlap: **{5}** only (Jaccard low)

Near-best band (`best − mAP ≤ 0.01`) is short-lived on both runs — consistent with **spike peaks**, not a long flat plateau at ~0.05–0.06.

### 4) Fixed-epoch comparison

| epoch | rep1 mAP | rep2 mAP | \|Δ\| | \|Δ\|loss |
|------:|---------:|---------:|----:|---------:|
| 5 | 0.0261 | 0.0441 | 0.0180 | small |
| 10 | 0.0188 | 0.0367 | 0.0179 | small |
| 15 | 0.0237 | 0.0350 | 0.0113 | small |
| 20 | 0.0243 | 0.0342 | 0.0099 | small |

Fixed-epoch mAP gaps stay mostly **≤0.018**, while train losses stay tightly matched.

## Observed curve shapes (important correction)

Hypothetical “ep2=0.0599, ep3–7=0.052–0.058, ep8=0.059” **does not match data**.

Actual (rounded):

- **rep1:** early spike `0.045 → 0.060` then collapse to **~0.02** plateau
- **rep2:** mid-run elevated band **~0.03–0.05**, peak `0.050`@ep8, late **~0.034–0.038**

So: losses agree; val mAP level/shape still differs moderately after the early phase. Best-on-val is especially brittle because it locks onto short noisy spikes.

Missing logged series: `precision`, `recall`, `preds@0.1` (not in `training_history.csv`). Used `loss_vfl` as classification-loss proxy.

## Decision

**`caseC_metric_reproducible_checkpoint_unstable`** (with caveat)

Prefer this over labeling J3b as “training chain still clearly nondeterministic”:

- same seed, matched order hash
- best mAP within 0.01
- loss curves nearly identical
- fixed-epoch / top-3 aggregates much closer than best-epoch gap suggests

Caveat (do not overclaim Case C purity):

- mAP Pearson is negative; late levels still differ by ~0.01–0.018
- this can still mix **small-val checkpoint noise** with **mild residual nondeterminism in detection outputs**
- but it is **not** strong enough to justify immediate deep CUDA operator hunting

### Rejected next step

Deep CUDA/RNG hunt **now** — deferred unless a later protocol using fixed-epoch/last shows large gaps or early loss bifurcation.

## Checkpoint protocol recommendation (freeze before A4/P01)

1. **Primary:** report **last @ epoch 20**
2. **Auxiliary:** report best-on-val + **best−last gap**
3. Optional later: smoothed best (e.g. 3-eval moving mean) or larger val set — must be frozen pre-hoc

Do **not** keep `|Δ|best_epoch ≤ 2` as a hard unblocker for A4/P01.

## A4 / P01

Remain **blocked** until the revised checkpoint + reproducibility protocol is frozen.  
Unblock should require small same-seed noise under the **frozen rule** (prefer last/fixed/window), not bitwise-identical best epochs.

## Next

1. Freeze protocol text: primary=last20, aux=best, report gap; acceptance on last/fixed/window.
2. Optionally plot the CSV curves for the research log.
3. Only reopen CUDA/RNG if fixed-epoch or last-window gaps grow under that protocol.
