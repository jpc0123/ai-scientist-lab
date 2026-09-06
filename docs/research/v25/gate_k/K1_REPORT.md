# Gate K1 — A2 Seed Sensitivity Curve Audit (no retrain)

**Status:** completed  
**Decision:** `proceed_k2_single_factor_sensitivity`  
**Reporting rule:** `a2_baseline_checkpoint_v2` (primary last@20)  
**Blocked:** A4 / P01

## One-line status

Checkpoint reporting is isolated; A2 core instability remains — same protocol yields three trajectories: healthy (42), peak-then-collapse (43), low (44).

## last@20 reminder

| Seed | last@20 | best | best−last | trajectory |
|-----:|--------:|-----:|----------:|------------|
| 42 | 0.0822 | 0.0833 | 0.0012 | healthy_convergence |
| 43 | 0.0380 | 0.1082 | 0.0702 | peak_then_collapse (late cliff) |
| 44 | 0.0222 | 0.0414 | 0.0192 | low / weak then fade |

## Available vs missing signals

**Available now:** train_loss, loss_vfl, loss_bbox, loss_giou, lr, mAP50_95, mAP50, AP75, APS, COCO AR* from `log.txt`, endpoint `preds@0.1`.

**Not logged per epoch (cannot answer fully in K1):** precision/recall curves, per-epoch preds@0.1, matched positive counts, gradient norms, per-class AP, per-sequence AP.  
`metrics.json` precision/recall are currently 0.0 placeholders.

## seed43 — why late collapse?

Full mAP50_95 (rounded):

`0.028, 0.035, **0.108**, 0.090, 0.081, 0.089, 0.096, 0.093, 0.094, 0.090, 0.093, 0.097, 0.102, 0.102, 0.100, **0.103**, 0.082, 0.059, 0.044, **0.038**`

关键 facts:

- Peak at ep3, but performance stays high (~0.09–0.10) through **ep16**.
- Collapse is a **late cliff (ep17–20)**, not an immediate post-peak crash.
- Train losses continue improving through the end; no bbox/GIoU late rebound in the logged means.
- Endpoint `preds@0.1` = **17125** > seed42 **12698** → not under-firing; more consistent with confidence/quantity drift + val instability / overfit.

**Working hypothesis:** late over-optimization / schedule too aggressive for this seed on small data — favors **K2-A (LR×0.5)** and later **K2-C (more independent sequences)**. Extending to 40ep without LR/data change is likely harmful.

## seed44 — why low?

Full mAP50_95 (rounded):

`0.013, 0.033, 0.027, 0.020, 0.024, … climbs to ~0.041 @ep10–15, then fades to **0.022**`

关键 facts:

- Ep1 already far below seed42 (0.013 vs 0.044).
- Never reaches a healthy band; best only 0.041@ep15.
- Train-loss curve still highly similar to seed42 (global optimization “looks fine”).
- Endpoint `preds@0.1` = **3769** ≪ 42/43 → under-activated detections.

**Working hypothesis:** weak early activation / matching / warmup sensitivity and/or insufficient diverse data — favors **K2-B (warmup×2)** and **K2-C**, with **K2-A** still useful as low-cost first probe.

## seed42 control

Healthy: best≈last, late mAP still rising into ep19–20. Proves current protocol *can* succeed; do **not** spend first K2 budget re-running 42.

## Cross-seed takeaway

| observation | implication |
|-------------|-------------|
| train losses similar across seeds | not random compute failure |
| val mAP trajectories diverge sharply | seed × data/optimization interaction |
| 43 late cliff + high preds@0.1 | overfit / late instability |
| 44 early lag + low preds@0.1 | under-activation / warmup/data |
| last@20 range still ~0.060 | core A2 instability unresolved |

## K2 matrix (ready; not auto-launched)

| Exp | Change | Seeds |
|-----|--------|------:|
| K2-A | LR × 0.5 (base 2e-4 → 1e-4; backbone 1e-4 → 5e-5) | 43, 44 |
| K2-B | warmup × 2 | 43, 44 |
| K2-C | more independent train/val sequences | 43, 44 |

Order: **A → B → C**. One factor at a time. Primary metric last@20; watch collapse gap and min(seed43,seed44).

## K3 epoch budget

Deferred. seed43 is late-degenerating (longer training may worsen); seed44 weak growth then fade; only revisit 40ep if a stabilized protocol shows unfinished growth — with scheduler retuned.

## A4 / P01

Remain frozen. Unstable baseline ⇒ method deltas uninterpretable.

## Next

Proceed to **K2-A** on seeds 43/44 under checkpoint v2 reporting, or wait for explicit launch approval.
