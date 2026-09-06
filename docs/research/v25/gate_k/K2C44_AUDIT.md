# Gate K2-C44 Audit (zero-cost)

**Decision:** `audit_pass__launch_seed43`  
**Exec:** `exec_4f4151310027`  
**Engineering anomalies:** none

## 1. Checkpoint re-eval (frozen K2-C val stage)

| ckpt | train log | re-eval COCO print | pass |
|------|-----------|--------------------|------|
| best_weights.pth (ep2, last_epoch=1) | 0.1432 | **0.143** | ✓ |
| last.pt (ep20, last_epoch=19) | 0.0728 | **0.073** | ✓ |

- Val images/anns: **650 / 4526** (registered = staged = prediction_audit batches)
- Val annotation/manifest hashes match `K2C_DATASET_FREEZE`
- Val transforms: Resize + ConvertPILImage only (no train aug)
- Checkpoint↔epoch alignment OK

## 2. Curve morphology → **B**

ep2 peaks at 0.143, but ep1–8 stay in a high band (~0.11–0.14), then soft mid/late decline to last≈0.073. Not a single-point spike-only (A), not irregular oscillation (C). Consistent with early overfitting / mid generalization drop under constant peak LR.

## 3. Comparison caveat

K2-C **val expanded** 10→13 sequences (+ `DJI_0115_2`, `DJI_0229_2`, `DJI_0309_1`).  
Do **not** claim Gate-H 0.022 → K2-C 0.073 as same-eval-set +0.051.  
Allowed wording: under the expanded K2-C protocol, seed44 fixed endpoint reaches 0.073, above the stability floor.

## 4. Routing

gap=0.070 remains a **diagnostic failure** (cannot freeze baseline on seed44 alone) but does **not** block the multi-seed matrix.  
→ Launch **K2-C seed43** identical protocol. A4/P01/cosine/K3-B stay frozen.
