# Gate K3-S0 — LR Schedule and Horizon Audit

**Completed:** `2026-08-03T07:58:17.924484+00:00`  
**Decision:** `k3s0_complete__prefer_k3a_earlier_decay__not_more_warmup_or_lr025`  
**Cost:** zero retrain (reuse existing histories/configs)

## Parent framing (accepted)

A2 is not a single-hyperparameter failure. Evidence supports two stages:

1. **Early start stability** — LR / warmup / init / matching
2. **Mid/late generalization stability** — schedule horizon, overfitting, data scale

K2-A/K2-B already showed neither LR×0.5 nor warmup×2 is a unified formal protocol.

## Logging convention

- `training_history.csv` column `lr` = `optimizer.param_groups[0]` = **backbone LR** (= 0.5 × configured base).
- Non-backbone peak LR = configured base (= **2 × logged lr** after shared warmup factor).
- Steps/epoch on Gate-H subset ≈ **1830** (batch=1).

## Scheduler facts (critical)

| Item | Value |
|------|-------|
| `lr_scheduler` | MultiStepLR |
| `milestones` | **[500]** (epochs) |
| `gamma` | 0.1 |
| Fires inside 20-epoch budget? | **No** |
| Shape after warmup | **Constant peak LR through ep20** |

**Implication:** ep17–20 late cliffs on original seed43 and K2-B seed44 occur while LR is still at peak. They are **not** explained by a MultiStep kink near ep16–17.

## Warmup timing

| Protocol | warmup steps | Ends at | Peak-LR loss vs original |
|----------|-------------:|---------|--------------------------|
| original / K2-A | 500 | ~0.27 into epoch 1 | — |
| K2-B | 1000 | ~0.55 into epoch 1 | only **~0.27 epoch** shorter peak exposure |

Therefore: seed43’s full-run ~0.04 cap under K2-B is **not** mainly “warmup ate the 20-epoch budget.” Soft start more likely changes the optimization trajectory/basin.

## Direct answers

1. **seed43 original cliff (ep17–20) LR:** backbone `1e-4`, non-backbone `2e-4` (constant peak).
2. **K2-B warmup ends:** within epoch 1 (step 1000/1830).
3. **Does warmup×2 crush peak-LR time?** Only ~0.27 epoch — not multi-epoch.
4. **Scheduler inflection at ep16–17?** **No** (milestone 500 never hits).
5. **Backbone vs other sync?** Shared LinearWarmup factor; ratio stays 1:2; no late differential decay.

## Fixed-endpoint table (mAP50_95)

| Protocol | Seed | mAP@16 | mAP@20 | Δ16→20 | best | best_ep | best−last |
|----------|-----:|-------:|-------:|-------:|-----:|--------:|----------:|
| original | 42 | 0.0552 | 0.0822 | +0.0270 | 0.0833 | 19 | 0.0012 |
| original | 43 | 0.1032 | 0.0380 | -0.0652 | 0.1082 | 3 | 0.0702 |
| original | 44 | 0.0351 | 0.0222 | -0.0129 | 0.0414 | 15 | 0.0192 |
| k2a_lr_half | 43 | 0.0354 | 0.0395 | +0.0040 | 0.0395 | 20 | 0.0000 |
| k2a_lr_half | 44 | 0.0625 | 0.0602 | -0.0023 | 0.0816 | 4 | 0.0214 |
| k2b_warmup_x2 | 43 | 0.0385 | 0.0394 | +0.0009 | 0.0428 | 18 | 0.0034 |
| k2b_warmup_x2 | 44 | 0.0782 | 0.0330 | -0.0452 | 0.0896 | 6 | 0.0566 |

### Original protocol cross-seed stats

| Endpoint | mean | sample std | range | min | max |
|----------|-----:|-----------:|------:|----:|----:|
| epoch16 | 0.0645 | 0.0350 | 0.0681 | 0.0351 | 0.1032 |
| epoch20 (last) | 0.0475 | 0.0311 | 0.0599 | 0.0222 | 0.0822 |

**Reading:**

- At **epoch16**, original seed43 is still healthy (~0.103); collapse is specifically **ep17–20**.
- Cross-seed **range at ep16 (~0.068) is not better** than last@20 (~0.060). Shortening to 16 alone would help seed43’s cliff but would **not** unify seeds (seed44 still ~0.035 at ep16).
- K2-A seed44 is the rare case where last@20 stays relatively high without a sharp ep17–20 cliff.
- K2-B seed44 recreates an ep17–20 cliff after a strong mid peak (~0.09) — late instability persists once early start is fixed.

## Interpretation for next train

Because current MultiStep **never decays** in 20ep, “earlier decay” is not tweaking an existing late kink — it is **introducing a missing late-stage cooling** while keeping:

- warmup = 500
- peak LR = 2e-4
- epochs = 20

That matches proposed **K3-A**. Do **not** try longer warmup or LR×0.25 next.

Caveat: existing ep16 numbers are diagnostic only. If formal horizon becomes 16ep and scheduler depends on total epochs, rematch full runs.

## Routing

| After K3-A | Next |
|------------|------|
| 43+44 both stable | consider seed42 under K3-A; candidate v3 path |
| 43 fixed, 44 early-low remains | K3-B moderate warmup + earlier decay (combo; label as such) |
| still mid/late overfit across protocols | stop scheduler chasing → **K2-C** expand independent sequences |

A4/P01 remain blocked.

## Artifacts

- `docs/research/v25/gate_k/K3S0_AUDIT.json`
- `docs/research/v25/gate_k/K3S0_LR_CURVES.json`
- `outputs/experiments/v25_real_rgbt/gate_k_seed_sensitivity/k3_s0/`
