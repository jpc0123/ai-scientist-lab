# Gate K2-A — LR×0.5 Launch Protocol

## Single factor

`learning_rate = 0.5 × 2e-4 = 1e-4`  
Backbone groups remain `0.5 × base` → `5e-5` (relative LR structure unchanged).

## Fixed (must not change)

A2 / early_concat · Gate-H manifest · 640×640 · queries 300 · 20ep · seeds 43/44 · pretrained · batch 1 · warmup · scheduler shape · aug · AMP on · checkpoint artifacts best+last · primary report **last@20**.

## Order

`K2A_seed43 → K2A_seed44` (sequential).

## Pre-frozen judgment

### seed43 effective if
- no clear ep17–20 cliff
- last@20 ≫ 0.038
- best−last ≪ 0.070
- mid/late curve smoother

### seed44 effective if
- early epochs less depressed
- preds@0.1 recovers
- last@20 > 0.022 and/or best > 0.041

### Outcome routing
1. both improve → rerun seed42 under same LR×0.5 (do not freeze v3 yet)
2. only 43 → K2-B warmup×2
3. only 44 → K2-B or scheduler review
4. neither → **do not** try LR×0.25; go K2-B

A4/P01 remain blocked.
