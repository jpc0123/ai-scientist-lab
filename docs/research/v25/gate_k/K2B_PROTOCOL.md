# Gate K2-B — warmup×2 Launch Protocol

## Single factor

`lr_warmup_scheduler.warmup_duration = 2 × 500 = 1000`  
`learning_rate = 2e-4` (**original A2 LR; do not stack K2-A LR×0.5**)

EMA `warmups` and MultiStepLR milestones remain unchanged.

## Fixed (must not change)

A2 / early_concat · Gate-H manifest · 640×640 · queries 300 · 20ep · seeds 43/44 · pretrained · batch 1 · base LR 2e-4 · scheduler endpoint · aug · AMP on · checkpoint artifacts best+last · primary report **last@20**.

## Order

`K2B_seed43 → K2B_seed44` (sequential).

## Pre-frozen judgment

### seed43 effective if
- retains mid/high peak (`best` or mid-max ≳ 0.08)
- no clear ep17–20 cliff
- last@20 ≫ 0.038
- best−last ≪ 0.070

### seed44 effective if
- early epochs less depressed and/or best/last clearly above 0.041 / 0.022
- prediction quality improves (not only more boxes)

### Outcome routing
1. both improve → rerun seed42 under warmup×2; candidate `baseline_v3`
2. 44 improve, 43 still late cliff → scheduler review
3. 43 improve, 44 still low → K2-C expand independent sequences
4. neither → **do not** increase warmup further; go K2-C
5. 43 mid-high + late cliff phenotype → late LR decay / earlier decay test
6. 44 early persistently low → prioritize expand data + matching stats

A4/P01 remain blocked.
