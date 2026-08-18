# Gate K2-B Interpretation (frozen before K3-S0)

**Decision context:** `only_44_improve__43_not_cliff__scheduler_or_lr_schedule_review`  
**Factor:** `warmup_duration 500 → 1000` at original LR `2e-4` (not stacked with K2-A)

## Seed43 — stable, but capped

| metric | original | K2-B |
|--------|---------:|-----:|
| best | 0.1082 | 0.0428 |
| last@20 | 0.0380 | 0.0394 |
| best−last | 0.0702 | 0.0034 |

Late cliff gone; gap collapsed. But the ep3–16 ~0.09–0.10 plateau never forms. This is **stability at a lower learning level**, not a healthier mid-peak trajectory.

## Seed44 — early start fixed; late cliff remains

| metric | original | K2-B |
|--------|---------:|-----:|
| best | 0.0414 | 0.0896 |
| last@20 | 0.0222 | 0.0330 |
| ep1–5 mean | ~0.023 | ~0.059 |
| best−last | 0.0192 | 0.0566 |

Early activation clearly improved (seed44 is trainable). ep17–20 late cliff returns — problem shifts toward mid/late generalization instability, closer to original seed43’s late failure mode.

## Allowed conclusion

Longer warmup can improve seed44 early performance but does not prevent its ep17–20 collapse; for seed43 it removes late cliff while compressing the whole platform to ~0.04. Combined with K2-A, neither permanent LR cut nor warmup×2 alone yields a unified formal protocol. Prefer a two-stage model (early start vs mid/late generalization) and audit actual LR/horizon before the next train.
