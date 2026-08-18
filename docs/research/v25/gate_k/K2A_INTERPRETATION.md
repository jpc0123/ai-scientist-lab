# Gate K2-A Interpretation (frozen before K2-B)

**Decision context:** `only_44_improve__go_k2b_or_scheduler`  
**Factor:** learning_rate ×0.5 (`2e-4 → 1e-4`)

## What K2-A showed

LR×0.5 is **not** a uniform fix; it acted differently on the two failing seeds.

### Seed44 — lower LR clearly helped

| metric | baseline | K2-A |
|--------|---------:|-----:|
| best | 0.0414 | 0.0816 |
| last@20 | 0.0222 | 0.0602 |

The original low trajectory was at least partly due to early optimization aggression/instability. Lower LR moved the run into a better region.

`preds@0.1 ≈ 3643` did **not** clearly rise → gains more likely from prediction quality / matching / localization than from “more boxes”.

### Seed43 — removed late cliff, but capped learning

| metric | baseline | K2-A |
|--------|---------:|-----:|
| best | 0.1082 | 0.0395 |
| last@20 | 0.0380 | 0.0395 |
| best−last | 0.0702 | 0.0000 |

Late cliff (ep17–20) and gap disappeared, but the ~0.09–0.10 mid plateau was lost. Seed43 may need original early learning speed plus gentler late updates / better schedule — not a permanently lower LR.

## Allowed conclusion

LR halving clearly improved seed44’s early-low problem and removed seed43’s late cliff, but also lowered seed43’s overall learning level. Therefore lower LR cannot be promoted as a unified stabilization protocol. Evidence suggests seed44 is more sensitive to early optimization stability, while seed43 may need to keep early learning speed and adjust warmup or late schedule.

## Implication for K2-B

Run warmup×2 at **original LR only**. Do not stack with LR×0.5, or attribution becomes ambiguous.
