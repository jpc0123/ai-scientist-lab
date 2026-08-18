# Gate K3-A — Late LR Cooling Protocol (frozen)

## Intent

Original A2 used `MultiStepLR milestones=[500]`, which **never fires** in 20 epochs → constant peak LR after warmup. K3-A adds the missing late cooling.

## Single factor

| Field | Value |
|-------|-------|
| `lr_scheduler.type` | MultiStepLR |
| `milestones` | **[14]** (was [500]) |
| `gamma` | 0.1 (unchanged) |

Verified by `K3A_LR_TRACE.json` (no full train):

- `scheduler.step()` runs **after each epoch**, only when warmup finished
- history **epoch14**: backbone `1e-4`, non-backbone `2e-4`
- history **epoch15–20**: backbone `1e-5`, non-backbone `2e-5`

## Fixed (must not change)

A2 / early_concat · Gate-H · 640×640 · queries=300 · batch=1 · warmup=**500** · peak LR=**2e-4** · 20ep · pretrained · aug · checkpoint_v2 · code/image

## Seeds / order

`K3A_seed43 → K3A_seed44` (sequential). **Do not** mix old-protocol seed42.

## Success (pre-frozen)

### seed43
- best ≥ 0.08
- last@20 ≥ 0.07
- best−last ≤ 0.03
- no clear ep17–20 cliff

Ideal: mid ~0.09–0.10, after decay hold ~0.08–0.10.

### seed44
- best ≥ 0.06
- last@20 ≥ 0.05
- best−last ≤ 0.03

## Routing

| Result | Next |
|--------|------|
| 43 & 44 retain mid | Rematch seed42 under K3-A → `formal_a2_baseline_v3_candidate` |
| 43 ok, 44 early-low | K3-B: moderate warmup + same late decay |
| Both mid peak then late cliff | Retune decay timing / smoother schedule |
| Overall suppressed | 10× step too strong → cosine; do not lower full-run LR |
| Seed-specific protocols needed | Stop scheduler chasing → K2-C |

## Status

**Diagnostic candidate only** — not formal baseline v3 until seed42 rematch under identical K3-A protocol.

A4 / P01 remain blocked.
