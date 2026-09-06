# Gate K2-C — Expand Independent Sequences

## Decision context

K3-A (and prior LR/warmup/decay variants) changed collapse *shape* but did not unify seed43/44 under one stable protocol. Stop scheduler chasing. Cosine / K3-B deferred.

## Single factor

**Only** expand independent train/val sequences. Keep original A2 optimization:

- warmup = 500  
- base LR = 2e-4 / backbone 1e-4  
- epochs = 20  
- default MultiStep `[500]` (inactive in 20ep)  
- reporting: `a2_baseline_checkpoint_v2` (primary last@20)  
- frame_stride = 5, max_frames_per_seq = 50, seed = 42  

## Data (frozen)

See `K2C_DATASET_FREEZE.json`.

| | Gate-H | K2-C |
|--|--------|------|
| train sequences | 40 | **72** (+32) |
| val sequences | 10 | **13** (+3) |
| train pairs | 1830 | **3342** |
| val pairs | 500 | **650** |

Gate-H sequences fully retained as subset. Leakage check passed. Category map unchanged.

`dataset_version`: `v1_gate_k2c_seq_expand`  
`split_reference`: `split:rgbt_tiny_v1_gate_k2c_seq`

## Order

1. **K2-C0** dataset freeze — done  
2. **K2-C1 seed44** first  
3. If seed44 clears floor → **seed43** same protocol  
4. If both improve → rematch seed42 → `formal_a2_baseline_v3_candidate`

## seed44 success floor

- last@20 ≥ 0.05  
- best−last ≤ 0.03  
- early not persistently low; mid capability retained  

If last still ~0.02 → data scale not sole cause.

## Blocked

A4 / P01 / cosine-now / K3-B / any scheduler change / densify-only stride.

## Launch notes (2026-08-04)

- First seed44 submit failed: `train: illegal_bbox_count=13` (`x<0` in DJI_0028_2/4).
- Dropped those 13 boxes (56663→56650); register now skips `x<0|y<0`; audit OK.
- Relaunch: `exec_4f4151310027` / `job_05837156ba89` under original A2 + `v1_gate_k2c_seq_expand`.
