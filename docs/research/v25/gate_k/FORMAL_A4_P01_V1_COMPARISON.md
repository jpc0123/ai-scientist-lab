# Formal A4 vs P01 v1 — Comparison

**Protocol:** `protocol_rgbt_formal_a4_p01_v1`  
**Single factor:** `neck.residual_scale=0.25`  
**Decision:** `failure_or_null__no_extra_gain_vs_a4`  
**Accepted 2026-08-08** as valuable negative → converge to **A4 final** (`DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json`). Do **not** open P02/P03.

## Per-seed (last@20)

| seed | A4 last | P01 last | Δ | A4 exec | P01 exec |
|------|---------|----------|---|---------|----------|
| 44 | 0.1217 | 0.0592 | -0.0625 | `exec_8bffa02fe333` | `exec_658ff436e569` |
| 43 | 0.1381 | 0.1139 | -0.0242 | `exec_1f526b78bc5f` | `exec_cbcaf007f8db` |
| 42 | 0.1483 | 0.1155 | -0.0328 | `exec_e5e0067e8d1b` | `exec_f9589a33a3ee` |

## Summary

| metric | value |
|--------|-------|
| A4 mean last | 0.1361 |
| P01 mean last | 0.0962 |
| mean Δ | -0.0398 |
| positive seeds | 0/3 |

## Interpretation

- Research question answered: residual scaling (0.25) does **not** improve accepted A4; it hurts under this protocol.
- Likely: A4 gains need full residual/enhancement path; and/or prior P00 gradient conflict was **gated×FDPN**, not a reason to weaken FDPN on early_concat.
- Final method: **keep A4**; drop P01 and P00.
