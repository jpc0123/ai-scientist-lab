# A2 Baseline V3 Report (K2-C three-seed)

**Decision:** `case_A_soft__v3_candidate_with_notes`  
**Acceptance:** `accepted_with_notes` (comparison candidate; **not** absolute freeze) — 2026-08-05  
**Next protocol:** `formal_a2_a4_v2` (A4 vs A2; reuse these A2 runs; do not retune A2)  
**Protocol:** `protocol_rgbt_v25_gate_k2c_seq_expand` · original A2 · early_concat  
**Data:** `v1_gate_k2c_seq_expand` (frozen; do not mix Gate-H seed42)  
**Primary metric:** last@20 (`a2_baseline_checkpoint_v2`)

## 1. Three-seed summary

| seed | last@20 | best | best−last | best_epoch | exec |
|------|---------|------|-----------|------------|------|
| 42 | 0.0529 | 0.1413 | 0.0884 | 4 | `exec_00cc3754ea3d` |
| 43 | 0.1093 | 0.1531 | 0.0438 | 17 | `exec_d4b4feb6cfb1` |
| 44 | 0.0728 | 0.1432 | 0.0703 | 2 | `exec_4f4151310027` |

## 2. Stability

| metric | Gate-H | K2-C (v3) |
|--------|--------|-----------|
| last mean | 0.0475 | **0.0783** |
| last std (pop) | 0.0254 | **0.0233** |
| last range (max−min) | 0.0599 (~0.060) | **0.0563** |

Range reduced vs Gate-H: **True**.

## 3. Checkpoint behavior

| seed | best−last |
|------|-----------|
| 42 | 0.0884 |
| 43 | 0.0438 |
| 44 | 0.0703 |

best checkpoint is sensitive to short-horizon peaks; primary results use fixed-epoch last checkpoint.

## 4. Interpretation

- This establishes a **more stable A2 baseline protocol**, not method superiority.
- A4 / P01 / FDPN / cosine remain **blocked** until v3 is accepted as the comparison anchor.
- Val set also expanded vs Gate-H; absolute last deltas vs Gate-H are not same-eval-set lifts.

## 5. Routing

- **case A**: accept `formal_a2_baseline_v3_candidate`, then A4 vs A2 formal compare.
- **case B**: seed42 low → dataset variance / sequence difficulty analysis (not A4).
- **case C**: all large gaps → consider horizon/cosine later, after v3 documentation.

Current: `case_A_soft__v3_candidate_with_notes` → **accepted_with_notes**; comparison protocol frozen at `formal_a2_a4_v2`.

## 6. Allowed / disallowed writing

**Allowed:** Under K2-C, A2 reaches non-degenerate last@20 on three seeds (mean 0.0783); lower cross-seed failure risk than Gate-H; A2 v3 is the **candidate** comparison baseline.

**Disallowed:** A2 fully stable; training determinism solved; SOTA / method superiority; A4/P01 gains before A2/A4 v2 completes.
