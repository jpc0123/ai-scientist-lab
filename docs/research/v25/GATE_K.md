# Gate K — A2 Seed Sensitivity (closed)

## Status

**closed → method frozen**  
Reporting: `a2_baseline_checkpoint_v2` (`last@20`)  
Final method: **A4 = early_concat + FDPN** (`exploratory_comparison`)  
Authoritative freeze: `gate_k/FINAL_METHOD_FREEZE.json`  
Decision: `gate_k/DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json`

Blocked / not reopened: cosine-now / K3-B / P02 / P03 / gated fusion+FDPN / further A2 tuning.

This file is the campaign log. Do not treat mid-campaign phrases such as “K2-A running” or “seed42 launching” as current status.

## Final route

```text
K2-C seq expand (optimizer = original A2)
  → three-seed A2 last@20: 0.0529 / 0.1093 / 0.0728  (case_A_soft)
  → accept A2 baseline v3 with notes
  → freeze protocol_rgbt_formal_a2_a4_v2
  → A4 vs A2: mean Δ last@20 ≈ +0.058  (3/3 positive)
  → P01 residual_scale=0.25: mean Δ vs A4 ≈ −0.040  (0/3 positive)
  → freeze A4
```

## Phase progress (all done)

| Phase | Status | Note |
|-------|--------|------|
| K1 curve audit | **done** | 42 healthy · 43 late cliff · 44 low/under-activated |
| K2-A LR×0.5 | **done** | helps 44; caps 43 mid peak |
| K2-B warmup×2 | **done** | helps 44 early; 43 stable-but-capped; 44 late cliff returns |
| K3-S0 LR/horizon audit | **done** | MultiStep[500] never fires; cliffs at constant peak LR |
| K3-A earlier late decay | **done — neither success** | changed collapse shape; stop scheduler chasing |
| K3-B / cosine | **deferred / not opened** | not needed after A4 freeze |
| K2-C expand sequences | **done** | seed44 → seed43 → seed42 rematch |
| A2 baseline v3 | **accepted with notes** | usable comparison candidate, not fully stable |
| A4 vs A2 | **done** | consistent last@20 lift |
| P01 vs A4 | **done (negative)** | no extra gain |

## K2-C three-seed A2 (`last@20`)

| seed | exec | last@20 | best |
|------|------|---------|------|
| 42 | `exec_00cc3754ea3d` | 0.0529 | 0.1413 |
| 43 | `exec_d4b4feb6cfb1` | 0.1093 | 0.1531 |
| 44 | `exec_4f4151310027` | 0.0728 | 0.1432 |

Mean last@20 = **0.0783**. Floor ≥0.05 on all three seeds. `best−last` remaining gap is diagnostic-only, not a reason to keep tuning A2.

## Two-stage model (accepted, historical)

1. Early start stability (LR / warmup / init / matching)
2. Mid/late generalization stability (horizon, overfitting, data scale)

K2-C changed coverage (independent train/val sequences), not the optimizer.

## Artifacts

- Protocol: `gate_k/K2C_PROTOCOL.md` · `gate_k/K2C_DATASET_FREEZE.json`
- A2 v3: `gate_k/A2_BASELINE_V3_REPORT.md` · `gate_k/DECISION_ACCEPT_A2_V3_FREEZE_A2_A4_V2.json`
- A4: `gate_k/FORMAL_A2_A4_V2_COMPARISON.md`
- P01: `gate_k/FORMAL_A4_P01_V1_COMPARISON.md`
- Closeout: `gate_k/FINAL_EXPERIMENT_SUMMARY.md` · `gate_k/EXPERIMENT_REPORT_THESIS.md`
