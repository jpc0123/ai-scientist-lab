# Experiment Index — v2.4 RGB-T Fusion Campaign

Local artifacts live under `outputs/experiments/v24_a3/` (gitignored).  
Code / contracts: `feat/v24-full-fusion` @ `df97636`.

## Arms (definition)

| Arm | Fusion | Neck | Role in stage end-state |
|-----|--------|------|-------------------------|
| A2 | early_concat | HybridEncoder (standard) | Formal matched baseline |
| A3 | gated_multiscale | standard | Exploratory / diagnosis contrast |
| A4 | early_concat | FDPN | Exploratory candidate (not promoted) |
| P00 | gated_multiscale | FDPN | Negative interaction; diagnosed |
| P01 | gated_multiscale | FDPN + residual_scale=0.25 | Archived exploratory candidate |

## Formal candidate A2/A4 (80ep, seeds 42/43/44, best-on-val)

Protocol: `protocol_rgbt_formal_candidate_a2_a4_v1`

| Arm | Seed | Execution ID | best mAP50_95 | Notes |
|-----|-----:|--------------|--------------:|-------|
| A2 | 42 | `exec_c2e51de78fac` | 0.4051 | |
| A4 | 42 | `exec_52c4e5c820d9` | 0.4457 | Δ=+0.041 |
| A2 | 43 | `exec_fe1e1aafaac6` | 0.4479 | |
| A4 | 43 | `exec_be11548431c6` | 0.4095 | Δ=−0.038 |
| A2 | 44 | `exec_e3511b691e3b` | 0.3677 | |
| A4 | 44 | `exec_bb3e624f22c1` | 0.4109 | Δ=+0.043 |

Reports:

- `outputs/experiments/v24_a3/formal_candidate/FORMAL_CANDIDATE_PAIRED_REPORT.json`
- `outputs/experiments/v24_a3/formal_candidate/FORMAL_CANDIDATE_REPORT.md`
- `outputs/experiments/v24_a3/formal_candidate/CLAIM_GATE_A4_VS_A2.json`
- `outputs/experiments/v24_a3/formal_candidate/PROTOCOL_FREEZE.json`

E10/E11 budget-scale seed42 runs were **not** reused in the formal matrix (`E10_E11_REUSE_AUDIT.json`).

## Exploratory multi-seed (20ep Fast Eval subset)

| Arm | Summary artifact | Seeds |
|-----|------------------|-------|
| A2 | `A2_seeds/A2_SEED_SUMMARY.json` | 42/43/44 |
| A3 | `A3_seeds/A3_SEED_SUMMARY.json` | 42/43/44 |
| A4 | `A4_seeds/A4_SEED_SUMMARY.json` | 42/43/44 |
| P00 | `P00_seeds/P00_SEED_SUMMARY.json` | 42/43/44 |

2×2 factor matrix: `factor_matrix_2x2/`

## Budget scale (seed42, 80ep exploratory)

| ID | Arm | Summary |
|----|-----|---------|
| E10 | A2 | `budget_scale/` |
| E11 | A4 | mid-epoch crossover; final A4 lead (exploratory only) |

## P00 mechanism diagnosis (seed42, 40ep, diagnostic_only)

| Arm | Execution ID |
|-----|--------------|
| A3 | `exec_203d7f2395e2` |
| P00 | `exec_7b6492f038b4` |

Artifacts: `outputs/experiments/v24_a3/p00_diagnostics/`  
Key finding: `fusion_suppressed_in_P00` → launch P01 only; skip P02.

## P01 residual scale (seed42, 20ep exploratory)

| Field | Value |
|-------|------:|
| Execution | `exec_5809ec7606c1` |
| P01 mAP50_95 | 0.24047 |
| A3 seed42 ref | 0.20525 |
| Δ | +0.035 |
| Status | `archived_exploratory_candidate` (`docs/research/v24/P01_ARCHIVE.json`) |

## Protocol / contract templates (in repo)

Under `examples/`:

- `rgbt_protocol_formal_candidate_a2_a4_v1.json`
- `rgbt_fc_a2_contract.json` / `rgbt_fc_a4_contract.json`
- `rgbt_protocol_p00_mechanism_diagnosis_v1.json`
- `rgbt_protocol_p01_residual_scale_v1.json`
- plus Gate-D / multi-seed / smoke contracts for A3/A4/P00

## Strategy decisions

- `outputs/experiments/v24_a3/STRATEGY_DECISION_STOP_EXTEND_TO_DIAGNOSIS.json`
- `outputs/experiments/v24_a3/STRATEGY_DECISION_AFTER_P01.json`
- `docs/research/v24/DECISION_TRACE.json` (ordered archive trace)
- `docs/research/v24/STAGE_FREEZE.json` (final closure)
