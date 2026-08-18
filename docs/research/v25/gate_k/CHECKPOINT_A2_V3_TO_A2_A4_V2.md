# Checkpoint — A2 v3 accepted → A2/A4 v2 freeze

> **Superseded as the live resume point (2026-08-08).**  
> A4 vs A2 completed; P01 was a negative; method frozen as A4.  
> Current authority: `FINAL_METHOD_FREEZE.json` and `docs/research/v25/README.md`.  
> This checkpoint stays as history of the A2→A4 protocol freeze.

## 1. Current route (historical)

- Label: `formal_a2_a4_v2`
- Stage at this checkpoint: protocol frozen; A4 arm ready to launch
- Judgment: accept `formal_a2_baseline_v3_candidate` **with notes** (not absolute freeze) → freeze comparison protocol → open A4 vs A2

## 2. Current active node

- Authoritative protocol: `docs/research/v25/gate_k/formal_a2_a4_v2_protocol.json`
- Runner protocol: `examples/rgbt_protocol_formal_a2_a4_v2.json`
- Decision: `docs/research/v25/gate_k/DECISION_ACCEPT_A2_V3_FREEZE_A2_A4_V2.json`

## 3. Node history

- Gate-H A2 → seed instability
- K1/K2/K3 scheduler/LR chase → stopped
- K2-C sequence expand → three-seed `case_A_soft`
- Supersedes formal A2/A4 v1 (fast_eval / 80ep / different data)

## 4. Strongest retained result

- K2-C A2 last@20: 42=0.0529, 43=0.1093, 44=0.0728; mean=0.0783; range=0.0563
- A2 reuse execs: `exec_00cc3754ea3d` / `exec_d4b4feb6cfb1` / `exec_4f4151310027`

## 5. Do-not-reopen by default

- Further A2 tuning (cosine / epochs / EMA / LR)
- Absolute claim “A2 fully stable” or “determinism solved”
- P01 before A4 vs A2 answers FDPN gain
- Mixing Gate-H A2 runs into this comparison

## 6. Next resume step

- Launch A4 seed44 →43 →42 under frozen protocol; write paired Δ table

## 7. First-read files

1. `docs/research/v25/gate_k/formal_a2_a4_v2_protocol.json`
2. `docs/research/v25/gate_k/DECISION_ACCEPT_A2_V3_FREEZE_A2_A4_V2.json`
3. `docs/research/v25/gate_k/A2_BASELINE_V3_REPORT.md`
4. `docs/research/v25/gate_k/K2C_DATASET_FREEZE.json`

## 8. Reopen condition

- Only if A4 vs A2 shows protocol mismatch / hash drift / non-comparable arms, or user explicitly reopens A2 baseline movement.
