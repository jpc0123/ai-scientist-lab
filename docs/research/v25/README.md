# v2.5 Native-Resolution Baseline Mainline

Opened after v2.4 A+C archive. **Does not reopen** A4 promotion or P01 formalization.

## Gate F — PASSED
## Gate G — probe passed
## Gate H — PASSED (exec_4088755cafc6)
## Gate I — COMPLETED (engineering), **not frozen as formal baseline**

GATE_I_A2_FORMAL_MULTISEED · A2 · Gate-H subset · 20ep · seeds 42/43/44

| seed | exec | best ep | mAP50_95 |
|------|------|---------|----------|
| 42 | exec_4088755cafc6 (reuse H) | 19 | 0.0833 |
| 43 | exec_a6f6dff68971 | 3 | 0.1082 |
| 44 | exec_373196093d79 | 15 | 0.0414 |

- mean≈**0.0777** · stdev≈0.0337 · decision: **diagnose_seed_instability**
- Engineering all pass; formal candidates **not** OK (seed44 outlier)
- Report: GATE_I_REPORT.md · freeze: GATE_I_FREEZE.json

## Gate J — draft only
Throughput probe drafts remain; do not start until instability diagnosis decided.

## Gate J — Seed Instability Diagnosis (active)

- **J1 PASS**: subset membership identical across 42/43/44; yaml differs only by seed
- **J2 running**: seed44_rep1 = exec_a0f800e20389 → then seed43_rep1
- Docs: GATE_J.md · audit under outputs/.../gate_j_seed_diagnosis/
- Still blocked: A4/P01
