# v2.5 Native-Resolution RGB-T Campaign

Opened after v2.4 A+C archive. **Does not reopen** v2.4’s A4-promotion question under the old protocol.

**Campaign status: method frozen (2026-08-08).**  
Authoritative freeze: [`gate_k/FINAL_METHOD_FREEZE.json`](./gate_k/FINAL_METHOD_FREEZE.json)  
Decision: [`gate_k/DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json`](./gate_k/DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json)

This line is **exploratory comparison** at 640 / 20ep / `mAP50_95 last@20`.  
It is **not** MVP Formal C1 (160×160 / 2 epoch / APS). It is **not** SOTA.

## Final method

```text
K2-C expanded sequences
  → A2 baseline v3  (comparison baseline, case_A_soft — not absolute stability)
  → A4 = early_concat + FDPN   ← FINAL
  → P01 residual_scale=0.25    ← valuable negative; drop
```

Main numbers (`last@20`):

| seed | A2 | A4 | Δ(A4−A2) | P01 | Δ(P01−A4) |
|------|-----|-----|----------|-----|-----------|
| 42 | 0.0529 | 0.1483 | +0.095 | 0.1155 | −0.033 |
| 43 | 0.1093 | 0.1381 | +0.029 | 0.1139 | −0.024 |
| 44 | 0.0728 | 0.1217 | +0.049 | 0.0592 | −0.063 |
| mean | 0.0783 | **0.1361** | **+0.058** | 0.0962 | **−0.040** |

Keep: **A4**. Drop: P01, P00 gated+FDPN, further A2 scheduler chasing, cosine/K3-B.

Allowed claim: under the frozen K2-C protocol, A4 consistently lifts last@20 across three seeds; residual scaling does not add gain.  
Disallowed: significance, SOTA, “FDPN always helps”, mixing Gate-H numbers into this comparison.

## Gate history (closed)

| Gate | Status | Note |
|------|--------|------|
| F | PASSED | dataset register / quality |
| G | probe passed | tiny A2 probe |
| H | PASSED | `exec_4088755cafc6` curve health |
| I | COMPLETED (engineering) | 3-seed; **not** frozen as formal baseline (seed44 outlier) |
| J | closed into K | caseC: loss reproducible; best-on-val unstable; report `last@20` |
| K | **closed** | K1–K3-A / K2-C done; A2 v3 accepted with notes; A4 frozen |

Gate I seed table (historical; superseded by K2-C A2 v3):

| seed | exec | best ep | mAP50_95 |
|------|------|---------|----------|
| 42 | exec_4088755cafc6 (reuse H) | 19 | 0.0833 |
| 43 | exec_a6f6dff68971 | 3 | 0.1082 |
| 44 | exec_373196093d79 | 15 | 0.0414 |

## Do not reopen by default

- Further A2 tuning (cosine / extra epochs / EMA / LR)
- P02/P03 residual-scale search
- Gated fusion + FDPN (P00)
- Writing this campaign as MVP C1 or as a paper-protocol SOTA result

## First-read files

1. `gate_k/FINAL_METHOD_FREEZE.json`
2. `gate_k/FINAL_EXPERIMENT_SUMMARY.md`
3. `gate_k/DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json`
4. `gate_k/FORMAL_A2_A4_V2_COMPARISON.md`
5. `GATE_K.md` (campaign log; closed)
