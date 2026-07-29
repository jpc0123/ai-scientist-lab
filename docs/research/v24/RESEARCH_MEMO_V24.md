# RESEARCH MEMO — Scientist Lab v2.4 RGB-T Fusion Campaign

**Stage status:** archived / closed (`A + C`)  
**Date:** 2026-07-29  
**Branch / commit:** `feat/v24-full-fusion` @ `df97636`  
**Remote:** https://github.com/jpc0123/ai-scientist-lab.git

## Canonical conclusion

在冻结的正式候选协议下，A4 相对 A2 未表现出跨随机种子一致优势，因此不晋升为正式优越方案。P00 中 gated fusion 与 FDPN 的组合出现负交互，诊断显示 FDPN 对融合模块梯度存在压制。P01 通过残差缩放缓解该问题，并在单种子探索性实验中获得正向差值，但当前证据不足以支持稳定提升或正式优越性声明。

## Stage freeze

| Line | Status |
|------|--------|
| A2 / A4 formal candidate | **Closed** |
| A4 | **Not promoted** (exploratory candidate only) |
| P00 | Negative interaction; mechanism diagnosed |
| P01 | Archived exploratory candidate; not formalized |
| gated + FDPN joint branch | **Paused** |
| Seeds 45/46 | Not extended |
| Option B (P01 multi-seed formal) | **Not selected** |

## What was answered

1. **Can A4 stably beat A2 under a frozen formal-candidate protocol?**  
   No. Paired best-on-val deltas were `[+0.041, −0.038, +0.043]` (2/3 positive). Claim gate: `not_supported`.

2. **Why does P00 (gated fusion + FDPN) fail to help?**  
   Mechanism diagnosis (A3 vs P00, seed42): fusion gradient L2 suppressed (~8.76 → ~3.48) while FDPN grads dominate; not primarily gate saturation or near-copy redundancy.

3. **Does a constrained repair (P01 residual scale) recover?**  
   On one exploratory seed42 run, yes (+0.035 vs A3). Insufficient for stability or formal superiority.

## Research loop completed (AI Scientist value)

```text
propose methods
→ multi-arm ablation (A2/A3/A4/P00)
→ matched multi-seed formal candidate (A2/A4)
→ refuse inconsistent superiority claim
→ mechanism diagnosis (P00)
→ constrained repair (P01 only)
→ pause branch for insufficient evidence
```

This closed loop—especially **refusing promotion** and **stopping seed expansion**—is a primary deliverable of this stage, not only mAP numbers.

## Allowed vs forbidden claims

**Allowed**

- Formal-candidate A4 vs A2: mixed paired deltas; no consistent advantage.
- P00: negative interaction with fusion-gradient suppression evidence (diagnostic_only).
- P01: single-seed exploratory positive delta; archived, not promoted.

**Forbidden**

- “A4 显著优于 A2” / formal superiority / statistical significance on 3 seeds.
- “A4 更稳定” from n=3 sample std alone.
- “P01 稳定提升 / 正式优越方案”.
- Reusing exploratory P01 as formal multi-seed evidence without a new freeze.

## Next mainline priorities (post-archive)

1. Realistic input size and fuller dataset  
2. Stronger, reproducible D-FINE baseline  
3. Systematic APS / AP75 / recall (small-object) analysis  
4. End-to-end inference latency and deployment  
5. AI Scientist auto-decision / stop-policy hardening  
6. Revisit P01 **last**, and only under reopen conditions in `P01_ARCHIVE.json`

## Document map

| File | Role |
|------|------|
| `STAGE_FREEZE.json` | Machine-readable stage closure |
| `CLAIM_SUPPORT_MATRIX.json` | Claim → support status |
| `DECISION_TRACE.json` | Ordered route decisions |
| `EXPERIMENT_INDEX.md` | Execution / artifact index |
| `LIMITATIONS.md` | Scope and validity bounds |
| `P01_ARCHIVE.json` | P01 archival metadata |

Local run artifacts (gitignored): `outputs/experiments/v24_a3/`
