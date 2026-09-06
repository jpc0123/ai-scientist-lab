# Decision — STOP P4 as INCONCLUSIVE transfer probe

日期：2026-08-21  
战役：v2.6 / P4  
裁决：`stop`  
Human Gate：**用户明确 STOP。** 不把 P4 `max_rounds` 从 2 提到 3。不再打 GPU。

## Verdict

**STOP。** Accept C0 analysis. Archive P4 as **INCONCLUSIVE Cross-model Transfer Probe**. Persist evidence-linked strategy lesson. Keep ClaimGate **C0 / BLOCKED**. Do not BAN `gated_multiscale`.

## Action

`stop`：无 Protocol Amendment，无额外 GPU。Memory：`LESSON-V26-P4-PROBE-INCONCLUSIVE-001` + `STRATEGY-V26-P4-GATED-MULTISCALE-001`（`action=keep`）。

## Reason

再加一枪（`max_rounds` 2→3）信息增益低：无论跑 F3 seed43 还是 F1 seed43，都没有同 seed 对照，回答不了「F3−F1 在 RT-DETR 上是否可重复」。稳定性需要 matched pair（42 已有；43/44 各要两枪）。当前单 seed、2-epoch、指标符号冲突、成本 ×1.7、跨模型 Δ 约为 D-FINE 的 1/10，已经足够把 P4 收成有内容的 INCONCLUSIVE，而不是 SUCCESS 或 FAIL。

## Role lock

| P4 是 | P4 不是 |
|--------|---------|
| Cross-model Transfer Probe | Cross-model Generalization Validation |
| 没观察到足够强的跨模型正证据 | F3 在 RT-DETR 上无效 |
| INCONCLUSIVE | 失败的跨模型验证 |

## Rejected alternatives

- 把 `max_rounds` 2→3 再打一枪 unmatched seed
- `failed_cross_model=true` / BAN F3
- 把 KEEP 升成 KEEP-transfer-success 或 G3
- 把 D-FINE 路径写成已执行 F2（v2.6 未注册、未跑 F2）

## Evidence paths

- `outputs/v26_p4_r0/` · `outputs/v26_p4_r1/`
- `outputs/v26_p4_analysis/summary.json`
- `outputs/v26_p4_analysis/strategy_card_gated_multiscale.json`
- `outputs/v26_p4_r1/memory/research_memory.json`
- `docs/research/v26/V26_P4_C0_ANALYSIS.md`

## Next direction

**STOP。** 再开实验需要新的 Human Gate。若将来要 generalization claim，最低要求是 matched multi-seed F1/F3 pairs，而不是单枪补 seed。
