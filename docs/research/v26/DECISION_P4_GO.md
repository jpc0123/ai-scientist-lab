# Decision — GO P4 RT-DETR transfer

日期：2026-08-21  
战役：v2.6 / V26.7 P4  
裁决：`launch_experiment`  
Human Gate：**用户在 R5 闭合后说「继续」，视为 P4 GO**（不是再开 D-FINE F3 seed）。  
未 commit / push。未宣称 G2 / G3。ClaimGate 仍 C0。文献不得进 ClaimGate。

## Verdict

**GO。** 开启 P4：注册 RT-DETR Adapter（HOW，不是第五个 Agent），写独立 transfer Protocol，在同一冻结切片上跑 RT-DETR F1 基线 GPU。

## Action

`launch_experiment`：`outputs/v26_p4_r0`（HOW=F1，seed=42）。F3 迁移枪是 Protocol round 1（`outputs/v26_p4_r1`），不占用 D-FINE `max_rounds=6`。

## Reason

1. R5 已闭合。再跑 D-FINE seed 是被否决的路线（不要把 max_rounds 再往上抬去磨 D-FINE）。
2. 战役 P4 问的是 **策略是否可迁移**，不是「更好的检测器」。
3. 仓库此前只有 `DFINEAdapter`。本回合补上 `RTDETRAdapter` + `research_protocol_rgbt_rtdetr_transfer_v26`（`max_rounds=2`，`max_claim_strength=C0`）。
4. Dataset Contract 仍冻结：`rgbt_tiny_v1` + `low_light_subset_v1` + fingerprint `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6`。
5. F3 HOW 在 RT-DETR Adapter 上重实现（gated_multiscale 包 backbone）。**没有**把 D-FINE FDR / FDPN / A4 塞进第二模型。N1 在该 Adapter 上 `MATERIALIZE_REJECTED`。

## Rejected alternatives

- D-FINE R6 / 再把 `max_rounds` 从 6 往上抬
- 未注册 Adapter 就宣称开始 P4 GPU
- 注册 F2 / T1 / T2 / A4，或把 A4 包装成 LLM 发现
- 改切片 / 改 `max_claim_strength`
- 宣称 G2 或 G3 / transfer success

## Evidence paths

- `src/scientist_lab/adapters/rtdetr/`
- `schemas/examples/research_protocol_rgbt_rtdetr_transfer_v26.json`
- `experiment_apps/rgbt_detection_real/vendor_rtdetr/`（lyuwenyu/RT-DETR zoo）
- `docs/research/v26/V26_5_ROUND5.md`（R5 已闭合，F3 seed45 `APS_lowlight=0.035942673873977496`）

## Next direction

P4 两枪 GPU 已完成（F1 + F3）。ClaimGate 仍 C0。不要宣称 G3。不要开 D-FINE R6。下一步若继续，只能是用户明确批准的 C0 分析 / V26.6 多种子，或 STOP。
