# Decision — P4 C0 analysis (no extra GPU)

日期：2026-08-21  
战役：v2.6 / P4  
裁决：`launch_analysis_campaign`（lite）  
未改 `max_rounds`。未改 `max_claim_strength`。未开 D-FINE R6。未宣称 G2 / G3。文献未进 ClaimGate。

## Verdict

**GO analysis / NO-GO extra GPU。** P4 两枪已闭合。用户说「继续做」且明确本仓只做实验。现协议 `max_rounds=2` 已用尽；下一枪 GPU 种子必须 Protocol Amendment。本回合只分析已有 pack。

## Action

`launch_analysis_campaign`：对比 `outputs/v26_p4_r0`（F1）与 `outputs/v26_p4_r1`（F3），写入 `outputs/v26_p4_analysis/summary.json`。

## Reason

1. 父结果已存在：RT-DETR F3 `APS_lowlight` 相对 F1 仅 +0.001708，Rubric KEEP，ClaimGate BLOCKED/C0，`scientific_outcome=INCONCLUSIVE`。
2. 信息增益最高的问题是「这点涨幅是不是检测器无关的策略迁移」，不是再磨 D-FINE。
3. 再跑 P4 seed 需要把 `max_rounds` 从 2 往上抬，代码还硬校验 `<=2`。这不是 in-protocol Next Plan。
4. 已有 `aps_lowlight.json` / `metrics.json` / `sample_predictions.json` 足够做 C0 混号分析。

## Rejected alternatives

- 静默把 P4 `max_rounds` 从 2 抬高再打 GPU
- D-FINE R6
- 把 KEEP / 主指标微涨写成 G3 / transfer success
- 改切片 / 改 C0 / 打开 A4

## Evidence paths

- `outputs/v26_p4_r0/` · `outputs/v26_p4_r1/`
- `outputs/v26_p4_analysis/summary.json`
- `docs/research/v26/V26_P4_C0_ANALYSIS.md`
- `docs/research/v26/p4_c0_analyze.py`

## Next direction

分析已闭合，结论 **INCONCLUSIVE**。用户随后 **STOP**：不改 `max_rounds`，不开 GPU。见 [`DECISION_STOP_P4.md`](./DECISION_STOP_P4.md)。
