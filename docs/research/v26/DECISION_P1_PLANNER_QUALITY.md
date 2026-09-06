# Decision — P1 Planner quality: INCONCLUSIVE

日期：2026-08-24  
战役：v2.6 / P1（用已跑完的前端两轮，不新打 GPU）  
裁决：`stop`  
问题：同一协议、同一 RGBT-Tiny，规则 Planner 两轮 vs LLM Planner 两轮，比的是规划质量，不是 APS。

## Verdict

**INCONCLUSIVE。没有 Planner 赢家。**

两臂都把无人值守两轮跑通了，但规划质量尺子上打的是同一组失败：R1 空转同一 HOW+同一 seed，Rubric Δ 为 `null`，没有做成可证伪对照。KEEP ≠ Claim。不是 G2。

官方 `score.py` 若直接打这两场，会因为规则 R1 Plan **没写 `how_id`** 而把「空转」算成 false，从而误判 `winner=rules`。Adapter 实际跑的仍是 F1 `early_concat` seed 42。**作废该赢家。**

## Action

`stop`：不根据 APS 选边，不重跑这两臂 GPU。修补 seed / previous-shot Δ 之前，再打 2×2 只会重复空转。

## 比的是什么

| 尺子 | 过线条件 | 规则 | LLM |
|------|----------|------|-----|
| 对照 | R1 Rubric Δ 非空，或执行了不同 HOW | 未过 | 未过（只在候选里写了 F3/F0，没跑） |
| 空转同一 HOW | R1 执行 HOW+seed 与 R0 相同 | **空转** F1/42 | **空转** F1/42 |
| REPLICATE 换 seed | 同 HOW 复现时 `evaluation.seeds` 变化 | 未过 | 未过（口头 new seed，合同仍 42） |
| Δ 为负还 KEEP | Rubric Δ<0 且决策 KEEP | **测不到**（Δ=`null`，决策是 REPLICATE） | **测不到** |

R0 两边都是冻结种子计划 F1，不是 Planner 发现。质量只看 R0→R1。

## 本仗事实

同一协议 `research_protocol_rgbt_dfine_v26`，同一 `dataset:rgbt_tiny_v1`。

| 臂 | 战役 | R0 APS_lowlight | R1 计划 | R1 执行 | R1 APS_lowlight | Rubric |
|----|------|-----------------|---------|---------|-----------------|--------|
| 规则 | `p0_20260824T094123Z` | 0.03146 | `replicate_fusion`，无候选 HOW | F1 seed 42 | 0.00128 | REPLICATE，Δ=`null` |
| LLM | `p0_20260824T111526Z` | 0.00642 | 选 F1，不选 F3/F0；假设换 seed | F1 seed 42 | 0.01034 | REPLICATE，Δ=`null` |

事后 APS 差（**不是** Rubric Δ，不能当规划质量分）：规则 −0.03018，LLM +0.00392。规则侧大跌时系统仍 REPLICATE，是因为没把上一枪写进 Rubric，不是 Planner 在负 Δ 上主动 KEEP。

## Role lock

| 这是 | 这不是 |
|------|--------|
| 规划行为对照（空转 / 对照 / seed / Δ） | 谁 APS 更高谁赢 |
| C0 exploratory | G2 / 模块有效 |
| LLM 考虑过对照但没执行 | LLM 规划质量已证明更好 |

## Rejected alternatives

- 采用 `score.py` 的 `winner=rules`
- 用 APS 0.010 vs 0.001 宣布 LLM 赢
- 把规则侧 −0.030 写成「负 Δ 还 KEEP」的 Planner 污点（Reviewer 没看到 Δ）
- 立刻再打修补后的 2×2 GPU（先修合同）

## Evidence paths

- `.run/p1_planner_quality/SCORE.json`
- `.run/p1_planner_quality/CONTRACT.json`（写明这两场 unpatched，修补臂未跑）
- `.run/autonomous/p0_20260824T094123Z/`
- `.run/autonomous/p0_20260824T111526Z/`
- [`DECISION_P0_ACCEPT.md`](./DECISION_P0_ACCEPT.md)

## Next direction

**STOP GPU until the next Human Gate.** 2026-08-25 代码已分两条线：规则臂（人工特选）仍可 F1→F3→F0；**当前启用的是 LLM 路线**，不会把模型选的 HOW 改写成 F3，同 HOW 复现只 bump seed。冻结 `outputs/v26_r0/aps_lowlight.json` 可注入 baseline。重启 API 后再点一次开始才能验证。不要用 2026-08-24 那两场空转数字当 G2。不要预期下一枪一定是 F3。

## Status（2026-08-24）

已完成。无新 GPU。KEEP ≠ Claim。

## Status（2026-08-25）

Planner 已拆线：规则=人工特选对照；LLM=自主选 HOW（不强制 F3）。尚未 live GPU 复验。KEEP ≠ Claim。
