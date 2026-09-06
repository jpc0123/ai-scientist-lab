# Decision — ACCEPT P0 unattended two-round GPU

日期：2026-08-24  
战役：v2.6 / P0 无人值守闭环  
裁决：`stop`（本战役 GPU）+ 产品门 **ACCEPT**  
Human Gate：用户在 `/loop` 点了一次「开始自主实验」。不要再点第三次。KEEP ≠ Claim。

## Verdict

**P0 产品门通过。科学结论不升级。**

一次 Human Gate 之后，现有 Manager 在无 Cursor、无逐条 Allow 的情况下，自主完成 **两轮真实 GPU**，然后因 `max_extra_rounds=1` 停下。这证明的是 **规划 → 训练 → 读数 → 再规划** 的产品闭环，不是检测方法涨点，也不是 G2。

科学结局：**INCONCLUSIVE**。ClaimGate 停在 **C0 observational**。`keep_is_not_claim=true`。KEEP / REPLICATE ≠ Claim。

## Action

`stop`：本战役不再打 GPU。不自动开 P1。不把 0.00642 → 0.01034 写成提升。

## Reason

信息增益已经足够回答 P0 问题（「点一次能不能自己跑完两轮」）。再开第三轮只会重复 F1 seed 42，或者在比较合同未接线的情况下继续产出 `delta=None` 的 REPLICATE。

## 本仗事实

战役：`p0_20260824T111526Z`  
工作树：`.run/autonomous/p0_20260824T111526Z/`  
`status=completed` · `ok=true` · `execute=true` · `llm_live=true` · `gpu_rounds=2`  
Planner / Reviewer：`qwen3.7-max`  
停因：`max_extra_rounds reached; not starting another round`  
墙钟：`2026-08-24T11:15:26Z` → `2026-08-24T11:59:05Z`

| 轮 | Plan | HOW | seed（合同） | exec | APS_lowlight | Rubric | ClaimGate |
|----|------|-----|--------------|------|--------------|--------|-----------|
| 0 | `plan_v26_r0_dfine_f1`（bootstrap，非 LLM 选 HOW） | F1 `early_concat` | 42 | `exec_a21af31fdcdb` | 0.0064238897679556655 | REPLICATE | C0 observational / INCONCLUSIVE |
| 1 | `plan_round1_from_run_plan_v26_r0_dfine_f1`（Live LLM） | F1 | 42 | `exec_a4ee308b50c6` | 0.010341766523443782 | REPLICATE | C0 observational / INCONCLUSIVE |

Round 1 Planner 明确 **不选** F3、F0，理由是 Reviewer 要求先确认 F1 基线。这是反馈规划，不是发现新方法。

## 不能写成的话

- 不能写成 G2 / 融合模块有效 / F1 涨点。两轮是同一 HOW、同一 seed；数字差是 2-epoch 噪声量级，不是对照实验。
- 不能把这两枪接到冻结锚 `outputs/v26_r0` 的 `APS_lowlight=0.0045926865160844455` 上做 Δ。本战役 **没有** 写出 `baseline_metrics.json`，两轮 Reviewer 的 `delta` 都是 `null`。
- 不能写成「多种子复现」。LLM 假设写了 new seed，`evaluation.seeds` 仍是 `[42]`。
- 不能覆盖 V26.5 R1–R5 或 P4 的既有证据。那些战役仍然有效，本仗只验收无人值守入口。

## Role lock

| P0 是 | P0 不是 |
|--------|---------|
| 一次授权、两轮真实 GPU、无 Cursor | 检测方法结论 |
| Manager 按结果生成下一轮 Plan | LLM 发明了新算子 |
| KEEP / REPLICATE 是下一动作 | ClaimGate 升级 |
| 产品闭环验收 | G2 / G3 |

## Rejected alternatives

- 再点「开始」打第三轮 GPU
- 自动启动已取消的 P1 rules vs LLM（`.run/p1_planner_quality/STATUS.json` = `cancelled_by_user`）
- 把 Round 1 的更高 APS 写成 KEEP-success 或 G2
- 把 Harness / 新 HOW / 新 Agent 叠进这一仗

## Evidence paths

- `.run/autonomous/p0_20260824T111526Z/campaign.json`
- `.run/autonomous/p0_20260824T111526Z/runs/run_plan_v26_r0_dfine_f1/`（Round 0 归档）
- `.run/autonomous/p0_20260824T111526Z/result.json` · `review.json` · `claim_gate.json`（Round 1）
- `outputs/project_rgbt_cuda_001/exec_a21af31fdcdb` · `exec_a4ee308b50c6`
- 冻结 R0（**未**注入本战役 Rubric）：[`R0_BASELINE.md`](./R0_BASELINE.md)

## Next direction

**STOP GPU。** GPU 租约已释放，无 `scientist-exec-*` 占用。

下一拍需要新的 Human Gate，三选一（默认推荐 A）：

1. **A · 修比较合同，不打 GPU**：让 REPLICATE 同 HOW 时 seed 必 +1；把 `outputs/v26_r0/aps_lowlight.json` 注入战役 baseline。这两处本仗都没打上。
2. **B · P1 Planner 质量**：同一协议、同一 catalog，只换 Planner；新的一次授权。不要用这场 F1×2 当对照臂。
3. **C · 演示冻结**：对外只说无人值守两轮闭环已跑通；科学数字继续用 V26.5 / P4。

## Status（2026-08-24）

已完成。无新 GPU。KEEP ≠ Claim。
