# Decision — Live B coverage arm closeout → true invent

Date: 2026-08-30  
Campaign: `exp_rgbt_dfine_v26_lowlight_20260830T053239Z`  
裁决：`stop`（目录覆盖臂）+ **新开真正 invent 战役**  
KEEP ≠ Claim。不升 ClaimGate / G2。

## Verdict

**目录覆盖目标已满足，停止本场覆盖续跑。**

| 目标（active steer） | 结果 |
|------|------|
| F1 ≥2 seeds | **seed 42 / 43 已落盘** |
| F3 ≥2 seeds | **seed 42 / 43 / 44 已落盘**（R19 = F3 seed 43） |
| A4 ≥2 seeds | **seed 42 / 43 已落盘** |
| 禁 plugin/invent（本臂） | 遵守；P2 有历史 GPU，但非本臂要求 |

末轮 R19：`APS_lowlight≈4.1e-5`，KEEP，ClaimGate BLOCKED。  
停机直接原因：下一轮 Planner 遇 LLM `getaddrinfo failed`（瞬时 DNS），非科学 stop_rules。

## Why stop coverage

1. 人审要求的 F1/F3/A4 多种子覆盖已齐；再刷 catalog seed 边际低。  
2. 本场虽挂 `B_invent`，但 active steer 禁止 invent——**不算 Stage B invent 成功指标达成**。  
3. 按 `DECISION_LIVE_B_INVENT.md`：真正 invent 用**新 campaign + 新指纹副本**，不在覆盖脏历史上静默翻目标。

## Next — true invent

新 campaign：`llm_may_invent_how=true`，steer = invent-first（允许 invent-fallback / 文献 HOW 草稿 → 人审 → author/smoke → overlay；可选 GPU；禁空转刷 F1/F3/A4；`is_claim=false`）。

成功指标：至少一条 invent 草稿进入人审/author/smoke。不自动 Claim / G2。

## Evidence

- Workdir: `.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260830T053239Z/`
- R19: `run_plan_r19_e98d80677c`（F3 seed 43）
- Prior: `DECISION_LIVE_B_INVENT.md`
