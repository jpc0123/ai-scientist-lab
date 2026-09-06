# Decision — Live B: formal long-train epochs=8 (P3 vs F1)

Date: 2026-08-31  
Prior: `DECISION_LIVE_B_P3_GPU.md`（2ep P3 多 seed 方差主导 / INCONCLUSIVE）  
裁决：`continue`（**新 campaign**；用户前端手动 Start）  
Human Gate：用户确认「epoch8，改写一下，开新实验，前端手动开始」

## Verdict

**把 Adapter formal 训练从 2 epoch 提到 8 epoch；新开战役做同预算 F1 对照 + P3 多 seed。不换数据集。KEEP ≠ Claim。**

## Why

2ep 下 P3 `APS_lowlight` 均值≈0.005、σ≈0.005，best≈0.011。评测切片仍是 100 图；加长训优先于换库。`extend` 轮数不能加长单次训练。

## Action

1. 代码：`_FORMAL_TRAIN_KNOBS.epochs = 8`（`src/scientist_lab/adapters/dfine/how.py`）。  
2. **不要 resume** `…033419Z`（2ep 归档）。  
3. 用户在 Web 前端对新战役点 Start（见下方 UI）。  
4. Start 后：seed `registered_overlay.P3` + steer「先 F1@8ep 1–2 seed，再 P3 多 seed≥3」；`llm_may_invent_how=false`。  
5. 旧 R0/2ep 不可直接当 8ep baseline；本场至少跑 **F1@8ep** 配对对照。

## UI Start（用户手动）

前端**没有**上表那些英文 API 字段。页面「开始自主实验」会自动带：
`confirm_human_gate` / `execute` / `llm_live`，以及 `max_extra_rounds=11`；**默认不发明**（不传 invent）。

入口：**导航「闭环」** → `/loop`（标题区有「开始自主实验」）

操作：
1. 实验下拉选 **Low-light RGB-T … D-FINE (v2.6)**（`exp_rgbt_dfine_v26_lowlight`）
2. 审阅协议后勾选：**「我已审阅上述协议，确认冻结后启动」**
3. 插件作者选 **LLM**（不要选 harness）
4. 可选点「检查 GPU」
5. 点 **「开始自主实验」**，确认弹窗再确认  
6. **不要点「续跑」**旧战役 `…033419Z`

Start 成功后把新的 `campaign_id` 发助手，或说「已开」，再挂 P3 overlay + steer。

## Role lock

| 这是 | 这不是 |
|------|--------|
| 训练预算 Amendment（epochs） | 换 dataset / slice |
| 新战役可比证据 | 2ep 尖峰 Claim / G2 |
| F1@8ep 对照 + P3 复现 | 空转刷 catalog A4 |

## Evidence paths

- Adapter knobs：`src/scientist_lab/adapters/dfine/how.py`  
- 2ep 归档：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260831T033419Z/`  
- 插件：`experiment_apps/.../how_plugins/P3/plugin.py`
