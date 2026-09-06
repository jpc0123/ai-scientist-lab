# Decision — Live M1 closeout → next stage

Date: 2026-08-29  
Campaign: `exp_rgbt_dfine_v26_lowlight_20260827T144442Z`  
裁决：`stop`（本轮 GPU 结束后收工）→ 进入下一阶段设计，不自动翻发明开关

## Why stop now

Live M1 目标是目录 HOW 消融 + 多种子证据，不是自由发明。

已观察到：

- F1 / F3 / N1 均已跑过；F3 多种子方差极大（例如 seed45≈0.053 vs seed46≈0.006），**不能支撑稳健 G2**
- 后期多轮 APS_lowlight 极低或撞车，继续刷 seed 边际信息量低
- F0（RGB-only）对照在收工轮补齐；N0 可选，非必须

KEEP ≠ Claim。本场不升 ClaimGate / 不写 G2 成功。

## Closeout rule

1. `POST .../stop` → `pause_requested`，**等当前 GPU 轮结束**再 `paused`
2. 不 resume、不 extend、不开同协议新 seed 战役
3. 结论边界：catalog HOW @ 160/formal-lite 预算下，方差主导；无稳健 winner

## Next stage (choose explicitly)

默认 **不** 翻 `llm_may_invent_how`。可选：

| 选项 | 做什么 | 不做什么 |
|------|--------|----------|
| **A. HOW 插件沙箱（推荐默认）** | 文献 → `how_candidates` → Diff 写 `how_plugins/<id>/plugin.py` → 人审 + smoke → overlay | Plan 夹 Python；主树 merge；自动 Claim |
| **B. 放开发明** | 人确认后翻 `llm_may_invent_how`，新开 campaign | 在本场脏历史上继续发明 |
| **C. 加预算复现** | 更高分辨率 / 更长 epoch，只复现 F3±对照 | 当创新；当 G2 |

推荐：**A**，新 campaign + 新 protocol fingerprint；本场只作 M1 负结果/高方差证据归档。

## Not in this decision

- 自动 Claim / G2
- 静默翻 `llm_may_invent_how`
- 继续本场 multi-seed 空转
