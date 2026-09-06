# Decision — Live A evidence-gap closeout

Date: 2026-08-30  
Campaign: `exp_rgbt_dfine_v26_lowlight_20260829T114407Z`  
裁决：`stop` + **归档 Stage A / 证据缺口战役**  
KEEP ≠ Claim。不升 ClaimGate / G2。

## Verdict

**缺口目标已满足，停止续跑并归档。**

| 目标 | 结果 |
|------|------|
| Neck 轴（A4 或 N1） | **N1 有 GPU 跑次（seed 42/43）**；另有 **A4 seed 45** |
| Best 多种子 | **F1 多 seed 已落盘**（至少 42/43/45/46）；监督器在 gpu≈20 时记 `GOAL_HIT`（F1 2/2 + N1 done） |
| Fusion 对照 | F0/F1/F3 均有多 seed GPU 证据 |
| 只测写插件 | **P4W smoke_ok → human register overlay**；不开 GPU |

`evidence_gap_priority` 已关。战役 `status=completed` / `stop`。

说明：停后 API `live_m1_brief` 曾把 `used_how_ids` 收成仅 F0/F1/F3，并把 best 指到弱 F0（单 seed）——**以 `runs/run_plan_r*` 落盘为准**，不以停后瞬时 brief 反证缺口未补。

## Why stop

1. 用户授权的缺口清单（N1/A4 + F1 第二 seed）在 run 产物上已齐。  
2. 写插件验收以 smoke + overlay register 收口；P4W **不做** 新 GPU 枪——插件通路已被 P1/P3A/P3B 在 GPU 上验过，再刷 P4W 边际低。  
3. 目录 HOW 在现预算下仍是 **方差主导**；再刷 seed / 未用插件不改变「无稳健 Claim」结论。

## P4W decision

**Register：是。GPU：否。**

- LLM 曾因 `can_enter_claim_gate=false` 误 reject；该标志对插件是契约，不是拒 register 的理由。  
- Human Gate register 进 `registered_overlay`，完成 author→smoke→accept。  
- **不** resume 为了选中 P4W；**不** 进 ClaimGate。

## Rejected alternatives

- 继续 GPU 刷 A4（N1 已覆盖 neck 轴起步）  
- 为 P4W 再开训练枪  
- 翻 `llm_may_invent_how`  
- 升 G2 / Claim

## Evidence pointers

- Campaign workdir: `.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260829T114407Z/`  
- Supervisor: `_gap_finish_supervisor.log` → `GOAL_HIT`（N1 done + seeds 2/2）  
- Pending: `howc_write_accept_P4W` → `registered`, `smoke_ok=true`  
- Prior: `DECISION_LIVE_M1_CLOSEOUT.md`, `DECISION_LIVE_A_HOW_PLUGIN.md`

## Next direction

默认 **停**。若再开：新 campaign / 新协议指纹，显式选 B（发明）或 C（加预算复现），不要在本脏历史上静默续跑。
