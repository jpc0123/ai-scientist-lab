# Decision — Live A: HOW plugin sandbox (post-M1)

Date: 2026-08-29  
Prior: `DECISION_LIVE_M1_CLOSEOUT.md`  
裁决：`continue`（新 campaign，阶段 A）  
Human Gate：用户确认「先做 A」

## Verdict

**进入 HOW 插件沙箱阶段。不翻 `llm_may_invent_how`。**

在冻结的 v2.6 协议上新开战役：LLM 可 add → author（`how_plugins/<id>/plugin.py`）→ smoke → accept 进战役 `registered_overlay`；Planner 只选已物化 HOW。文献与插件不得进 ClaimGate。KEEP ≠ Claim。

## Action

1. M1 campaign `exp_rgbt_dfine_v26_lowlight_20260827T144442Z` 保持 `completed` 归档，不 resume。  
2. 新 campaign 绑定同一实验 `exp_rgbt_dfine_v26_lowlight`：`llm_live=true`，`llm_how_lifecycle=true`（默认随 llm_live），`plugin_worker=llm`，`execute=true`，`max_extra_rounds=11`。  
3. 成功指标：至少出现一条可 smoke 的 plugin 草稿，或明确记录拒接/空转边界。不自动升 G2。

## Reason

M1 已证明目录 HOW（F0/F1/F3/N1）在当前预算下方差主导、无稳健 winner。方法空间仍冻在 catalog 时，再刷 seed 无增量。阶段 A 用既有 Diff 沙箱扩 HOW，比解冻 Planner 发明更小、更符合 Architecture Freeze。

## Role lock

| 这是 | 这不是 |
|------|--------|
| Adapter 侧插件生命周期 | Planner 夹 Python / 自由发明 |
| 战役 overlay 注册 | merge 主树 / 改 vendor D-FINE |
| 文献 → how_candidates 草稿 | ClaimGate 证据 |
| 新 campaign | 在 M1 脏历史上续跑发明 |

## Rejected alternatives

- B：翻 `llm_may_invent_how`（用户未选）  
- C：仅加预算复现 F3（用户未选）  
- 在已 completed 的 M1 campaign 上 resume 发明  

## Evidence paths

- `docs/research/v26/DECISION_LIVE_M1_CLOSEOUT.md`
- `docs/research/v26/DECISION_OPEN_HOW_PLUGIN.md`
- `docs/research/v26/DECISION_LLM_HOW_LIFECYCLE.md`
- 新 campaign 目录：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_<stamp>/`

## Next direction

GPU 启动后盯 `how_pending` / plugin smoke。若长期 0 草稿，记能力边界后停，再议是否升 B。

## Amendment �� deferred GPU (2026-08-29)

�û����ָʾ���Ȳ���ʼʵ�项���� stop ����ֹ GPU ������ս�۱���Ϊ paused / deferred���� resume ֱ������ȷ�ϡ�


## Amendment �� execute A (2026-08-29)

�û�ȷ�ϡ���ִ��A�����¿� campaign `exp_rgbt_dfine_v26_lowlight_20260828T231852Z`��`max_extra_rounds=5`��lifecycle on��invent off������ǰ deferred `...T183033Z` ���� paused���� resume��

