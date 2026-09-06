# Decision — Live B: unlock invent HOW (post-A)

Date: 2026-08-30  
Prior: `DECISION_LIVE_A_EVIDENCE_GAP_CLOSEOUT.md`, `DECISION_LIVE_M1_CLOSEOUT.md`  
裁决：`continue`（**新 campaign，阶段 B**）  
Human Gate：用户确认「开 B」

## Verdict

**进入 Stage B：翻 `llm_may_invent_how=true`。新开战役，不 resume Stage A。**

在冻结的 v2.6 协议副本上新开战役：允许 invent-fallback 提出新插件 HOW 草稿（人审 → Diff 写 `how_plugins/<id>/plugin.py` → smoke → overlay）。Planner **仍不得夹 Python**。文献/插件 **不得** 进 ClaimGate。KEEP ≠ Claim。不自动升 G2。

## Action

1. Stage A campaign `exp_rgbt_dfine_v26_lowlight_20260829T114407Z` 保持 `completed`，不 resume。  
2. 新 campaign 绑定同一实验 `exp_rgbt_dfine_v26_lowlight`：`llm_live=true`，`llm_how_lifecycle=true`，`llm_may_invent_how=true`，`execute=true`，`max_extra_rounds=8`，`plugin_worker=llm`。  
3. 协议副本打 `fingerprint_id` 后缀 `-LIVE-B`；`stop_rules.max_rounds≥12`。  
4. 成功指标：至少一条 invent-fallback（或文献）HOW 草稿进入人审/author/smoke；可选 GPU。不自动 Claim。

## Reason

Stage A 已证明插件沙箱与证据缺口补齐路径；目录 HOW 仍方差主导。用户显式选 B。比在 A 脏历史上静默续跑更干净。

## Role lock

| 这是 | 这不是 |
|------|--------|
| Campaign 级 invent gate + 插件 Diff | Plan JSON 夹 Python |
| 人审 invent 草稿后 author | 自动 Claim / G2 |
| 新 campaign + 新指纹 | resume Stage A |
| invent-first draft arm | merge 主树 / 改 vendor D-FINE |

## Rejected alternatives

- 在 Stage A completed 历史上 resume 发明  
- C：仅加预算复现（用户未选）  
- 静默翻开关不经 Human Gate  

## Evidence paths

- `docs/research/v26/DECISION_LIVE_A_EVIDENCE_GAP_CLOSEOUT.md`
- `docs/research/v26/DECISION_LIVE_A_HOW_PLUGIN.md`
- `src/scientist_lab/core/how_pending.py` (`set_llm_may_invent_how`)
- 新 campaign：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_<stamp>/`

## Next direction

盯 invent 草稿 → human release/register → smoke → overlay。若长期空转，记能力边界后停。
