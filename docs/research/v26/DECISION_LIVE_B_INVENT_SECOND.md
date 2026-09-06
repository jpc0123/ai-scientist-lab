# Decision — Live B: close E8 arm + optional second invent proof

Date: 2026-09-01  
Prior: `PROOF_LIVE_B_P3_INVENT_CAPABILITY.md`，`DECISION_LIVE_B_P3_E8_RESUME.md`  
裁决：`stop`（E8 补种臂）+ **已执行** `continue`（新 invent 战役 → P4 证明完成）  
收口：见 `DECISION_LIVE_B_DUAL_INVENT_CLOSEOUT.md`（**两份 invent 已立；勿 resume**）  
Human Gate：用户确认「写证明包；第二份 invent = 新战役 ≠P3；勿 resume E8」

## Verdict

1. **E8 战役 `…20260831T081633Z`：归档停机，不要 resume。**  
   合法 P3@8ep（n=5）vs F1@8ep（n=3）已写入证明包；终态 PermissionError 不阻碍证据。  
2. **第一份 invent 能力证明以 P3 收口（KEEP ≠ Claim）。**  
3. **若做第二份 invent 证明：必须新开战役**，`llm_may_invent_how=true`，steer 要求 **新机制 ≠ P3 thermal spatial gate**；GPU **可后置**（先 invent→author→smoke）。

## Why

- 再刷目录 HOW / 续跑 E8 对 invent 能力无增量。  
- 第二份证明的科学问题是「可重复 invent」，不是再抬 P3 APS。  
- Resume E8 会把 invent 目标与已污染的 materialize/锁文件历史缠在一起。

## Action

### A. E8 closeout（立即）

- 不调用 resume/extend `exp_rgbt_dfine_v26_lowlight_20260831T081633Z`。  
- 证据引用以 `PROOF_LIVE_B_P3_INVENT_CAPABILITY.md` 为准。  
- r3（none/2ep）保持作废。

### B. 第二份 invent（仅当用户要开时）

前端 `/loop` 新战役 Start（**不要续跑 E8**）：

1. 实验：Low-light RGB-T D-FINE v2.6  
2. 勾选协议 Human Gate  
3. 插件作者：**LLM**  
4. 需要 invent：启动参数 / 后端需带 `llm_may_invent_how=true`（及 lifecycle）；**前端若默认不 invent，Start 后由助手翻开关 + steer**  
5. Steer（建议原文）：

```
INVENT-SECOND: Propose a NEW fusion HOW mechanism distinct from P3
(ThermalSpatialAttentionFusion / thermal HxW spatial gate). Prefer a
different operator family (e.g. cross-modal gating not spatial mask,
frequency/channel interaction, or late fusion head — still FeatureFusion
plugin only). invent → human review → LLM author → smoke. Do NOT re-author
P3. Do NOT idle F1/F3/A4. GPU optional after smoke_ok. KEEP!=Claim.
is_claim=false.
```

成功指标（最小）：

- ≥1 条 invent 草稿：人审 → `plugin_authored_by=llm` → smoke_ok → overlay  
- 机制文本 / 代码与 P3 可区分  
- **不要求**立刻超过 P3 APS  

可选加强：smoke 后 1–2 seed @8ep vs F1 mean（后置）。

## Role lock

| 这是 | 这不是 |
|------|--------|
| 第二份 invent 可重复性 | Resume E8 补 seed |
| 新机制 ≠ P3 | 再训 P3 / 刷 catalog |
| smoke 优先，GPU 后置 | 自动 Claim / G2 |

## Rejected

- Resume `…081633Z` 开 invent  
- 用 P5/human_file 冒充第二份 invent  
- 把证明包升格为 ClaimGate SUPPORTED  

## Evidence paths

- Proof：`docs/research/v26/PROOF_LIVE_B_P3_INVENT_CAPABILITY.md`  
- E8 archive：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260831T081633Z/`  
- Invent-1：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260830T171900Z/` + `how_plugins/P3/`
