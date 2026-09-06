# Decision — Live B: dual invent closeout（两份 invent 已立）

Date: 2026-09-01  
Prior: `PROOF_LIVE_B_P3_INVENT_CAPABILITY.md`，`PROOF_LIVE_B_P4_INVENT_CAPABILITY.md`，`DECISION_LIVE_B_INVENT_SECOND.md`  
裁决：`stop` / **archive**（E8 + invent-second）  
Human Gate：用户确认「写 P4 证明包；E8 / invent-second 都不要 resume；写清两份 invent 已立」

## Verdict

**Live B invent 能力线收口：两份独立 LLM invent 证明已立（P3 + P4）。E8 与 invent-second 战役归档，禁止 resume。KEEP ≠ Claim。**

| # | HOW | 链路 | GPU 对照包 | Campaign |
|---|-----|------|------------|----------|
| 1 | **P3** `ThermalSpatialAttentionFusion` | invent→llm author→smoke | E8 vs F1 n=3 / P3 n=5 | invent `…171900Z` + E8 `…081633Z` |
| 2 | **P4** `SpatialConfidenceFusion` | invent→llm author→smoke | 本场 vs F1 n=1 / P4 n=6 | invent-second `…20260901T025804Z` |

机制可区分：P3 = 热引导 → concat+reduce；P4 = 双向门控 → sum(+residual)。

## Why

- 再 invent / 再 resume 对「系统能 invent」无增量信息。  
- 指标增益仅支持 KEEP；升 Claim 需要新协议级对照，不是续跑脏战役。

## Action（立即）

1. **勿 resume / extend**  
   - `exp_rgbt_dfine_v26_lowlight_20260831T081633Z`（E8）  
   - `exp_rgbt_dfine_v26_lowlight_20260901T025804Z`（invent-second）  
2. 证据以两份 PROOF 文档为准；磁盘战役标 `*_CLOSED` / `closed_reason`。  
3. 默认下一方向（若用户再开）：对齐多 seed 科学对照 **或** 换问题 Amendment —— **不是** invent P11。

## Role lock

| 这是 | 这不是 |
|------|--------|
| 双 invent 能力 closeout | Claim / G2 / 论文主结论 |
| 归档停机 | Resume E8 或 invent-second |
| P3+P4 KEEP 证据索引 | 把跨战役 APS 拼成单一 Claim |

## Rejected

- Resume 任一已归档战役开新 invent  
- 用 P5/human_file 冒充第三份 invent  
- 静默把 KEEP 升格 ClaimGate  

## Evidence paths

- `docs/research/v26/PROOF_LIVE_B_P3_INVENT_CAPABILITY.md`  
- `docs/research/v26/PROOF_LIVE_B_P4_INVENT_CAPABILITY.md`  
- `.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260831T081633Z/`  
- `.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260901T025804Z/`  
- Plugins：`how_plugins/P3/`，`how_plugins/P4/`
