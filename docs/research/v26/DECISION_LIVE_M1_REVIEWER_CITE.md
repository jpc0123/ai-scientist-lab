# Decision — Live M1 Reviewer may cite executed N1/FDPN

日期：2026-08-28  
战役：v2.6 / Live M1  
裁决：`continue`（合同修补，不翻发明开关）  
Human Gate：本决议不自动 resume GPU。KEEP ≠ Claim。

## Verdict

**先按 Live M1 走。不翻 `llm_may_invent_how`。**

Reviewer 可以在语义提案里**引用已执行计划中的 N1 / A4 / FDPN**（观察，不是发明）。  
仍禁止：write-Python、new network、custom operator、在未执行 N1/A4/fdpn 时引入 FDPN、提出新的 fusion_method。

## Action

`continue`：修补 `reviewer_contract` 文本闸门；补单测。不 resume 失败 campaign，除非人另发指令。

## Reason

Live M1 放开的是「从证据设计下一轮 + verification_plan」，不是自由发明。  
上一场 `exp_rgbt_dfine_v26_lowlight_20260827T144442Z` 已合法跑完 N1，Reviewer 解释里写 `FDPN` 被 `\bfdpn\b` 误杀，与提示「可引用已执行 HOW」自相矛盾。修的是审阅引用，不是解冻 Planner 发明。

## Role lock

| 这是 | 这不是 |
|------|--------|
| Reviewer 引用已执行 N1/FDPN | 翻 `llm_may_invent_how` |
| 仍禁 write-Python / 新网络 | Planner JSON 夹代码 |
| Live M1 合同一致 | 把 A4 包装成 LLM 发现 |
| fail_closed 仅拦真发明 | 自动 Claim / G2 |

## Rejected alternatives

- 完全放开自由发明（翻 `llm_may_invent_how`）
- 删除 Reviewer 全部发明禁词
- 静默 resume 失败 campaign

## Evidence paths

- `src/scientist_lab/llm/reviewer_contract.py`
- `tests/unit/test_llm_reviewer_v25c.py`（`test_reviewer_may_cite_executed_n1_fdpn_live_m1`）
- 失败现场：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260827T144442Z/`
- 本文件

## Amendment — KEEP status coerce (2026-08-28)

Live M1：`KEEP` + `hypothesis_status=inconclusive_budget|needs_replication|needs_validation`
软纠正为 `not_a_claim`，不再 fail_closed。不扩大 Claim 语义；不翻发明开关。

## Next direction

人确认后再 `resume` 该 campaign，或开新 Live M1 轮。仍走目录 HOW + 插件沙箱，不发明算子。
