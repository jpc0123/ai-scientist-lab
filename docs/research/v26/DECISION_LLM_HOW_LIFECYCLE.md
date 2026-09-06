# Decision — LLM HOW lifecycle inside a frozen Protocol

日期：2026-08-26  
战役：v2.6  
裁决：`continue` + Protocol Amendment（`max_rounds` 6→12，`protocol_version` 1→2）  
Human Gate：用户确认 **协议由人冻结；HOW 是否新增、代码是否接入、下一枪选哪个 HOW，战役内由 LLM 决定。** KEEP ≠ Claim。本文件不启动 GPU。

## Verdict

**比 overlay 自动注册更进一步，但仍不是解冻宪法。**

人只定一次大协议（切片、主指标、Gate、ClaimGate、stop_rules）。一次「开始自主实验」授权本战役预算。之后：

1. LLM 分析要不要 **加** 一个 HOW 草稿  
2. LLM 写 `how_plugins/<id>/plugin.py`（Diff 沙箱 + smoke）  
3. LLM 决定这段代码 **接不接** 进战役 `registered_overlay`  
4. LLM Planner **选择** 可物化 HOW  
5. Gate / Rubric / ClaimGate 照旧；长轮次看策略和 `APS_lowlight` 是否相对 R0 进化  

不把 `llm_may_invent_how` 翻成 true。Planner JSON 仍不得夹 Python。Vendor D-FINE / `train_dfine.py` 仍不可写。文献与插件不得进 ClaimGate。主树不 merge。

## Action

1. Protocol Amendment：`research_protocol_rgbt_dfine_v26` `stop_rules.max_rounds` **6→12**，`protocol_version` **1→2**。ClaimGate 仍 C0。  
2. 战役 `llm_how_lifecycle=true`（一次 Human Gate 覆盖）：NEED_HUMAN 因未物化 HOW 时，不等人点 register，走 LLM add → author → accept。  
3. `/loop` 默认 `max_extra_rounds=11`（最多 12 枪 GPU）。上限 11。不伪造指标。  
4. 本回合只接线 + 单测。**不打 GPU。**

## Reason

P0 两轮同 HOW 不能证明设计。A（smoke 后自动 overlay）仍把「接不接代码」收成人或规则。用户要验证的是：在冻结协议下，LLM 能否在长轮数里自己扩展方法空间并迭代；失败（拒接、smoke 不过、空转被拦）也是能力边界。

## Role lock

| 这是 | 这不是 |
|------|--------|
| 人冻宪法，LLM 管 HOW 生命周期 | 第五个 Agent / 文献 Agent |
| 战役 overlay 注册 | merge 主目录 / 改 vendor |
| Planner 选已物化 HOW | Planner 当场发明算子 |
| 长轮次能力验证 | G2 / 模块有效声称 |
| KEEP / DISCARD 是下一动作 | ClaimGate 升级 |

## Rejected alternatives

- 只做 A：smoke 自动进 overlay，人不再审，但 LLM 不决定接不接代码  
- 翻 `llm_may_invent_how`，让 Plan 夹 Python  
- 每轮仍要人点 register  
- 本回合直接开 12 轮 GPU  

## Evidence paths

- `src/scientist_lab/core/how_lifecycle.py`
- `src/scientist_lab/services/autonomous_campaign.py`
- `schemas/examples/research_protocol_rgbt_dfine_v26.json`
- 本文件

## Next direction

单测过线后，用户在 `/loop` 点一次开始，才是这场能力验证的 GPU。成功也不升 G2，除非另开多种子 Amendment。
