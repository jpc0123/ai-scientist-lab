# Decision — Live B: fix E8 schema/harvest, complete legal P3 multi-seed

Date: 2026-08-31  
Prior: `DECISION_LIVE_B_LONGTRAIN_E8.md`；campaign `…20260831T081633Z`  
裁决：`continue`（同战役 resume；不 invent）  
Human Gate：用户「修执行/schema → 只补合法 P3 多 seed；开始」

## Verdict

**修 plan `expected_effect` schema 清洗 + harvest HOW/epochs 一致性校验；剔除无效 r3；续跑只补合法 P3@8ep 多 seed（对照已有 F1@8ep）。KEEP ≠ Claim。**

## Why

1. 战役死于 `SchemaValidationError: expected_effect.reference_last*`（LLM 多写字段，`additionalProperties:false`）。  
2. r3（标称 P3 seed42）**无效**：plan/materialization 为 `plugin:P3` / epochs=8，但产物为 `fusion=none` / `epochs=2`（伪 harvest）。  
3. **唯一合法 P3@8ep**：r4 seed43，`APS_lowlight≈0.0737`。  
4. F1@8ep 对照已齐：0.0215 / 0.0110 / 0.0113（seeds 42/43/44）。

## Action

1. 代码：`sanitize_expected_effect`；harvest 用 artifact knobs（fusion/epochs）拒伪匹配；`SchemaValidationError` 可 resume。  
2. 续跑 `…081633Z`：steer 重做 **P3 seed42@8ep**，再 **seed44**（凑 ≥3，含已有 seed43）；禁止 invent / 空转 F1。  
3. 不 resume 2ep 旧战役；不把 r3 计入 P3 均值。

## Role lock

| 这是 | 这不是 |
|------|--------|
| 执行正确性 + P3 合法多 seed | 新 invent / 换库 |
| F1@8ep 已冻结对照 | ClaimGate / G2 |
| r3 作废重跑 | 用 0.0024 证伪 P3 |

## Evidence paths

- Campaign：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260831T081633Z/`  
- 合法 P3：`outputs/.../exec_4a1959fd827f/`（plugin:P3, epochs=8）  
- 无效 r3：`outputs/.../exec_03ff72243d35/`（none, epochs=2）  
- Adapter knobs：`src/scientist_lab/adapters/dfine/how.py`（epochs=8）
