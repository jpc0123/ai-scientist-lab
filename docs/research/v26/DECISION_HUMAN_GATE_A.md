# Decision — Human Gate A：Protocol Amendment `max_rounds` 5→6

日期：2026-08-20  
战役：v2.6 / V26.5  
裁决：`continue` + `launch_experiment`  
用户选择：**A**（不是 B）。未 commit / push。未宣称 G2。文献不得进 ClaimGate。P4 仍推迟。

## Verdict

**Human Gate A 已执行。** 仅把 `stop_rules.max_rounds` 从 **5 改为 6**，以便 `round_index=5`（pack `outputs/v26_r5`，F3 `gated_multiscale` seed 45）能过 stop_rules 并上 GPU。

## Action

1. Protocol Amendment：`max_rounds` **只** 5→6。
2. 续跑已命名的 R5 pack（HOW=F3，seed=45），live Semantic Scholar + live LLM Reviewer。
3. ClaimGate 仍为 C0 / BLOCKED。KEEP ≠ Claim。即使 `APS_lowlight` 高于 R0/R1/R3/R4，**不宣称 G2**。
4. **P4 / RT-DETR 本回合不启动。**

## Reason

Planner 已选出 F3 seed 45，但 R5 从未 GPU 跑过，没有 `APS_lowlight`。证据环在 stop_rules 处断开。关闭该环只需要允许 `round_index=5`（`5 >= 6` 为假）。不需要跳到第二 detector。

## 本回合未改（冻结）

- `max_claim_strength`（仍由 `allow_scientific_claims=false` 落到 C0）
- 切片 `low_light_subset_v1` / `dataset_id=rgbt_tiny_v1` / fingerprint `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6`
- HOW 目录（F3 已注册；不是 A4；不是 F2/T1/T2）
- A4 对 Planner 隐藏
- 不注册 RT-DETR Adapter

## Rejected alternatives

- **B**：开 P4 / 注册 RT-DETR Adapter（用户未选；本回合明确推迟）
- 把 `max_rounds` 提到 >6
- 把 R5 的 `round_index` 改成 4 骗过 stop_rules
- 静默改切片 / HOW / A4 / claim strength
- 用文献把 STOPPED 改成声称 G2

## Evidence paths

- `schemas/examples/research_protocol_rgbt_dfine_v26.json` → `stop_rules.max_rounds=6`（Human Gate A）
- `outputs/v26_r5/protocol.json` → 同字段
- `outputs/v26_r5/plan_round5.json` → HOW=F3，`evaluation.seeds=[45]`
- `outputs/v26_r5/seed_contract.json` → aligned
- 前序 STOP：[`DECISION_STOP_MAX_ROUNDS.md`](./DECISION_STOP_MAX_ROUNDS.md)

## Next direction

R5 GPU → Reviewer + ClaimGate（期望 C0/BLOCKED）→ **STOP**。P4 等用户在 R5 证据之后另开。

## Status（2026-08-21）

已完成。`outputs/v26_r5/` GPU `exec_b91239a81097`，`APS_lowlight=0.035942673873977496`，`semantic_review.json` `not_a_claim`，ClaimGate **BLOCKED/C0**。未宣称 G2。**P4 仍推迟**。
