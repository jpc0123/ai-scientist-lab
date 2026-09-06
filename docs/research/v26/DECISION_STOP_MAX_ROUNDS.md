# Decision — STOP V26.5 GPU；等 Human Gate

日期：2026-08-20  
战役：v2.6 / V26.5  
裁决：`stop` + `request_user_decision`  
未改 Protocol。未 commit / push。未宣称 G2。文献未进 ClaimGate。

## Verdict

**STOP。** 用户确认 Semantic Scholar 已连通后，本回合核对磁盘协议与 pack：没有可在现协议内点火的下一轮 GPU。

## Action

`request_user_decision`：二选一（也可明确拒绝两者）。**不要**同时 silently 改 `max_rounds` 并注册新 detector。

## Reason

1. **R5 GPU 需要 Protocol Amendment。** Pack `outputs/v26_r5/` 已有 F3 seed 45 Next Plan（HOW 已对齐），但 Manager 在 Gate/GPU 前硬停：`round_index=5 >= stop_rules.max_rounds=5`。`run_state=STOPPED`。无 `APS_lowlight`。`metrics_forged=false`。把 `max_rounds` 从 5 提到 ≥6 才能跑 R5；本回合**没有**改这个字段。
2. **P3 GPU 轮次已用尽下限。** 已落地 Live LLM + GPU：R1–R4（4 轮）。P3 目标是 3～5 轮；协议硬上限拦住第 5 个 `round_index`。再跑 R5/R6 不是 in-scope Next Plan，是修宪法。
3. **P4 不能直接开 GPU。** 战役文档把 P4（RT-DETR strategy transfer）列为 P3 之后的工程步，但仓库里 **只有** `DFINEAdapter`。没有已注册的 RT-DETR Adapter、没有 transfer Protocol、没有 RT-DETR 上的 F3 HOW 翻译。开 P4 = 新 detector 注册 + 新协议（很可能还要新的 `max_rounds`）。这不是现有 D-FINE HOW 菜单能静默执行的。
4. **文献 live 不能绕过 stop_rules。** 本回合 live 检索成功：`provider=semantic_scholar`，`literature_query_id=litq_687d9c49f056`，`can_enter_claim_gate=false`。LiteratureEvidence ≠ ExperimentEvidence。它不能把 R5 从 STOPPED 改成 APPROVED，也不能把 ClaimGate 从 C0 BLOCKED 改成 G2。

## Rejected alternatives

- 自行把 `max_rounds` 改为 6 再跑 R5 F3 seed 45
- 把 R5 的 `round_index` 改成 4 来骗过 stop_rules
- 未注册 RT-DETR Adapter 就宣称开始 P4 GPU
- 把 live 文献摘要写进 ClaimGate / 宣称 G2
- 解冻 A4 / 注册 F2/T1/T2 / 改切片
- 把 ClaimGate `max_claim_strength` 从 C0 升到能对外说 G2

## Evidence paths

- `schemas/examples/research_protocol_rgbt_dfine_v26.json` → `stop_rules.max_rounds=5`
- `outputs/v26_r5/experiment_run.json` → `round_index=5`，`run_state=STOPPED`
- `outputs/v26_r5/next_plan_note.json` → `stopped_before_gpu=true`，HOW=F3，seeds=`[45]`
- `outputs/v26_r4/claim_gate.json` → `status=BLOCKED`，`max_claim_strength=C0`
- `src/scientist_lab/adapters/__init__.py` → 仅 `DFINEAdapter`
- `outputs/v26_literature_live/litq_687d9c49f056.json` → live Semantic Scholar probe（非实验证据）

## 已冻结的 GPU 事实（不重跑、不改数字）

| Pack | HOW | seed | APS_lowlight | Rubric | ClaimGate |
|------|-----|------|--------------|--------|-----------|
| R0 | F1 | 42 | 0.0045926865160844455 | — | — |
| R1 | F3 | 42 | 0.02135704762627717 | KEEP | BLOCKED/C0 |
| R2 | F0 | 42 | 1.3452432199741713e-06 | KEEP/INCONCLUSIVE | BLOCKED/C0 |
| R3 | F3 | 43 | 0.04881493259150456 | KEEP | BLOCKED/C0 |
| R4 | F3 | 44 | 0.04204268629433699 | KEEP | BLOCKED/C0 |
| R5 | F3 | 45 | （无；STOP before GPU） | — | — |

三 F3 seed 同方向高于 R0，是可判定 G2 的**证据门槛**，**不是** G2 成功声明。KEEP ≠ Claim。

## Human Gate（只需回答这一问）

Semantic Scholar 已 live。V26.5 GPU 被 `max_rounds=5` 挡住。请选 **A** 或 **B**（或明确拒绝两者）：

**A.** 批准 Protocol Amendment：把 `stop_rules.max_rounds` 从 5 提到 ≥6，然后在 **live 文献 + live LLM** 下跑 R5（F3 seed 45，`outputs/v26_r5`）。ClaimGate 仍 C0，不宣称 G2。

**B.** 不改 `max_rounds`，批准开启 **P4**：先注册 RT-DETR Adapter + 写 transfer Protocol（同一 `dataset_id`/`slice_id`/fingerprint 家族；HOW 必须在新 Adapter 上重实现，不得把 D-FINE 模块塞进第二模型），再跑 RT-DETR baseline 与迁移实验。

## Next direction

**已关闭（2026-08-20）：** 用户选 **A**。见 [`DECISION_HUMAN_GATE_A.md`](./DECISION_HUMAN_GATE_A.md)。`max_rounds` 仅 5→6。P4 仍推迟。
