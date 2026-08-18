# v2.5-A LLM Planner Gateway（post-MVP）

不解冻 MVP。tag `mvp-freeze-m1-m4-claimgate-c1` @ `130b02cfbb5521829e959d10b99d17fd5fff28ab` 仍只读。
本文件描述 **feat/post-mvp-v25** 上的 LLM 认知后端，不是 freeze 的一部分。

## 与 MVP Freeze 的关系

- Agents 仍只有 Manager + Planner + Executor + Reviewer。
- **LLM Gateway 不是第五个 Agent**。它是 Planner（随后 Reviewer）的认知后端。
- Planner = WHAT/WHY；Adapter = HOW；Gate / ClaimGate / DecisionRubric 不可被 LLM 覆盖。
- MemoryWriter 仍是唯一写 Memory 的人。LLM 只提案。
- 默认 `planner.backend=rules`，与 freeze 测试同一条规则 Planner。显式打开 `llm` 才走 Gateway。
- 不改 D-FINE 源码，不发明 FDPN，不执行 GPU。

## 开关方式

1. **默认（freeze 安全）**：`Planner()` / `propose_and_gate_next()` → `rules`
2. **配置**：`config/scientist-lab.yaml` 中 `planner.backend: rules`（本阶段文档开关；Manager 默认仍构造规则 Planner）
3. **显式 LLM**：
   - `Planner(backend="llm")`
   - `propose_and_gate_next(backend="llm")`
   - 环境变量 `SCIENTIST_LAB_PLANNER_BACKEND=llm`
4. **真实 API**：环境变量 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`，并且 CLI 加 `--live`。密钥不进 git。
5. **回退规则 Planner**：默认关闭。仅当 `fallback_to_rules=True`（或 CLI `--fallback-to-rules`）时，fail-closed 后记 event 再走 rules。禁止静默回退。

## Fail closed

下列情况 **不 APPROVED、不执行、不静默回退 rules**：

- 非法 JSON / 缺字段
- `requested_module` 不在 Protocol `editable_scope`，或不在 Adapter HOW（仅 `neck`→rgb+none，`fusion`→rgbt+early_concat）
- 发明算子 / FDPN / 要求写 Python
- 自行把 `budget_class` 升到 `formal`
- 写 Memory、伪造 lesson/strategy id
- 覆盖 KEEP/DISCARD/REPLICATE 或 ClaimGate

多候选放在 Plan 附件 `candidate_experiments`；**只有 selected 那一份**进 Gate。`llm_trace` 记录 provider/model/`prompt_hash`/raw（密钥已脱敏）。

## REPLAY（不点火）

历史 DISCARD stub：`tests/fixtures/llm_plan_replay/discard_stub`

REPLAY 把 Memory 复制到 `<run-dir>/.llm_plan_replay/`，不改写源包的 `memory/`。

```text
scientist-lab llm-plan-replay --run-dir tests/fixtures/llm_plan_replay/discard_stub
```

有 key 时可选一次真 API（仍不 GPU）：

```text
scientist-lab llm-plan-replay --run-dir tests/fixtures/llm_plan_replay/discard_stub --live
```

`--live` 需要 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。无 key 时 fail closed，exit 1。

也可指向 Manager 历史 run-dir（需含 `protocol.json`、`plan.json` 或 `previous_plan.json`、`memory/`）。

## v2.5-B 历史 REPLAY 考试

不是再调一次 API，而是用 **真实冻结 Evidence** 考 Planner 是否理解 DISCARD。

Evidence 包：`.run/real_fast_eval_m4_rounds3` 的 **Round1 DISCARD**（exec `exec_ef974841f0b8`，GPU APS=0.0 vs labeled synthetic_control 0.6，delta=-0.6）。精简副本（无 metrics 大文件、不入库 `.run/`）：

`tests/fixtures/llm_plan_replay/m4_rounds3_discard/`

- 真实 lesson：`LESSON-run_plan_round1_neck_hr-001`（negative_evidence，module=neck）
- 真实 strategy：`STRATEGY-run_plan_round1_neck_hr-001`（deprioritize neck）
- Previous Plan：`plan_round1_neck_hr`（modification_scope=neck）
- Rubric：`review_decision=DISCARD`

验收：

1. mock LLM 理解 DISCARD，selected **不是** neck
2. `memory_refs` 引用真实 lesson id
3. 已 DISCARD 的模块可以出现在 `candidate_experiments`，但必须带 `reason_not_selected`；不得作为 selected
4. `candidate_experiments` ≥ 2；selected 通过现有 `experiment_plan` schema
5. 合法 fusion HOW（rgbt+early_concat）Gate **APPROVED**；越权/FDPN fail closed
6. A/B：同一包上规则 Planner vs LLM。规则版是死选项翻转（历史上 Round3 又选回 neck）；LLM 必须有 `decision_summary` / 选择依据。对比写入 `replay_report.json`，**不编造 live API 结果**
7. 不写源包 Memory、不改 `review_decision`、不升 formal、不点火

```text
scientist-lab llm-plan-replay --ab --run-dir tests/fixtures/llm_plan_replay/m4_rounds3_discard
```

报告默认：`<run-dir>/.llm_plan_replay/replay_report.json`（含 `rules_plan` / `llm_plan` / `gate` / `memory_refs` / `discarded_module` / `ab`）。

有 key 时仍不 GPU：

```text
scientist-lab llm-plan-replay --ab --run-dir tests/fixtures/llm_plan_replay/m4_rounds3_discard --live
```

无 `LLM_API_KEY` 时 `--live` fail closed，exit 1，不得假装成功。

## 复用的现有网关

不另起第二套 Provider。`scientist_lab.llm.gateway` 是现有 FakeProvider / OpenAICompatibleProvider / factory 的门面。v1.3 `PLANNER_OUTPUT_SCHEMA`（Workbench 候选树）保持不变；Freeze Planner 使用 `PLANNER_CONTRACT_SCHEMA` → 现有 `experiment_plan` schema。

## 下一步 v2.5-C（已实现，见 `docs/V25_LLM_REVIEWER.md`）

同一 Gateway，Reviewer role prompt。DecisionRubric 仍出 KEEP/DISCARD/REPLICATE。LLM 不得覆盖 Rubric 或 ClaimGate。MemoryWriter 仍是唯一写入者。

```text
scientist-lab llm-review-replay --run-dir tests/fixtures/llm_plan_replay/m4_rounds3_discard
```

## 下一步 v2.5-D（已实现，见 `docs/V25D_REAL_LOOP.md`）

Human-gated probe REAL loop。`manager-run --planner-backend llm --reviewer-backend llm` 默认仍 dry-run。真 GPU 用 `llm-real-loop --execute --require-live-ready`。probe ≠ C1。
