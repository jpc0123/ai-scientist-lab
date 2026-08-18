# v2.5-C LLM Reviewer + Semantic Memory（post-MVP）

不解冻 MVP。tag `mvp-freeze-m1-m4-claimgate-c1` @ `130b02cfbb5521829e959d10b99d17fd5fff28ab` 仍只读。
本文件描述 **feat/post-mvp-v25** 上的 Reviewer 认知后端。不是 freeze 的一部分，也不是第二个物理模型。

## 与 MVP Freeze 的关系

- Agents 仍只有 Manager + Planner + Executor + Reviewer。
- **LLM Gateway 不是第五个 Agent**。它是 Planner / Reviewer 共用的认知后端（role prompt 不同）。
- **DecisionRubric 仍是唯一 KEEP/DISCARD/REPLICATE 来源。** LLM 不得覆盖 `review_decision`。
- **ClaimGate 仍是声称闸。** LLM 不得把 DISCARD 写成「模块无效」的正式声称，也不得把 KEEP 写成 SUPPORTED。
- **MemoryWriter 仍是唯一写 Memory 的组件。** LLM 只提案；Writer 校验 `evidence_refs` 后才写入。
- EventAppender 记事实：`model` / `prompt_hash` / `raw`（密钥已脱敏）。
- Reviewer = 科研语义分析（为什么、替代解释、下一轮该验证什么），不是 HOW。
- 默认 `reviewer.backend=rules`，与 freeze `test_reviewer_replay_m2.py` 同一条规则 Reviewer。
- 不改 D-FINE 源码，不发明 FDPN，不执行 GPU，不做 v2.5-D 真闭环点火。

## 开关方式

1. **默认（freeze 安全）**：`Reviewer()` → `rules`
2. **配置**：`config/scientist-lab.yaml` 中 `reviewer.backend: rules`（本阶段文档开关；Manager 默认仍构造规则 Reviewer）
3. **显式 LLM**：
   - `Reviewer(backend="llm")`
   - 环境变量 `SCIENTIST_LAB_REVIEWER_BACKEND=llm`
4. **真实 API**：环境变量 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`，并且 CLI 加 `--live`。密钥不进 git。无 key 时 fail closed，不得假装 live 成功。
5. **回退规则 Reviewer**：默认关闭。仅当 `fallback_to_rules=True`（或 CLI `--fallback-to-rules`）时，fail-closed 后保留 Rubric packet。禁止静默回退。

## 输入 / 输出

输入：Evidence（必须 VALID）+ Plan + Rubric decision + Protocol + metrics。主指标是 Protocol `objective.primary`（本协议为 **APS**）。**不得用 mAP 冒充 APS。**

输出是提案，不是 decision：

```text
observation, hypothesis_status, interpretation, alternative_explanations,
next_research_priority, evidence_refs, created_from, confidence
```

`hypothesis_status` 对齐：

| Rubric `review_decision` | 允许的语义 `hypothesis_status` |
| --- | --- |
| DISCARD | `not_supported_under_current_protocol` 或 `inconclusive_budget` |
| KEEP | `not_a_claim`（KEEP ≠ claim supported） |
| REPLICATE | `needs_replication` 或 `inconclusive_budget` |
| VALIDATE | `needs_validation` 或 `not_a_claim` |

文档 `review_decision.schema.json` 里的 `hypothesis_status`（`REJECTED` / `SUPPORTED` / …）仍由 Rubric 路径写入，LLM **不得**把该枚举当作声称结果覆盖上去。

语义 Lesson 必须 evidence-linked（`run_id` + `evidence_refs`）。无 VALID 不得写「模块有效/无效」科学结论。无 `evidence_refs` 时 MemoryWriter **拒写**。

## Fail closed

下列情况 **不覆盖 Rubric、不写源包 Memory、不静默回退 rules**：

- 非法 JSON / 缺字段
- 输出 `review_decision` / `claim_gate` / Memory 写入键
- `hypothesis_status` 抄文档枚举（`SUPPORTED` / `REJECTED` …）或与锁定 decision 不对齐
- 把 DISCARD 写成模块无效、把 KEEP 写成 SUPPORTED
- 用 mAP 冒充 APS
- 发明算子 / FDPN / 输出 HOW（`fusion_method` / `early_concat`）
- `--live` 但没有 `LLM_API_KEY`

## REPLAY（不点火）

同一历史包：`tests/fixtures/llm_plan_replay/m4_rounds3_discard`（Round1 neck DISCARD，APS=0.0 vs labeled synthetic_control 0.6）。

不改源包。产物写到 `<run-dir>/.llm_review_replay/`。

```text
scientist-lab llm-review-replay --run-dir tests/fixtures/llm_plan_replay/m4_rounds3_discard
```

有 key 时仍不 GPU：

```text
scientist-lab llm-review-replay --run-dir tests/fixtures/llm_plan_replay/m4_rounds3_discard --live
```

无 `LLM_API_KEY` 时 `--live` fail closed，exit 1，不得假装成功。

## 与 Rubric / ClaimGate 的边界

- Rubric 先跑。`packet.review_decision` 与 `document.review_decision` 保持 Rubric 结果（本包为 DISCARD）。
- `document.hypothesis_status` 保持 freeze 枚举（本包 DISCARD → `REJECTED`）。
- LLM `semantic_proposal.hypothesis_status` 是语义标签，不是声称闸。
- ClaimGate 仍独立评价；KEEP ≠ SUPPORTED；DISCARD ≠ 模块无效。
- `ReviewPacket.to_memory_review()` 仍只交出 Rubric lessons/strategies。语义 lesson 必须显式走 `MemoryWriter.persist_semantic_proposal`。

## 下一步 v2.5-D（已实现，见 `docs/V25D_REAL_LOOP.md`）

Human-gated probe REAL loop。Manager 可显式 `planner_backend=llm` + `reviewer_backend=llm`，默认仍 rules。`llm-real-loop` 把历史 DISCARD → Reviewer 提案 → MemoryWriter → Planner → Gate → 可选 GPU。probe ≠ C1。formal 仍禁止自动升。
