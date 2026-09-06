# v2.5-D Human-Gated REAL Loop（post-MVP，比赛版闭环）

不解冻 MVP。tag `mvp-freeze-m1-m4-claimgate-c1` @ `130b02cfbb5521829e959d10b99d17fd5fff28ab` 仍只读。
本文件描述 **feat/post-mvp-v25** 上把 LLM 决策送进 Scientist Lab **真执行路径** 的验收口径。不是 freeze 的一部分。

## 这一次 Human Gate 的范围

用户说「下一步」= 许可 **这一次 probe REAL**（fast_eval 16/8 子集）。

- **许可**：probe 预算、已有 Adapter HOW、Gate APPROVED 之后的 `--execute`。
- **仍禁止自动**：`budget_class=formal`、4h formal、SOTA、发明 FDPN、改 D-FINE 源码、绕过 Gate、把 probe 结果写成 C1。
- GPU 与 LLM live **独立**：无 `LLM_API_KEY` 时用 mock LLM；有 GPU 才 `--execute --require-live-ready`。

## 闭环

```text
历史 VALID Evidence（m4_rounds3 Round1 DISCARD）
  → LLM Reviewer 语义提案（不覆盖 Rubric）
  → MemoryWriter（必须 evidence_refs）
  → LLM Planner → Structured Plan
  → 现有 Gate（不可绕过）
  → Human Gate（本句「下一步」仅覆盖 probe REAL）
  → Adapter HOW（selected 模块的已有 HOW：fusion → rgbt + early_concat）
  → --execute --require-live-ready 真 GPU（doctor false 则拒绝并 exit 非 0，不伪造 metrics）
  → 新 Evidence
  → LLM Reviewer 再解释
```

默认 `manager-run` 仍是 `planner-backend=rules` / `reviewer-backend=rules`。本环的 CLI 是 `llm-real-loop`，默认 llm backend，但仍走同一 Manager / Gate / Adapter。

## 开关

```text
# freeze 安全（默认）
scientist-lab manager-run --protocol ... --plan ... --output-dir ...

# 显式打开 LLM 认知后端，仍默认 dry-run
scientist-lab manager-run --planner-backend llm --reviewer-backend llm ...

# 比赛版闭环 dry-run（无 GPU、无 key）
scientist-lab llm-real-loop --run-dir tests/fixtures/llm_plan_replay/m4_rounds3_discard

# 有 GPU 的 probe REAL（Human Gate 已由 --execute 表达；formal 仍禁止）
scientist-lab llm-real-loop --execute --require-live-ready --output-dir .run/v25d_llm_real_loop

# 有 LLM key（仍不自动 GPU）
scientist-lab llm-real-loop --live
```

环境变量：`SCIENTIST_LAB_PLANNER_BACKEND` / `SCIENTIST_LAB_REVIEWER_BACKEND`。
LLM live：`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`。无 key 的 `--live` fail closed，不得假装成功。

## probe ≠ formal / ≠ C1

- 本环使用现有 fast_eval 子集（16 train / 8 val）。这是 **工程闭环证据**，不是科学 C1。
- ClaimGate 对 probe + 非配对 fingerprint 必须是 BLOCKED / INCONCLUSIVE / PARTIALLY_SUPPORTED，**不得** SUPPORTED C1。
- APS 涨或跌都可以。负结果 + Reviewer 反思算闭环成功叙事，不算模块无效声称。
- 超时按 probe（约 1200s），不要 4h formal。

## 验收清单

1. Manager 可显式 `planner_backend=llm` + `reviewer_backend=llm`；默认仍 rules；freeze 测试不破。
2. 无 Gate APPROVED 不得点火（live_runner 不被调用）。
3. Rubric / ClaimGate 不被 LLM 覆盖。
4. `require-live-ready` 且 doctor false → 禁止伪造 metrics，exit 非 0。
5. 无 key：mock 路径必须能 dry-run 整环（Gate → skip live）。
6. 产物目录 `.run/v25d_llm_real_loop/`（`.gitignore` 已忽略 `.run/`）。

## 比赛版是否可停在 D

可以停。D 证明「LLM 决策能进入 Scientist Lab 真路径且 Gate/ClaimGate 仍在」。它 **不** 证明 fusion 有效，也不是 C1/SOTA。下一步若做 E，应是独立的 formal 配对实验，而不是把本 probe 升格。
