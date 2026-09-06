# Scientist Lab MVP · 答辩证据链（一页）

日期：2026-08-17  
本页是答辩现场索引，**不改数字、不升级声称**。完整封版索引仍以 `docs/MVP_DEFENSE_EVIDENCE.md` 为准。  
**.run/ 大文件不进 git**；现场打开本机磁盘即可。不要把 datasets / web / llm / v2.5 脏文件打进 freeze tag。

## 身份（先核）

| 项 | 值 |
|----|----|
| annotated tag | `mvp-freeze-m1-m4-claimgate-c1` |
| SHA | `130b02cfbb5521829e959d10b99d17fd5fff28ab` |
| 保护分支 | `freeze/mvp-m1-m4-claimgate-c1`（同 SHA；不得移动 tag） |
| Audit | `docs/ARCHITECTURE_TO_CODE_AUDIT.md` **PASS**（9/9） |
| 封版说明 | `docs/MVP_FREEZE.md`、`docs/MVP_RELEASE_NOTES.md` |

```text
git rev-parse mvp-freeze-m1-m4-claimgate-c1^{}
# 130b02cfbb5521829e959d10b99d17fd5fff28ab
```

## 产品故事（写死）

当前 MVP **不是**完整 LLM AI Scientist，也 **不是**更好的检测器。  
它是 **规则/状态驱动的可控自主实验系统**：人设目标 / 边界 / 预算，系统跑闭环，并用 ClaimGate 限制能说到哪一步。  
研发阶段科研认知由「人 + GPT」协作提供；下一代才把 Planner / Reviewer 经 LLM API 搬进系统。

## 两条闭环（git freeze 内源码）

1. **行为闭环** Gate → Run → Evidence → Rubric → Memory → Next Plan  
   `src/scientist_lab/core/manager.py` + GateEngine / ResultParser / EvidenceValidator / DecisionRubric / MemoryWriter / next_plan
2. **声称闭环** ClaimGate 限制 Evidence 最多支持的 Claim 强度  
   `src/scientist_lab/core/claim_gate.py`（确定性规则，不是 Agent，不覆盖 Reviewer）

KEEP ≠ Claim。probe ≠ formal。mAP ≠ APS。early_concat ≠ FDPN。不得 C2。  
`scientific_outcome` 是只读投影，可与 ClaimGate 并存，不是第二条声称状态机。

## Formal C1（数字不得改）

同一 Frozen Fingerprint；160×160 / 2 epoch **staging**，不是 640/20ep 论文协议。

| 臂 | HOW | APS | 磁盘（gitignore） |
|----|-----|-----|-------------------|
| Formal-01 | rgb + none | **0.0163** | `.run/formal_c1_aps_early_concat/baseline/` |
| Formal-02 | rgbt + early_concat | **0.0326** | `.run/formal_c1_aps_early_concat/candidate_resume/` |
| 配对 ClaimGate | `--baseline-run-dir` | **SUPPORTED** | `.run/formal_c1_aps_early_concat/claim_gate_c1.json` |
| Manager 单臂 | 无配对指纹 | **BLOCKED**（正确） | `candidate_resume/claim_gate.json` |

协议 / 计划 / 声称模板（在 git freeze 内）：

- `schemas/examples/research_protocol_rgbt_dfine_formal_c1_v1.json`
- `schemas/examples/experiment_plan_formal_01_rgb_none.json`
- `schemas/examples/experiment_plan_formal_02_early_concat.json`
- `schemas/examples/claim_c1_fusion_aps.json`

不要把 `candidate/`（中断臂）当 C1。不要演示伪造 metrics。REAL 必须 `--execute`。

## 现场打开顺序

1. 两臂 `run/metrics.json`（APS）与 `handle.json`（五哈希一致，HOW 在 notes）
2. `claim_gate_c1.json`（SUPPORTED，`keep_is_not_claim=true`）
3. `candidate_resume/claim_gate.json`（BLOCKED）
4. 命令重放：见 `docs/defense/DEMO_SCRIPT.md`

## 不要讲成已交付

新 GPU 重跑、`trajectory_step`、训练飞轮、Bounded Tree、LLM Planner/Reviewer、FDPN C2、论文级 SOTA、Web 产品化、v2.5 GATE / dataset 战役。  
工作树里的 v2.5 脏文件是 **post-MVP** 状态，开发线 `feat/post-mvp-v25`，不得回灌 `mvp-freeze-m1-m4-claimgate-c1`。
