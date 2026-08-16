# MVP 封版说明（一页答辩）

日期：2026-08-17  
产品定位：**可控自主实验系统**（人设目标 / 边界 / 预算，系统跑 Gate→Run→Evidence→Memory→Next Plan，并用 ClaimGate 限制能说到哪一步）。**不是**更好的检测器，不是 SOTA 论文结果。

停止扩功能。本页不升级声称。v2.5 / LLM / Web / dataset 战役不在本封版内。

## Git 身份

| 项 | 值 |
|----|----|
| 仓库 | `d:\AI Scientist_tiao\scientist-lab` |
| annotated tag | `mvp-freeze-m1-m4-claimgate-c1` |
| SHA | `130b02cfbb5521829e959d10b99d17fd5fff28ab` |
| 保护分支 | `freeze/mvp-m1-m4-claimgate-c1`（同 SHA；不得移动 tag） |
| 封版时所在分支 | `feat/v24-full-fusion` ahead 1，未 push |
| 测试口径 | **676 passed / 12 skipped**（Audit 核对；本封版材料未改代码、未重跑） |

后续开发不得把 commit 打进这个 tag。权威冻结文本：`docs/MVP_FREEZE.md`、`docs/ARCHITECTURE_TO_CODE_AUDIT.md`。答辩证据链：`docs/MVP_DEFENSE_EVIDENCE.md`。

## 两条已冻结闭环

1. **自主科研行为闭环**：Gate → Run → Evidence → Rubric → Memory → Next Plan  
   源码：`src/scientist_lab/core/manager.py` + GateEngine / ResultParser / EvidenceValidator / DecisionRubric / MemoryWriter / next_plan。
2. **科学声称闭环**：ClaimGate 限制 Evidence 最多支持的 Claim 强度。  
   源码：`src/scientist_lab/core/claim_gate.py`（确定性规则服务，不是 Agent，不覆盖 Reviewer）。

Architecture-to-Code Audit：**PASS**（checklist 9/9）。详见 `docs/ARCHITECTURE_TO_CODE_AUDIT.md`。

## Formal C1（已接受，不得改数字）

同一 Frozen Fingerprint（HOW 只进 notes，不进哈希）：

| 臂 | HOW | APS |
|----|-----|-----|
| Formal-01 baseline | rgb + none | **0.0163** |
| Formal-02 candidate | rgbt + early_concat | **0.0326** |

配对 ClaimGate（`--baseline-run-dir`）→ C1 **SUPPORTED**。  
Manager 单臂 `baseline_metrics`（无配对指纹）→ **BLOCKED**（正确，不是 bug）。

`.run/` 不进 git。路径索引见证据链文档。

## 能力边界（答辩必须先说）

- 160×160 / 2 epoch 是 staging / Formal-C1，**不是** 640/20ep 论文协议。
- early_concat 是仓库已有的 staging 融合 HOW，**不是 FDPN**，不得升 C2。
- probe ≠ formal evidence；16/8 fast_eval 不得当 C1。
- mAP ≠ APS；声称 small-object 但证据只有 mAP → BLOCKED。
- KEEP / DISCARD ≠ ClaimGate SUPPORTED / INCONCLUSIVE / BLOCKED。
- `scientific_outcome` 是只读投影，可与 ClaimGate 并存，不得混读成第二条声称状态机。
- 科研负结果不是 error；FAILED / INVALID 才是工程 / 协议错误。

## 明确未进入 MVP

新 GPU probe/formal、`trajectory_step`、训练飞轮、preference / SFT / DPO / RL、Bounded Tree、LLM Planner / Reviewer、FDPN C2、论文级 SOTA、Web 产品化、v2.5 GATE 战役与 dataset 注册。

当前工作树里的 v2.5 脏文件是 **post-MVP worktree state**，开发线 `feat/post-mvp-v25`，不得回灌 freeze tag。
