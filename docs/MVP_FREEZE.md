# MVP Freeze（M1–M4 + ClaimGate + Formal C1）

日期：2026-08-17  
仓库（磁盘为准）：`d:\AI Scientist_tiao\scientist-lab`  
分支：`feat/v24-full-fusion`  
权威架构文本（不要整篇改写）：`d:\AI Scientist_tiao\设计架构.md`、`d:\AI Scientist_tiao\新版构建方案.md`

> 纠正：用户文本里「当前会话没有源码仓库」已过时。Scientist Lab 源码在 `d:\AI Scientist_tiao\scientist-lab`，本 Freeze 以该仓库核对结果为准，不伪造「无仓库」。

**停止扩功能。** 不新开 GPU probe/formal，不跑 trajectory_step / 训练飞轮 / Bounded Tree / LLM Planner / FDPN / SOTA。

## 冻结的两条核心能力

1. **自主科研行为闭环**：Gate → Run → Evidence → Rubric → Memory → Next Plan  
2. **科学声称闭环**：ClaimGate 限制 Evidence 最多支持的 Claim 强度

源码锚点：

- 行为闭环：`src/scientist_lab/core/manager.py` + GateEngine / ResultParser / EvidenceValidator / DecisionRubric / MemoryWriter / next_plan  
- 声称闭环：`src/scientist_lab/core/claim_gate.py`（不是 Agent，不覆盖 Reviewer）

## 已接受 Formal C1

同一 Frozen Fingerprint：

| 臂 | 目录 | HOW | APS |
|----|------|-----|-----|
| baseline | `.run/formal_c1_aps_early_concat/baseline` | rgb + none | **0.0163** |
| candidate | `.run/formal_c1_aps_early_concat/candidate_resume` | rgbt + early_concat | **0.0326** |

配对 ClaimGate（`--baseline-run-dir`）：`.run/formal_c1_aps_early_concat/claim_gate_c1.json` → C1 **SUPPORTED**。  
Manager 自动 `claim_gate.json`（仅 `baseline_metrics`、无配对指纹）→ **BLOCKED**，这是正确行为，不是 bug。

这些 `.run/` 产物 gitignore，不进入 freeze commit。

## 能力边界（必须保留，不得升格）

- 160×160 / 2 epoch 是 staging/formal-C1，不是 640/20ep 论文协议
- early_concat 是仓库已有的 staging 融合 HOW，**不是 FDPN**，不得升 C2
- probe ≠ formal evidence；16/8 fast_eval 不得当 C1
- mAP ≠ APS；声称 small-object 但证据只有 mAP → BLOCKED
- KEEP/DISCARD ≠ ClaimGate SUPPORTED/INCONCLUSIVE/BLOCKED
- `scientific_outcome` 是只读投影，不得与 ClaimGate 混读，不得形成平行科研判断状态机
- 科研负结果不是 error；FAILED/INVALID 才是工程/协议错误

## 明确不进 MVP

新 GPU probe/formal、`trajectory_step`、训练飞轮、preference/SFT/DPO/RL、Bounded Tree、LLM Planner/Reviewer、FDPN C2、论文级 SOTA。

## Audit

详见 `docs/ARCHITECTURE_TO_CODE_AUDIT.md`。核对后终态 **PASS**（checklist 9/9；pytest 676 passed / 12 skipped）。

## Git Freeze 意图

- 只提交 freeze 文档 + M1–M4 / ClaimGate / Formal C1 协议与 HOW 源码
- 不提交 `.run/`、secrets、`.env`、数据集图像、v2.5 GATE 战役文档、bench 产物
- 不改 git config，不 push，不用 `--no-verify`
- annotated tag：`mvp-freeze-m1-m4-claimgate-c1`
