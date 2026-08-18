# v2.5-D Demo 验收（比赛版闭环）

日期：2026-08-18  
分支：`feat/post-mvp-v25`  
不解冻 MVP。tag `mvp-freeze-m1-m4-claimgate-c1` @ `130b02c` 只读。

**Verdict:** `accept_v25d_demo_loop`  
**Action:** stop at D（工程闭环已证明）。不把本 probe 升格为 C1，不开 Formal E，除非比赛规则另要求。

## 证明的链

```text
历史 VALID Evidence（Round1 neck DISCARD）
  → LLM Reviewer（不覆盖 Rubric）
  → MemoryWriter（evidence_refs 通过才写）
  → LLM Planner（selected = fusion，不是 neck）
  → Plan（budget_class=probe）
  → Gate APPROVED
  → REAL GPU fast_eval 16/8
  → 新 Evidence（VALID，APS=0.0）
  → LLM Reviewer 再解释（仍 DISCARD）
  → ClaimGate BLOCKED（不是 SUPPORTED C1）
```

LLM 无 `LLM_API_KEY`，认知后端为 **mock / FakeProvider**。GPU 与 LLM live 独立。

## 命令

```text
scientist-lab llm-review-replay --run-dir .run/v25d_demo/history_source
scientist-lab llm-plan-replay --ab --run-dir .run/v25d_demo/history_source
scientist-lab llm-real-loop --run-dir .run/v25d_demo/history_source --output-dir .run/v25d_llm_real_loop
scientist-lab dfine-cuda-doctor   # live_ready=true
scientist-lab llm-real-loop --execute --require-live-ready --run-dir .run/v25d_demo/history_source --output-dir .run/v25d_llm_real_loop_gpu
```

「下一步」仅许可这一次 probe REAL。formal / 4h / SOTA / 发明 FDPN 仍禁止。

## 闸门核对

| 项 | 结果 |
|----|------|
| 默认 manager-run 仍 rules | 保持（本环显式 llm） |
| Gate | **APPROVED**；无 APPROVED 不得点火 |
| Rubric 历史 | Round1 **DISCARD**，LLM 未覆盖 |
| 新一轮 Rubric | **DISCARD**（probe APS=0.0 vs labeled control 0.6） |
| MemoryWriter | 写入 `LESSON-run_plan_round1_neck_hr-semantic-001`；Plan 引用真实 lesson |
| Planner selected | **fusion**（neck 仅作 `reason_not_selected`） |
| Adapter HOW | rgbt + early_concat；`invented_operators=[]`；不是 FDPN |
| doctor | `live_ready=true`（RTX 5070 Ti + `scientist-rgbt-detection:v2-cuda`） |
| 点火 | `ignited=true`，`metrics_forged=false` |
| ClaimGate | **BLOCKED**（`baseline comparison lacks matched Frozen Fingerprint`）；`claim_gate_not_c1_supported=true` |
| 证据等级 | `engineering_probe_not_c1` · 16 train / 8 val · 1 epoch smoke |

## REAL 数字（不得写成 C1 / 模块无效）

| 项 | 值 |
|----|----|
| exec | `exec_824958918784` |
| device | cuda |
| duration | 39.3 s |
| peak GPU mem | 364.5 MB |
| subset | 16 / 8 |
| APS / mAP50 / mAP50_95 | **0.0 / 0.0 / 0.0** |
| prediction_count | 200（score≥0.1 为 0） |

APS=0.0 是 **VALID 负结果**。闭环成功叙事是「LLM 决策进了真路径，Gate / Rubric / ClaimGate 仍在」。  
**禁止**写成 fusion 无效、C1 SUPPORTED、或 Formal E 已完成。

## 磁盘产物（gitignore，不入库）

- dry-run：`.run/v25d_llm_real_loop/loop_report.json`
- REAL：`.run/v25d_llm_real_loop_gpu/loop_report.json`
- GPU 原始：`outputs/project_rgbt_cuda_001/exec_824958918784`

## Rejected

- 把 probe APS=0.0 写成科学结论或 C1
- 自动升 `budget_class=formal`
- 现在就开独立 Formal E（仅当比赛规则明确需要）

## Next direction

比赛版可停在 D。若规则要求 Formal E，另开配对 formal，不升格本 probe。
