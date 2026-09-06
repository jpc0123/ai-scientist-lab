# V26.5 Round 5 — facts

日期：2026-08-20 / 2026-08-21  
战役：v2.6 / V26.5  
产物：`outputs/v26_r5/`  
不是 G2 成功声明。不是 A4。不是第五个 Agent。未 commit / push。ClaimGate 仍 C0。**P4 推迟。**

## Human Gate A

用户选 **A**。Protocol Amendment：`stop_rules.max_rounds` **仅** 5→6。未改切片 / HOW / A4 / `max_claim_strength`。裁决：[`DECISION_HUMAN_GATE_A.md`](./DECISION_HUMAN_GATE_A.md)。

此前 Manager 在 Gate/GPU 前硬停：`round_index=5 >= max_rounds=5` → STOP，无 `APS_lowlight`，`metrics_forged=false`。Amendment 之后 `5 >= 6` 为假，R5 可点火。备份：`experiment_run.stopped_max_rounds5.json`。

## 跑了几轮

本回合完整落地 **第 5 轮**（独立 pack）。Next Plan：F3 第四 seed **45**。seed 合同点火前已对齐。`--max-extra-rounds 0`，未进入 R6 / P4。

## 相对 R0 与前四轮

| 项 | R0（冻） | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|----|----------|---------|---------|---------|---------|---------|
| HOW | F1 | F3 | F0 | F3 | F3 | **F3** |
| seed | 42 | 42 | 42 | 43 | 44 | **45** |
| **APS_lowlight** | **0.0045926865160844455** | **0.02135704762627717** | **1.3452432199741713e-06** | **0.04881493259150456** | **0.04204268629433699** | **0.035942673873977496** |
| Δ vs R0 | — | +0.01676 | -0.00459 | +0.04422 | +0.03745 | **+0.03134998735789305** |
| mAP50_95_lowlight | 0.003022687959416924 | 0.01417395290498084 | 1.1731030922997514e-06 | 0.031873018410005366 | 0.027537416857751757 | 0.024002524093960583 |
| AP50_lowlight | 0.019216142380591137 | 0.07135570259460988 | 1.1731030922997514e-05 | 0.102994826126478 | 0.06352353238168447 | 0.07224359624863212 |

dataset / slice / fingerprint 家族与 R0 相同：`rgbt_tiny_v1` + `low_light_subset_v1` + `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6`。切片 val 100/851。pycocotools。budget=formal。

全集 `APS=0.04944428649938864` / `mAP50_95=0.042500669459533756` **不是** `APS_lowlight`。

## 闭环事实

1. Live Plan：Round 4 Next Plan，F3 seed 45。`source=llm` / `qwen3.7-max`。N1/A4 未出现。HOW 仅 F0/F1/F3/N0/N1 白名单内的 F3。
2. 文献：pack 内 live `provider=semantic_scholar`，`literature_query_id=litq_a4d7bff94296`（query=`RGB-T gated fusion low-light small object`）。另有点火前 live 探查 `litq_687d9c49f056`（`outputs/v26_literature_live/`）。pack 里还留着一条早期 fake `litq_991c0163c299`，**不是**本轮 ClaimGate 依据。`used_by_plan=[]`。LiteratureEvidence ≠ ExperimentEvidence。**未进 ClaimGate**。
3. Adapter：`fusion_method=gated_multiscale`，`how_id=F3`，`seed=45`（`run/metrics.json`）。
4. Human Gate：`--confirm-human-gate`。Protocol 只改了 `max_rounds` 5→6。
5. GPU：`live_ready=true`。约 27 分钟（`duration_seconds=1634.8463077545166`）。`metrics_forged=false`。exec=`exec_b91239a81097`。`run_state=MEMORY_WRITTEN`。`evidence_status=VALID`。
6. Rubric：**KEEP**。KEEP ≠ Claim。
7. Live Reviewer：GPU 当时 `review.json` 为 DecisionRubric KEEP。随后 `llm-review-replay --run-dir outputs/v26_r5 --live --persist-pack` 写出 `semantic_review.json`（`hypothesis_status=not_a_claim`）。Rubric KEEP 未覆盖。TLS=`windows_root_store`。未假装成功。不是 G2。
8. ClaimGate：**BLOCKED**（max C0；`claim_strength C1 exceeds protocol max_claim_strength=C0`）。不是 G2 成功声明。
9. Memory：`LESSON-run_plan_round5_...-001`（positive_evidence）+ `STRATEGY-...-001`。另有语义 lesson `LESSON-...-semantic-001`（提案，不是声称）。

## 4-seed F3 方向（仍不可对外宣称 G2）

| F3 seed | APS_lowlight | vs R0 |
|---------|--------------|-------|
| 42 | 0.02135704762627717 | 高于 R0 |
| 43 | 0.04881493259150456 | 高于 R0 |
| 44 | 0.04204268629433699 | 高于 R0 |
| 45 | 0.035942673873977496 | 高于 R0 |

同方向。幅度有方差（约 0.021–0.049）。协议 `max_claim_strength=C0`，ClaimGate BLOCKED，**仍不可宣称 G2 成功**。不得把最大 seed（43）当成更强发现。

## Live Reviewer replay（2026-08-21）

未重跑 GPU。`APS_lowlight` 仍为 **0.035942673873977496**。

```text
scientist-lab llm-review-replay --run-dir outputs/v26_r5 --live --persist-pack
```

写出 `outputs/v26_r5/semantic_review.json`。`hypothesis_status=not_a_claim`。ClaimGate 仍 BLOCKED。不是 G2。

LLM sidecar 里的 `next_research_priority` 提到「还可再跑一轮」——**本回合不执行**。Human Gate A 范围在 R5 证据环闭合后 **STOP**。不启动 P4 / RT-DETR。

## STOP

R5 GPU + Reviewer + ClaimGate 已闭合。随后用户说「继续」= **P4 GO**（不是 D-FINE R6）。见 [`DECISION_P4_GO.md`](./DECISION_P4_GO.md)。

## 可见性

- http://127.0.0.1:5174/loop/v26_r5
- http://127.0.0.1:5174/training 战役行 `V26_R5`
