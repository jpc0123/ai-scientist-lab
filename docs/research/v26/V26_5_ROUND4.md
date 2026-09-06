# V26.5 Round 4 — facts

日期：2026-08-20  
战役：v2.6 / V26.5  
产物：`outputs/v26_r4/`  
不是 G2 成功声明。不是 A4。不是第五个 Agent。未 commit / push。

## 跑了几轮

本回合完整落地 **第 4 轮**（独立 pack）。Next Plan：F3 第三 seed **44**。seed 合同点火前已对齐。

## 相对 R0 与前三轮

| 项 | R0（冻） | Round 1 | Round 2 | Round 3 | Round 4 |
|----|----------|---------|---------|---------|---------|
| HOW | F1 | F3 | F0 | F3 | **F3** |
| seed | 42 | 42 | 42 | 43 | **44** |
| **APS_lowlight** | **0.0045926865160844455** | **0.02135704762627717** | **1.3452432199741713e-06** | **0.04881493259150456** | **0.04204268629433699** |
| Δ vs R0 | — | +0.01676 | -0.00459 | +0.04422 | **+0.03744999977825254** |
| mAP50_95_lowlight | 0.003022687959416924 | 0.01417395290498084 | 1.1731030922997514e-06 | 0.031873018410005366 | 0.027537416857751757 |
| AP50_lowlight | 0.019216142380591137 | 0.07135570259460988 | 1.1731030922997514e-05 | 0.102994826126478 | 0.06352353238168447 |

dataset / slice / fingerprint 家族与 R0 相同：`rgbt_tiny_v1` + `low_light_subset_v1` + `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6`。切片 val 100/851。pycocotools。budget=formal。

全集 `APS=0.07224547515365953` / `mAP50_95=0.06387059388987293` **不是** `APS_lowlight`。

## 闭环事实

1. Live Plan：Round 3 Next Plan，F3 seed 44。`source=llm` / `qwen3.7-max`。N1/A4 未出现。
2. 文献：fake `litq_b01729a6f7aa`。无 S2 key。未进 ClaimGate。
3. Adapter：`fusion_method=gated_multiscale`，`how_id=F3`，`seed=44`（`run/metrics.json`）。
4. Human Gate：`--confirm-human-gate`。Protocol 未改。
5. GPU：`live_ready=true`。约 29 分钟（`duration_seconds=1713.6`）。`metrics_forged=false`。exec=`exec_a6dedd7f6c00`。
6. Rubric：**KEEP**。KEEP ≠ Claim。
7. Live Reviewer：本轮 GPU 当时未 persist-pack。`run_id` 101 字符，与 R3 同一嵌套截断风险。2026-08-20 与 R3 同一修复后 `--live --persist-pack` 已写出 `semantic_review.json`（`hypothesis_status=not_a_claim`）。Rubric KEEP 未覆盖。TLS=`windows_root_store`。未假装成功。不是 G2。
8. ClaimGate：**BLOCKED**（max C0）。不是 G2 成功声明。
9. Memory：`LESSON-run_plan_round4_...-001`（positive_evidence）+ `STRATEGY-...-001`。另有语义 lesson `LESSON-...-semantic-001`（提案，不是声称）。

## 3-seed F3 方向（仍不可对外宣称 G2）

| F3 seed | APS_lowlight | vs R0 |
|---------|--------------|-------|
| 42 | 0.02135704762627717 | 高于 R0 |
| 43 | 0.04881493259150456 | 高于 R0 |
| 44 | 0.04204268629433699 | 高于 R0 |

同方向。幅度有方差。协议 `max_claim_strength=C0`，ClaimGate BLOCKED，**仍不可宣称 G2 成功**。

## Live Reviewer replay（2026-08-20）

未重跑 GPU。`APS_lowlight` 仍为 **0.04204268629433699**。

```text
scientist-lab llm-review-replay --run-dir outputs/v26_r4 --live --persist-pack
```

写出 `outputs/v26_r4/semantic_review.json`。`hypothesis_status=not_a_claim`。ClaimGate 仍 BLOCKED。不是 G2。

## 可见性

- `/loop/v26_r4`
- `/training` 战役行 `V26_R4`
