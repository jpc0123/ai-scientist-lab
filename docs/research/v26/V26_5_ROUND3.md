# V26.5 Round 3 — facts

日期：2026-08-20  
战役：v2.6 / V26.5  
产物：`outputs/v26_r3/`  
不是 G2 成功声明。不是 A4。不是第五个 Agent。未 commit / push。

## 跑了几轮

本回合完整落地 **第 3 轮**（独立 pack）。seed 合同点火前已对齐为 **43**，没有静默用 42 重跑第 1 轮。

## 相对 R0 与前两轮

| 项 | R0（冻） | Round 1 | Round 2 | Round 3 |
|----|----------|---------|---------|---------|
| HOW | F1 `early_concat` + standard neck | F3 `gated_multiscale` + standard neck | F0 RGB-only `none` + standard neck | **F3** `gated_multiscale` + standard neck |
| seed | 42 | 42 | 42 | **43** |
| dataset / slice | `rgbt_tiny_v1` / `low_light_subset_v1` | 同 | 同 | 同 |
| fingerprint 家族 | `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6` | 同 | 同 | 同；`fingerprint_comparable=true` |
| budget | formal / full_train | 同 | 同 | 同 |
| **APS_lowlight** | **0.0045926865160844455** | **0.02135704762627717** | **1.3452432199741713e-06** | **0.04881493259150456** |
| Δ vs R0 | — | +0.016764361110192725 | -0.004591341272864471 | **+0.044222246075420114** |
| mAP50_95_lowlight | 0.003022687959416924 | 0.01417395290498084 | 1.1731030922997514e-06 | 0.031873018410005366 |
| AP50_lowlight | 0.019216142380591137 | 0.07135570259460988 | 1.1731030922997514e-05 | 0.102994826126478 |
| 切片 val | 100 图 / 851 标注 | 同 | 同 | 同 |
| evaluator | pycocotools | pycocotools | pycocotools | pycocotools |

全集 `APS=0.0450449997658558` / `mAP50_95=0.03533560040145398` 只是补充，**不是** `APS_lowlight`。不得用全集数字冒充切片结论。

## seed 合同

- Next Plan `proposed_changes.detail.seed=43`，骨架曾为 `evaluation.seeds=[42]`。
- 点火前对齐：`evaluation.seeds=[43]` 且 `detail.seed=43`。Adapter 读 `evaluation.seeds[0]`。`outputs/v26_r3/run/metrics.json` 的 `training.seed=43`。
- 不是静默 seed-42 重跑 Round 1。

## 闭环事实

1. Live Plan：沿用 Round 2 已写出的 Next Plan（F3 第二 seed）。`source=llm` / `qwen3.7-max`。HOW=F3。N1/A4 未出现。
2. 文献：`SEMANTIC_SCHOLAR_API_KEY` 未配置。`literature-search` **fake**（`litq_52d7430cd4bb`）。LiteratureEvidence ≠ ExperimentEvidence。无全文。未进 ClaimGate。
3. Adapter HOW：`input_mode=rgbt`，`fusion_method=gated_multiscale`，`how_id=F3`。N1/A4 未出现。
4. Human Gate：战役一次性 `--confirm-human-gate`。Protocol `full_training=approval_required` 未改。
5. GPU：`live_ready=true`，镜像 `scientist-rgbt-detection:v2-cuda`。约 30 分钟（`duration_seconds=1762.5`）。`metrics_forged=false`。未伪造指标。exec=`exec_74c7a4f99640`。
6. Rubric：**KEEP**。Δ vs R0 = +0.04422。KEEP ≠ Claim。
7. Live Reviewer：GPU 当轮两次 `--live --persist-pack` 曾 fail-closed。根因：嵌套 `run_id`（80 字符）被 LLM 截成 `...exec_31eff20c`（缺尾缀 `4e0c`，git-short-SHA 习惯），精确匹配拒写。不是 schema `maxLength` 层、不是 TLS。DecisionRubric `review.json` 未覆盖。ClaimGate 未覆盖。KEEP 仍由 Rubric 锁定。2026-08-20 代码修复后 live replay 已补闭合，见下节。
8. ClaimGate：**BLOCKED**。原因：`claim_strength C1 exceeds protocol max_claim_strength=C0`。协议禁止科学声称。不是 G2 成功声明。
9. Memory：Rubric 写出 `LESSON-run_plan_round3_from_run_plan_round2_from_run_plan_round1_from_exec_31eff20c4e0c-001`（positive_evidence）+ `STRATEGY-...-001`（prioritize fusion）。

## 2-seed 对 F3 的确认材料（不是对外 G2 宣称）

| F3 seed | APS_lowlight | vs R0 0.0045926865160844455 |
|---------|--------------|-----------------------------|
| 42（Round 1） | 0.02135704762627717 | 同方向，高于 R0 |
| 43（Round 3） | 0.04881493259150456 | 同方向，高于 R0 |

同方向：两个 seed 的 F3 都高于冻结 R0。数值上已进入「可判定 G2 的证据门槛」（V26.6：2-seed 确认 best）。**仍不可对外宣称 G2 成功**：协议 `max_claim_strength=C0`，ClaimGate BLOCKED，KEEP ≠ Claim。两 seed 幅度差大（0.021 vs 0.049），只报告方差，不把较大的那次说成更强发现。

## Replay 命令

```text
scientist-lab llm-review-replay --run-dir outputs/v26_r3 --live --persist-pack
```

## Live Reviewer replay（2026-08-20）

未重跑 GPU。未改 `APS_lowlight` / `max_claim_strength` / `max_rounds`。`review.json` / `claim_gate.json` / `result.json` 未覆盖。

- 拒写层：`parse_reviewer_completion` 要求 `evidence_refs`/`created_from` **精确等于**合同 `run_id`。LLM 合法 JSON 通过 schema，但把 `exec_31eff20c4e0c` 收成 8 位 hex。不是 `selected` 套用。
- 修复：合同 `run_id` 的 unambiguous 前缀（≥32 字符）规范化回全 id；外键/短前缀仍 fail-closed。Prompt 禁止当 git SHA 截断。
- 结果：`ok=true`，`provider=openai-compatible`，`model=qwen3.7-max`，`tls_verify_source=windows_root_store`。写出 `outputs/v26_r3/semantic_review.json`。`hypothesis_status=not_a_claim`。删除 `reviewer_live_fail_closed.json`。Rubric 仍 KEEP。ClaimGate 仍 BLOCKED（max C0）。KEEP ≠ Claim。不是 G2。
- `APS_lowlight` 仍为 **0.04881493259150456**（相对 R0 **0.0045926865160844455**）。

## 可见性

- `/loop/v26_r3`（catalog `v26_r3`）
- `/training` 战役行 `V26_R3`

## 下一步

- 第 4 轮仅当 Next Plan 仍在战役范围（HOW ∈ F0/F1/F3/N0/N1，同一 slice/fingerprint，不动 A4）。
- 配 Semantic Scholar key 后，文献改为 live provenance；仍不得进 ClaimGate。
- 不要把本轮数字说成 G2 已成功。
