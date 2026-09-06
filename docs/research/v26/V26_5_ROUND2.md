# V26.5 Round 2 — facts

日期：2026-08-20  
战役：v2.6 / V26.5  
产物：`outputs/v26_r2/`  
不是 G2 成功声明。不是 A4。不是第五个 Agent。未 commit / push。

## 跑了几轮

本回合完整落地 **第 2 轮**（独立 pack）。没有开第 3 轮 GPU。Next Plan 已写出、未执行。

## 相对 R0 与第 1 轮

| 项 | R0（冻） | Round 1 | Round 2 |
|----|----------|---------|---------|
| HOW | F1 `early_concat` + standard neck | F3 `gated_multiscale` + standard neck | **F0** RGB-only `none` + standard neck |
| dataset / slice | `rgbt_tiny_v1` / `low_light_subset_v1` | 同 | 同 |
| fingerprint 家族 | `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6` | 同（HOW 不进 Frozen hash） | 同；`fingerprint_comparable=true` |
| budget | formal / full_train / seed 42 | 同 | 同 |
| **APS_lowlight** | **0.0045926865160844455** | **0.02135704762627717** | **1.3452432199741713e-06** |
| Δ vs R0 | — | +0.016764361110192725 | **-0.004591341272864471** |
| Δ vs Round 1 | — | — | **-0.021355702383057194** |
| mAP50_95_lowlight | 0.003022687959416924 | 0.01417395290498084 | 1.1731030922997514e-06 |
| AP50_lowlight | 0.019216142380591137 | 0.07135570259460988 | 1.1731030922997514e-05 |
| 切片 val | 100 图 / 851 标注 | 同 | 同 |
| evaluator | pycocotools | pycocotools | pycocotools |

全集 `APS=0.013645801790600211` / `mAP50_95=0.013590938514004714` 只是补充，**不是** `APS_lowlight`。不得用全集数字冒充切片结论。

## 闭环事实

1. Live Planner：`openai-compatible` / `qwen3.7-max`，`source=llm`。选 **F0** RGB-only 对照（检验 thermal 是否为 Round 1 F3 KEEP 所必需）。未选同 seed 重复 F3，也未重复 R0 F1。
2. 文献：`SEMANTIC_SCHOLAR_API_KEY` 未配置。`literature-search` **fake**（`litq_1716267bf83f`）。LiteratureEvidence ≠ ExperimentEvidence。无全文，未声称正文细节。LiteratureEvidence 未进 ClaimGate。
3. Adapter HOW：`input_mode=rgb`，`fusion_method=none`，`how_id=F0`。N1/A4 未出现。
4. Human Gate：战役一次性 `--confirm-human-gate`。Protocol `full_training=approval_required` 未改。
5. GPU：`live_ready=true`，镜像 `scientist-rgbt-detection:v2-cuda`。约 25 分钟（`duration_seconds=1492.5`）。`metrics_forged=false`。未伪造指标。
6. Rubric：**KEEP**。Δ vs R0 = -0.00459，未越过 `keep_requires.primary_not_worse_than=-0.1`，也未越过 `discard_if.primary_worse_than=-0.5`。`hypothesis_status=INCONCLUSIVE`。KEEP ≠ Claim。相对 Round 1 F3 的切片崩塌是机制观察，不是 Rubric DISCARD。
7. Live Reviewer：`scientist-lab llm-review-replay --run-dir outputs/v26_r2 --live --persist-pack`。`ok=true`，`tls_verify_source=windows_root_store`。`semantic_proposal.hypothesis_status=not_a_claim`。未覆盖 `review.json`。
8. ClaimGate：**BLOCKED**。原因：`claim_strength C1 exceeds protocol max_claim_strength=C0`。协议禁止科学声称。不是 G2。
9. Memory：`LESSON-run_plan_round2_from_run_plan_round1_from_exec_31eff20c4e0c-001`（inconclusive）+ `STRATEGY-...-001`（keep fusion）+ semantic `LESSON-...-semantic-001`（提案，不是声称）。
10. Next Plan（未执行 GPU）：Live LLM 选 F3，假设为第二 seed 确认 Round 1 KEEP。`plan_round3_next.json` 的 `proposed_changes.detail.seed=43`，但骨架 `evaluation.seeds` 仍为 `[42]`。第 3 轮若点火，必须先对齐 seed 合同，不能把这份 JSON 直接说成已经 2-seed。

## Replay 命令

```text
scientist-lab llm-review-replay --run-dir outputs/v26_r2 --live --persist-pack
```

## 可见性

- `/loop/v26_r2`（catalog `v26_r2`）
- `/training` 战役行 `V26_R2`

## 下一步

- 第 3 轮：同一 slice/fingerprint 家族、`budget_class=formal`。Next Plan 指向 F3 第二 seed 确认；执行前对齐 `evaluation.seeds`。允许负结果 / DISCARD。
- 配 Semantic Scholar key 后，文献改为 live provenance；仍不得进 ClaimGate。
- G2 需要 ≥2 seed 同协议确认后才能说「高于 R0」。Round 1 单 seed KEEP 与 Round 2 F0 对照都不是 G2。
