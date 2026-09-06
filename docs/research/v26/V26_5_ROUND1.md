# V26.5 Round 1 — facts

日期：2026-08-19  
战役：v2.6 / V26.5  
产物：`outputs/v26_r1/`  
不是 G2 成功声明。不是 A4。不是第五个 Agent。未 commit / push。

## 跑了几轮

本回合完整落地 **第 1 轮**。没有假装做完 3～5 轮。没有开第 2 轮 GPU。

## 相对 R0

| 项 | R0（冻） | Round 1 |
|----|----------|---------|
| HOW | F1 `early_concat` + standard neck | **F3** `gated_multiscale` + standard neck |
| dataset / slice | `rgbt_tiny_v1` / `low_light_subset_v1` | 同 |
| fingerprint | `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6` | 同家族（HOW 不进 Frozen hash） |
| budget | formal / full_train / seed 42 | 同 |
| **APS_lowlight** | **0.0045926865160844455** | **0.02135704762627717** |
| Δ vs R0 | — | **+0.016764361110192725** |
| mAP50_95_lowlight | 0.003022687959416924 | 0.01417395290498084 |
| AP50_lowlight | 0.019216142380591137 | 0.07135570259460988 |
| 切片 val | 100 图 / 851 标注 | 同 |
| evaluator | pycocotools | pycocotools |

全集 `APS` / `mAP50_95` 只是补充，**不是** `APS_lowlight`。

## 闭环事实

1. Live Planner：`openai-compatible` / `qwen3.7-max`，`source=llm`。选 F3；未选 F0（丢 thermal）与 F1（R0 重复）。
2. 文献：`SEMANTIC_SCHOLAR_API_KEY` 未配置。`literature-search` **fake**（`litq_b353b5876102`）。LiteratureEvidence ≠ ExperimentEvidence。无全文，未声称正文细节。
3. Adapter HOW：`fusion_type=GatedMultiscaleFusion`，`how_id=F3`。N1/A4 未出现。
4. Human Gate：战役一次性 `--confirm-human-gate`。Protocol `full_training=approval_required` 未改。
5. GPU：`live_ready=true`，镜像 `scientist-rgbt-detection:v2-cuda`。约 27 分钟。未伪造指标。
6. Rubric：**KEEP**（Δ=+0.01676）。KEEP ≠ Claim。
7. Live Reviewer：GPU 当轮曾 schema fail-closed（`reviewer_live_fail_closed.json`，`$.selected`）。KEEP/DISCARD 仍由 DecisionRubric 写出。2026-08-20 live replay 已补闭合，见下节。
8. ClaimGate：**BLOCKED**。原因：`claim_strength C1 exceeds protocol max_claim_strength=C0`。协议禁止科学声称。G2 在 2-seed 确认前不对外宣称成功。
9. Memory：`LESSON-run_plan_round1_from_exec_31eff20c4e0c-001`（positive_evidence）+ `STRATEGY-...-001`（prioritize fusion）。另有语义 lesson `LESSON-run_plan_round1_from_exec_31eff20c4e0c-semantic-001`（提案，不是声称）。

## Live Reviewer replay（2026-08-20）

未重跑 GPU。未开第 2 轮。`review.json` / `claim_gate.json` / `result.json` 哈希未变。

```text
scientist-lab llm-review-replay --run-dir outputs/v26_r1 --live --persist-pack
```

- 根因：Reviewer structured output 被套用 Planner `selected`；合法 Reviewer JSON 无 `selected` 应通过。TLS 上 conda `SSL_CERT_FILE` 曾挡住 Windows ROOT 企业/自签 CA；现优先 `LLM_CA_BUNDLE` → Windows ROOT，未关 TLS。
- 结果：`ok=true`，`provider=openai-compatible`，`model=qwen3.7-max`，`tls_verify_source=windows_root_store`。
- 写出 `outputs/v26_r1/semantic_review.json`。`semantic_proposal.hypothesis_status=not_a_claim`。KEEP ≠ Claim。不是 G2。
- 删除 `reviewer_live_fail_closed.json`。Rubric 仍 KEEP。ClaimGate 仍 BLOCKED（max C0）。
- `APS_lowlight` 仍为 **0.02135704762627717**（相对 R0 **0.0045926865160844455**）。

## 下一步

- 第 2 轮已落地：见 [`V26_5_ROUND2.md`](./V26_5_ROUND2.md) · `outputs/v26_r2/`（F0 `APS_lowlight=1.3452432199741713e-06`，Rubric KEEP/INCONCLUSIVE，不是 G2）。
- 第 3 轮：同一 slice/fingerprint 家族、`budget_class=formal`。Next Plan 指向 F3 第二 seed 确认；执行前对齐 seed 合同。允许负结果 / DISCARD。
- 配 Semantic Scholar key 后，文献改为 live provenance；仍不得进 ClaimGate。
- G2 需要 ≥2 seed 同协议确认后才能说「高于 R0」。本轮数字只是单次 formal 对照。
