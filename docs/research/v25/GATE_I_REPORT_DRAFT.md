# Gate I Report — Formal A2 Multiseed（骨架草稿）

> **状态：** `DRAFT` — seed44 仍在跑；数字以 `MULTISEED_SUMMARY.json` / `build_gate_i_summary.py` 终稿为准。  
> **权限：** `baseline_formal_candidates` · 不允许科学优越性 / SOTA / A4·P01 排名。

## 1. 一句话结论（待填）

- Engineering（三种子）：`TBD`
- Formal candidates OK：`TBD`
- Decision：`TBD`  
  候选：`freeze_a2_formal_baseline_candidates` / `diagnose_seed_instability` / `fail_engineering`

## 2. 协议与不变量

| 项 | 冻结值 |
|--|--|
| Gate | `GATE_I_A2_FORMAL_MULTISEED` |
| Arm | A2 / `early_concat` only |
| Dataset | `dataset:rgbt_tiny_v1` · `v1_gate_h_expand` |
| Split | sequence-level · train 1830 / val 500 / test 300 |
| Input | 640×640 · queries=300 · batch=1 · 20ep · lr=2e-4 · AMP |
| Seeds | 42 / 43 / 44 |
| Docker | `scientist-rgbt-detection:v2-cuda` |
| Blocked | A4 · P01 · FDPN · `gated_multiscale` |

继承自 Gate H；相对 H **仅允许改 seed**。seed42 **复用** `exec_4088755cafc6`（同超参同子集，不重训）。

## 3. 执行矩阵（部分已填）

| seed | policy | execution_id | status | best ep | mAP50_95 | mAP50 | eng |
|------|--------|--------------|--------|---------|----------|-------|-----|
| 42 | reuse_gate_h | `exec_4088755cafc6` | completed | 19 | 0.083347 | 0.189676 | true |
| 43 | fresh (resume rerun) | `exec_a6f6dff68971` | completed | 3 | 0.108174 | 0.226189 | true |
| 44 | fresh | `exec_373196093d79` | **running** | — | — | — | — |

备注：
- seed43 曾 `exec_a36b54f68731` 用户暂停取消；正式计入的是重跑 `exec_a6f6dff68971`。
- seed43 best@**ep3** + 较大 best−last（约 0.070）→ 曲线形态弱于 Gate H seed42；汇总时写入 limitations，不单独否决工程完成，除非三种子发散超阈值。

## 4. 聚合指标（seed44 完成后填）

- mean mAP50_95 = **TBD**
- stdev = **TBD**
- min / max = **TBD**
- spread_ok 规则：`(max−min) ≤ max(0.05, 0.75·mean)` → **TBD**

## 5. 可写 / 不可写

**可写**
- 在 Gate-H 冻结子集上，A2 early_concat 完成三种子执行与工程门禁。
- 给出 baseline **候选**均值/方差（exploratory / formal-candidates 层级）。

**不可写**
- A2 已是正式 SOTA / 方法优越性 / 显著性结论。
- 据此开启或宣称 A4 / P01 / FDPN 有效。
- 用 seed43 早期峰值单独论证“数据已足够”或“应立刻加方法”。

## 6. 与 Gate G / H 对照

| | Gate G | Gate H (seed42) | Gate I (三种子) |
|--|--------|-----------------|----------------|
| 子集 | 小 | 扩（1830/500） | 同 H |
| epochs | 10 | 20 | 20 |
| best mAP50_95 | ≈0.0063 | ≈0.0833 | mean TBD |
| 角色 | 探针 | 曲线健康 | 正式 A2 候选 |

## 7. 风险与局限（预写）

1. **batch=1**：5070 Ti 利用率常 <30%；吞吐低，但不改变当前协议可比性。  
2. **seed 间曲线形态不一致**（H@ep19 vs I-seed43@ep3）→ 多种子均值需伴随 stdev 与 best-epoch 分布。  
3. **DFINE staging / 宿主机占用**可能引入时间噪声；指标以 val mAP 为准，不以 wall-clock 排名。  
4. 子集仍非全量 RGBT-Tiny；结论绑定 `v1_gate_h_expand`。

## 8. 决策树（终稿勾选）

- [ ] `freeze_a2_formal_baseline_candidates` → A2 正式基线候选冻结；下一扇门另开  
- [ ] `diagnose_seed_instability` → 查 outlier seed / 配对 / checkpoint  
- [ ] `fail_engineering` → 先修工程门禁  

## 9. 建议下一扇门（仅候选，不自动执行）

1. **Gate J（工程）**：吞吐探针（见并列草稿）——不改科学结论，只测 `batch/workers`。  
2. **数据扩容**：更大序列子集或降低 stride（需新 DATA_SUBSET 冻结）。  
3. **仍冻结方法**：在 A2 候选未稳前，**不**开 A4/P01。

## 10. 产物路径

- 运行根：`outputs/experiments/v25_real_rgbt/gate_i_a2_formal_multiseed/`
- 终稿脚本：`build_gate_i_summary.py` → `GATE_I_FREEZE.json` · `MULTISEED_SUMMARY.json` · `GATE_I_REPORT.md`
- 文档镜像：`docs/research/v25/`

---
*草稿生成于 seed44 训练期间；完成后用 summary 脚本覆盖数字并去掉 DRAFT 标记。*
