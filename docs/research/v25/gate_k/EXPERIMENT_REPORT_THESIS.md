# 实验报告（论文 / 答辩级）— RGB-T 检测 · A4 最终方法

**线：** v2.5 真实 RGB-T（RGBT-Tiny）  
**最终方法：** A4 = early_concat + FDPN  
**声明级别：** exploratory comparison（禁止显著性 / SOTA）  
**权威冻结：** `FINAL_METHOD_FREEZE.json`  
**决策：** `DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json`

---

## 1. 问题设定

在可见光–热红外（RGB–T）小目标检测中，早期通道拼接（early_concat）是强而简单的融合基线，但多尺度特征利用仍可能不足。本工作在**冻结数据与训练协议**下，检验将 FDPN 颈（多尺度增强）接到 early_concat 堆栈上是否带来一致增益，并排除“靠残差缩放再拧一把”与“换融合再叠 FDPN”两类伪改进。

---

## 2. Dataset（K2-C 冻结）

| 项 | 值 |
|----|-----|
| 注册集 | `dataset:rgbt_tiny_v1` |
| 版本 | `v1_gate_k2c_seq_expand` |
| 划分 | `split:rgbt_tiny_v1_gate_k2c_seq`（序列级） |
| 序列数 | train 72 / val 13 / test 6 |
| 图像对 | train 3342 / val 650 / test 300 |
| 实例数 | train 56650 / val 4526 / test 2792 |
| 采样 | frame_stride=5，max_frames_per_seq=50 |
| 类别 | 7（ship, car, cyclist, pedestrian, bus, drone, plane） |
| 冻结文档 | `K2C_DATASET_FREEZE.json`（含 manifest / annotation hashes） |

相对 Gate-H：K2-C **只**扩大独立训练/验证序列；优化器与评测协议不随之漂移。正式结论**不得**混用 Gate-H 指标。

---

## 3. Protocol（训练与报告）

| 项 | 值 |
|----|-----|
| 检测器 | D-FINE-S（vendored） |
| 融合 | early_concat（数据阶段） |
| 输入尺寸 | 640×640 |
| batch | 1 |
| epochs | 20 |
| LR | 2e-4，warmup=500 |
| 调度 | MultiStep milestones=`[500]`（20ep 内不触发） |
| 主指标 | **mAP50_95 last@20**（非 best-of-run 宣称） |
| 种子 | 42, 43, 44 |
| 环境 | `scientist-rgbt-detection:v2-cuda` · RTX 5070 Ti |
| 报告协议 | `a2_baseline_checkpoint_v2` |

比较协议：`protocol_rgbt_formal_a2_a4_v2`（只重训 A4，复用已冻结 A2 v3）。  
消融协议：`protocol_rgbt_formal_a4_p01_v1`（唯一变量 `residual_scale=0.25`）。

---

## 4. Baseline 稳定性（A2 v3）

A2 = early_concat + 标准 HybridEncoder 颈。在 K2-C 三种子上：

| seed | last@20 | best | exec |
|------|---------|------|------|
| 42 | 0.0529 | 0.141 | `exec_00cc3754ea3d` |
| 43 | 0.1093 | 0.153 | `exec_d4b4feb6cfb1` |
| 44 | 0.0728 | 0.143 | `exec_4f4151310027` |
| **mean** | **0.0783** | — | range **0.0563** |

说明：last@20 仍有种子间波动，但相对早期 Gate-H 不稳路线，K2-C + last@20 已构成可比较基线；后续增益以 **Δ(last) 与正种子数** 叙述，不做显著性检验。

---

## 5. 主结果：A2 vs A4

A4 = early_concat + FDPN（`residual_scale=null` → 完整 FDPN 输出路径）。

| seed | A2 last | A4 last | Δ | A4 exec |
|------|---------|---------|---|---------|
| 42 | 0.0529 | 0.1483 | **+0.095** | `exec_e5e0067e8d1b` |
| 43 | 0.1093 | 0.1381 | **+0.029** | `exec_1f526b78bc5f` |
| 44 | 0.0728 | 0.1217 | **+0.049** | `exec_8bffa02fe333` |
| mean | 0.0783 | **0.1361** | **+0.058** | 3/3 正向 |

**判定：** `strong_evidence__consistent_gain`（协议内）。  
**可写：** 在 K2-C 正式协议下，A4 相对 A2 三种子一致提升。  
**不可写：** 显著提升、SOTA、泛化到任意融合栈。

---

## 6. 消融

### 6.1 残差缩放 P01（α=0.25）

唯一变量：`outs = x_proj + 0.25·FDPN(x)` vs A4 全量 FDPN。

| seed | A4 | P01 | Δ(P01−A4) |
|------|-----|-----|-----------|
| 42 | 0.1483 | 0.1155 | −0.033 |
| 43 | 0.1381 | 0.1139 | −0.024 |
| 44 | 0.1217 | 0.0592 | −0.063 |
| mean | 0.1361 | 0.0962 | **−0.040** · 0/3 正向 |

**判定：** 有效负结果 → **不**开 P02/P03 / learnable α。  
**解读：** 当前 early_concat 上，削弱残差路径会伤害收益；“再拧 α”会变成参数狩猎并稀释叙事。

### 6.2 融合 × FDPN 放置（P00，先验）

gated multiscale + FDPN 曾表现为负/无协同。结合 A4 正结果，说明收益是 **组合特异**（early_concat×FDPN），而非“FDPN 无论接在哪都好”。

| 组合 | 结论 |
|------|------|
| early_concat + FDPN（A4） | 保留为最终方法 |
| gated + FDPN（P00） | 放弃 |
| early_concat + FDPN·α=0.25（P01） | 放弃 |

---

## 7. Efficiency

细节与 caveats：`efficiency/EFFICIENCY_TABLE.md`。

| Arm | Params | GFLOPs | Latency mean | FPS | Infer peak VRAM |
|-----|--------|--------|--------------|-----|-----------------|
| A2 | 10.23M | 25.01 | 17.3 ms | 58.0 | 140 MB |
| A4 | 11.34M | 30.12 | 15.4 ms | 65.1 | 111 MB |

条件：RTX 5070 Ti · 640² · bs=1 · FP32 · **仅模型前向**（不含预处理/NMS）。  
A4 约 +11% 参数、+20% FLOPs；本机测得前向时延未变差。训练峰值显存（正式 20ep）A2≈836 MB、A4≈803 MB（均值）。

---

## 8. Limitations（答辩必说）

1. **数据规模与域：** RGBT-Tiny 子集 + 序列级划分；非公开完整 benchmark 全量对比。  
2. **统计：** 三种子、无置信区间、无显著性检验 → 不得写“显著”。  
3. **指标：** 主报 last@20；best 仅作辅助，避免挑选峰值叙事。  
4. **效率：** forward-only；非端到端部署 FPS。  
5. **结构依赖：** FDPN 增益不能外推到任意融合；P00 已给出反例。  
6. **未做：** 公开集大对比、蒸馏/量化部署、时序建模、更强双流骨干。

---

## 9. 结论（允许表述）

在 K2-C 冻结协议下，**A4（early_concat + FDPN）** 相对 **A2** 在三个随机种子上均取得一致的 last@20 增益（mean Δ ≈ +0.058）；将 FDPN 残差强度缩至 0.25 **无额外收益且全面下降**；gated 融合与 FDPN 的组合未显示协同。最终方法冻结为 A4；方法搜索关闭，进入打包与报告，不再调参。

---

## 10. 证据索引

| 文档 | 用途 |
|------|------|
| `FINAL_METHOD_FREEZE.json` | 代码 SHA / 数据 hash / exec / ckpt |
| `K2C_DATASET_FREEZE.json` | 数据冻结 |
| `FORMAL_A2_A4_V2_COMPARISON.*` | 主比较 |
| `FORMAL_A4_P01_V1_COMPARISON.*` | 残差消融 |
| `DECISION_ACCEPT_P01_NEGATIVE_FREEZE_A4.json` | 收敛决策 |
| `efficiency/EFFICIENCY_TABLE.*` | 效率表 |
| `FINAL_EXPERIMENT_SUMMARY.md` | 一页摘要 |
