# Gate J（草案）— A2 吞吐探针协议

> **状态：** `DRAFT` · **claim_authority：** `engineering_probe`  
> **目的：** 在不改动 Gate I 科学矩阵的前提下，诊断/缓解 GPU 利用率偏低（观测约 16–34%，`batch=1`）。  
> **禁止：** 用本门禁结果替换 Gate I 的 mAP 正式候选；禁止顺带开 A4/P01。

## 1. 动机

Gate I / H 冻结：`batch_size=1` · `num_workers=0` · 640 · DFINE-S。  
实测：`data_time≈0.02s` ≪ `iter_time≈0.40s` → **主要不是 DataLoader 饿死**，而是小 batch 填不满 5070 Ti。  
本门禁回答：**在相同数据与模型下，增大 batch / workers 能否提升 GPU util 与 samples/s，且不破坏训练可跑性。**

## 2. 门禁定义

| 项 | 值 |
|--|--|
| Gate ID | `GATE_J_A2_THROUGHPUT_PROBE` |
| Arm | A2 / `early_concat` only |
| Dataset / split | **复用** Gate H `v1_gate_h_expand`（1830/500/300） |
| Seed | **仅 42**（吞吐不要求三种子） |
| Epochs | **短预算：3**（或 1 个完整 epoch 的 timed segment + 可选 3ep） |
| Image / queries | 640×640 · 300 · 与 I 相同 |
| Docker | `scientist-rgbt-detection:v2-cuda` |
| Primary metric（工程） | `images_per_second`, `gpu_util_avg`, `peak_gpu_memory_mb`, `data_time_avg`, `iter_time_avg` |
| Secondary（健康） | 最终或末 epoch `mAP50_95` 非退化（仅 smoke，不作正式对比） |

## 3. 实验矩阵（建议）

相对 Gate I 基线（B0），**每次只改吞吐相关变量**：

| ID | batch_size | num_workers | 备注 |
|----|------------|-------------|------|
| B0 | 1 | 0 | 对照（可引用 I-seed42 的资源字段，或重跑 3ep） |
| T1 | 2 | 0 | 测 batch |
| T2 | 4 | 0 | 测 batch；OOM 则记失败并停止上探 |
| T3 | 2 | 2 | batch+workers |
| T4 | 4 | 2 | 最大探针；OOM/不稳则降级 |

可选后续（本草案默认不做）：`persistent_workers`、TF32、`torch.compile` —— 另开子条款。

## 4. 不变量（相对 Gate I）

保持不变：
- fusion=`early_concat` · pretrained · lr · AMP · image size · queries=300 · category map · 子集 · 代码 SHA / Docker tag（开跑时冻结）

允许改变：
- `batch_size` · `num_workers` · `epochs`（短预算） · `protocol` 字段名

## 5. 通过 / 失败标准

**Engineering pass（单配置）**
- 训练 `trained=true` · `device=cuda`
- 无 OOM；pair audit ok
- 写出 `resource_usage` / 逐步 timing（至少 epoch0 的 mean `data_time`/`iter_time`）

**Throughput useful（相对 B0）**
- `images_per_second` 提升 ≥ **1.5×**，或  
- 平均 GPU util 提升 ≥ **+20 个百分点**，且  
- 峰值显存 < 11 GiB（留余量给桌面/WDDM）

**Decision**
- `adopt_batch_for_future_gates`：选定一个 (batch, workers) 作为 **后续门禁默认**（需新协议冻结；**不回溯改 Gate I**）
- `keep_batch1_science_default`：吞吐有收益但科学矩阵仍锁 batch=1（两套配置：科学 vs 工程）
- `oom_or_unstable`：记录上限，停止上探

## 6. Claim 边界

**可写**
- “在相同 A2/子集下，batch=K 时 GPU util / throughput 为 …”
- “未来非 Gate I 实验可采用 batch=K 作为工程默认”

**不可写**
- “batch=K 的 mAP 优于 Gate I 正式候选”（短预算 + 单 seed 不够）
- 任何 A4/P01/融合方法结论

## 7. 运行产物（建议路径）

```
outputs/experiments/v25_real_rgbt/gate_j_a2_throughput_probe/
  PROTOCOL_DRAFT.json
  contracts/B0.json … T4.json
  THROUGHPUT_MATRIX.json
  GATE_J_REPORT.md
```

## 8. 与 Gate I 的关系

```
Gate I (science, batch=1, seeds 42/43/44)  --冻结--> A2 formal candidates
                         \
                          \--并行/之后--> Gate J (engineering throughput)
```

Gate J **不得**与 Gate I 抢同一 GPU；须等 I 的 seed44 结束或用户明确暂停后再跑。

## 9. 开跑检查清单（执行前）

- [ ] Gate I 已结束或用户批准占用 GPU  
- [ ] Worker `8081` healthy · 镜像 tag 与 I 一致  
- [ ] 写入 `PROTOCOL_FREEZE`（code SHA + 矩阵）  
- [ ] 明确：mAP 只做 smoke，不进正式 baseline 表  

---
*协议草稿；正式开跑前将本文件提升为 `GATE_J_OPEN.json` + 可执行 contracts。*
