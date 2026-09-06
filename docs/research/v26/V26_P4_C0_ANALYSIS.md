# V26 P4 C0 analysis — RT-DETR F3 transfer is inconclusive

日期：2026-08-21  
父结果：[`V26_P4.md`](./V26_P4.md) · 裁决：[`DECISION_P4_C0_ANALYSIS.md`](./DECISION_P4_C0_ANALYSIS.md)  
数字：`outputs/v26_p4_analysis/summary.json`  
脚本：`docs/research/v26/p4_c0_analyze.py`（只读 pack，不训练）

**不是 G2。不是 G3。不是 transfer success。** ClaimGate 仍 C0。KEEP ≠ Claim。

## Question

F3 `gated_multiscale` 在 D-FINE 上相对 F1 有同方向涨幅。把它迁到 RT-DETR 后，还是不是同一套策略？

## Slice / 合同（未改）

- `dataset_id=rgbt_tiny_v1` + `slice_id=low_light_subset_v1`
- fingerprint `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6`
- Primary = `APS_lowlight`（切片 val 100 图 / 851 框，pycocotools）
- 两枪都是 seed=42，都只训了 **2 epoch**（可比，但噪声大）

## RT-DETR F3 − F1

| 指标 | F1 `v26_p4_r0` | F3 `v26_p4_r1` | Δ |
|------|----------------|----------------|---|
| **APS_lowlight** | 0.013241256515861267 | 0.014949471669213135 | **+0.0017082151533518684** |
| mAP50_95_lowlight | 0.008713823961250661 | 0.009865956567154483 | +0.001152 |
| Recall_small_lowlight | 0.11139240506329114 | 0.1291139240506329 | +0.017722 |
| AP50_lowlight | 0.04324188304470019 | 0.042318947943377504 | **−0.000923** |
| 全集 mAP50 | 0.12063575569253525 | 0.0954089184043112 | **−0.025227** |
| 全集 mAP50_95 | 0.02371525254237785 | 0.030909030434578412 | +0.007194 |

主指标和 AP50_lowlight / 全集 mAP50 **符号不一致**。

## 跨模型 Δ 对照（C0，不可宣称 matched C1）

同一 HOW、同一 seed=42：

| 检测器 | F1 → F3 `APS_lowlight` Δ |
|--------|--------------------------|
| D-FINE R0→R1 | +0.016764361110192725 |
| RT-DETR P4 | +0.0017082151533518684 |

RT-DETR 上的 Δ 大约是 D-FINE 的 **0.102**。不能把它读成「策略已迁移成功」。

## 成本

| | F1 | F3 | 比 |
|--|----|----|----|
| 参数量 | 9 795 496 | 17 154 500 | ×1.75 |
| peak GPU MB | 347.28 | 575.68 | ×1.66 |
| 墙钟 s | 898.30 | 1183.08 | ×1.32 |
| score≥0.1 检出 | 11296 | 13063 | 更高分框更多 |

F3 更贵，检出更「自信」，但 AP50_lowlight 和全集 mAP50 掉了。

## Outcome

`scientific_outcome=INCONCLUSIVE`。

主指标微涨不足以支撑跨模型策略迁移。下一步 GPU（多种子）不在 P4 `max_rounds=2` 内。
