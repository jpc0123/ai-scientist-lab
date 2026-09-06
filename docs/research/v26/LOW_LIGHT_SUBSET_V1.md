# low_light_subset_v1

日期：2026-08-19  
战役：v2.6 P2  
**不是官方 night / day 标签。** RGBT-Tiny 的 COCO / VOC / split 列表都没有 illumination 字段。

## 冻结规则

| 项 | 值 |
|----|----|
| slice_id | `low_light_subset_v1` |
| method | `rgb_mean_rec709_luminance_train_p25` |
| 亮度 | Rec.709 `Y = 0.2126 R + 0.7152 G + 0.0722 B`，RGB 图像逐像素均值 |
| 阈值 | **仅在 train split** 上取 25 分位（线性插值，与 numpy 默认一致） |
| 成员 | 任一 split 中 `Y <= threshold` 的 `sample_id` |
| 改切片 | **Protocol Amendment**。LLM / Planner / 本轮 Human Gate 都不能改 |

General Set 仍评 `APS` / `mAP50_95`。Primary 是 **`APS_lowlight`** = 该切片上的 COCO `AP_small`（area < 32²）。全集 APS 或 mAP 不能冒充 `APS_lowlight`。

## 命令

```text
scientist-lab freeze-lowlight-subset --rule-only
scientist-lab freeze-lowlight-subset
```

无图像时只冻规则哈希。有 `datasets/registered/rgbt_tiny_v1`（或 pairs CSV 指向的 RGB 文件）时写入 membership 哈希。

## 本机冻结结果（2026-08-19）

规则哈希：`6aa3cf0a444b0373239c81fec10443e8e067f38f71e4e41dcf57d4d888965c49`  
成员哈希：`e716c55469a51bc47b2e9b5983f2c02a7b2e3de901e654ebd4ee832a622c246f`  
train 阈值 Y ≤ **0.1773**（不是 val 上拟合的）。

| split | 全集 | 切片 | 比例 |
|-------|------|------|------|
| train | 3342 | 836 | 25.0% |
| val | 650 | 100 | 15.4% |
| test | 300 | 100 | 33.3% |

val/test 比例可以不同于 25%，因为阈值只在 train 上计算。完整名单：`LOW_LIGHT_SUBSET_V1_FREEZE.json`。
