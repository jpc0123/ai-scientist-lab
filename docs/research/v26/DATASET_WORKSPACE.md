# Dataset Workspace（v2.6 插入步）

日期：2026-08-19  
性质：**基础设施**，不是第五个 Agent，也不是新的科研闭环。  
位置：P2 低照度切片之后、P3 GPU 战役之前。

## 为什么现在做

代码仓和数据仓必须分开。D-FINE / RT-DETR / YOLO 不能各自维护一份 `data/`，否则跨模型比较会被质疑成数据条件不一致。

Scientist Lab 真正消费的是：

`dataset_id` + `slice_id` + `dataset_fingerprint`

而不是 `D:\xxx\xxx\train`。

## 四层

| 层 | Git | 含义 |
|----|-----|------|
| `data/raw/` | 不进（只留 `.gitkeep`） | 原始数据指针，只读 |
| `data/processed/` | 不进 | 训练格式物化；当前 RGBT 仍指向 `datasets/registered/rgbt_tiny_v1`，不复制 8000+ 图 |
| `data/slices/` | 进（`*_ids.txt` + `slice_spec.json`） | 冻结样本名单，不复制整份数据 |
| `data/registry/` | 进 | Dataset Contract 身份 |

图像 / 标签 / checkpoints / `.run/` / `outputs/` 继续不进普通 Git。

## 权限

- Planner 可选 `dataset_id` / `slice_id`，不能改标签
- Adapter 把 Contract 译成模型 YAML / loader，不能偷换 split
- Runner 只读冻结数据，只写 checkpoint / logs / predictions
- Web `/data` 可查看、绑定/重绑本机 `host_path`、启用/禁用；**不返回切片成员名单**；**不能改 dataset_id / fingerprint**

改 `low_light_subset_v1` 仍然是 Protocol Amendment。

### 路径重绑 vs 换数据集

| 动作 | 允许？ | 说明 |
|------|--------|------|
| 同一 `dataset_id` 更换本机挂载路径 | ✅ | `POST /dataset-workspace/bind` + `host_path` + `rebind=true` |
| 战役中途换 `dataset_id` / slice / fingerprint | ❌ | `dataset_change: forbidden`；须 Protocol Amendment |
| Human Gate 批准绕过换数据集 | ❌ | 路径重绑 ≠ 换数据集 |

## 命令

```text
scientist-lab dataset-workspace
scientist-lab dataset-resolve --dataset-id rgbt_tiny_v1 --slice-id low_light_subset_v1
scientist-lab dataset-import-slice --from-freeze docs/research/v26/LOW_LIGHT_SUBSET_V1_FREEZE.json
```

当前冻结切片：`low_light_subset_v1`（非官方 night 标签；Rec.709 亮度，train P25）。
