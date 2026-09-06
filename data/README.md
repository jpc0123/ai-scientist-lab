# Dataset Workspace

Scientist Lab 的数据基础设施层，**不是第五个 Agent**。

```text
data/
  raw/          原始数据指针区（图像本体不进 Git）
  processed/    训练格式物化区（当前 RGBT 仍用 datasets/registered/）
  slices/       冻结样本名单（进 Git）
  registry/     Dataset Contract（进 Git）
  cache/        派生缓存（不进 Git）
```

权限：

| 角色 | 可以 | 不可以 |
|------|------|--------|
| Planner | 选 `dataset_id` / `slice_id` | 改标签、改切片成员 |
| Adapter | 把 Contract 转成模型 loader 配置 | 偷换 train/val |
| Runner | 只读冻结数据 | 改 registry |
| Web `/data` | 绑定 / **重绑本机路径**、启用禁用 | 改 dataset_id / fingerprint / 切片名单 |

跨模型（D-FINE / RT-DETR / YOLO）应消费同一个 `dataset_id` + `slice_id`。

**路径可换，身份不可静默换：** 同一 `dataset_id` 可 `rebind` 本机目录；换数据集须 Protocol Amendment。

## CLI

```text
scientist-lab dataset-workspace
scientist-lab dataset-resolve --dataset-id rgbt_tiny_v1 --slice-id low_light_subset_v1
scientist-lab dataset-import-slice --from-freeze docs/research/v26/LOW_LIGHT_SUBSET_V1_FREEZE.json
```

Web 控制台：`/data`（查看、绑定 SQLite、不能编辑切片名单）。
