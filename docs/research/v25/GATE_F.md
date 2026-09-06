# Gate F — RGBT-Tiny 真实数据注册

Gate E（`V25_GATE_E_DEBUG40`）已通过。本门禁**只注册 + 探针**，不训练、不跑 A4/P01。

## 输入

| 项 | 值 |
|----|----|
| 数据集 | **RGBT-Tiny** |
| 配置路径 | `D:\datasets\RGBT-Tiny\raw` |
| 实际根 | 解压后自动探测（可能是 `raw\RGBT-Tiny\`） |
| 注册输出 | `datasets/registered/rgbt_tiny_v1/` |

申请下载：[Google Forms](https://forms.gle/EeRooNEYzXXporQt9) · [Microsoft Forms](https://forms.cloud.microsoft/r/nN7JmKn4eJ) · [项目页](https://github.com/XinyiYing/RGBT-Tiny)

当前状态：**`waiting_for_raw`**（`raw` 目录已创建，内容为空）。

## 两段执行

### F0 — 完整注册（默认子集，不训练）

```text
原始目录识别 → RGB/Thermal 序列配对 → 标注解析 → 统一 COCO
→ 类别 0-based 映射 → 序列级 train/val/test → 质量审计 → DATASET_FREEZE.json
```

默认子集（防磁盘/内存打满）：

- `max_sequences=12`
- `max_frames_per_seq=40`
- `frame_stride=5`
- `seed=42`

全量（确认磁盘后）：`python scripts/register_rgbt_tiny_gate_f.py f0 --full`

**强制序列级划分**，禁止按帧随机切分。

路径策略：原图只保留一份；manifest 存绝对路径；`images/` 仅 hardlink/symlink；**禁止** raw 副本 / converted 全图副本 / Docker 内副本 / 实验目录拷图。

### F1 — 小规模只读探针

从冻结数据抽 train 40–100 / val 15–30：

manifest → 解码 → 对齐 → bbox/类别 → dataloader 单批 → CUDA 单批 forward

不保存正式科研指标。

## 审计字段

配对缺失、尺寸不一致、重复 sample_id、非法/越界 bbox、未知类、空标注帧、每类实例、目标尺寸分布、序列级 split 计数、跨划分序列泄漏。

## 命令

```powershell
# 解压后确认树
Get-ChildItem "D:\datasets\RGBT-Tiny\raw" -Depth 2 | Select-Object FullName

cd "D:\AI Scientist_tiao\scientist-lab"
.\.venv\Scripts\python.exe scripts\register_rgbt_tiny_gate_f.py f0
.\.venv\Scripts\python.exe scripts\register_rgbt_tiny_gate_f.py f1
.\.venv\Scripts\python.exe scripts\register_rgbt_tiny_gate_f.py status
```

## 实现

| 文件 | 作用 |
|------|------|
| `experiment_apps/rgbt_detection_real/rgbt_tiny_register.py` | F0 发现/配对/COCO/审计/冻结 |
| `experiment_apps/rgbt_detection_real/rgbt_tiny_probe.py` | F1 只读 + CUDA 探针 |
| `scripts/register_rgbt_tiny_gate_f.py` | CLI |
| `tests/fixtures/rgbt_tiny_mini/` | 布局夹具（F0/F1 已本地验证） |

备选：若暂时拿不到 RGBT-Tiny，可用 LLVIP 验证注册器，但不能替代最终小目标多类实验。
