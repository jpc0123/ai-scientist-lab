# V26.4 R0 Baseline Anchor

日期：2026-08-19  
战役：v2.6  
性质：**比较锚**，不是 LLM 发现，不是 Formal E，不是 v2.5-D probe 升格。

## 冻了什么

| 项 | 值 |
|----|----|
| anchor_id | `v26_r0_dfine_f1_lowlight` |
| HOW | **F1** `early_concat` + **standard neck** |
| 不是 | A4（F1+N1 / FDPN）、F0、F3、F2/T1/T2 |
| dataset | `rgbt_tiny_v1` |
| slice | `low_light_subset_v1`（非官方 night 标签） |
| fingerprint | `787c65bda4eb31a700e67fba7804f1e7e17079574d24c02399f74bb722d27ea6` |
| primary | `APS_lowlight` = 切片 val 上 COCO `AP_small` |
| run_level | `budget_class=formal` / `full_train`（禁止用 16/8 smoke 填 G2） |
| 状态 | **`metrics_bound`**（2026-08-19） |

比较合同：[`R0_METRIC_CONTRACT.json`](./R0_METRIC_CONTRACT.json)  
Protocol：`schemas/examples/research_protocol_rgbt_dfine_v26.json`  
Plan：`schemas/examples/experiment_plan_rgbt_dfine_v26_r0.json`

## 已绑定指标（事实）

Human Gate：`v26-r0-run --execute --confirm-human-gate --require-live-ready`（Docker 已开，`live_ready=true`）。

| 项 | 值 |
|----|----|
| 运行目录 | `outputs/v26_r0` |
| execution_id | `exec_31eff20c4e0c` |
| 镜像 | `scientist-rgbt-detection:v2-cuda` |
| 设备 | CUDA（RTX 5070 Ti Laptop） |
| staged | train 3342 / val 650（不是 16/8） |
| HOW 实测 | `fusion_method=early_concat`，`neck_type=standard` |
| **APS_lowlight** | **0.0045926865160844455** |
| mAP50_95_lowlight | 0.003022687959416924 |
| AP50_lowlight | 0.019216142380591137 |
| evaluator | pycocotools，切片 val 100 图 / 851 标注 |
| 全集 AP_small（补充，不是 primary） | 0.032561101341476786 |
| 全集 mAP50_95（补充，不是 primary） | 0.03102824450668555 |

`APS_lowlight` 来自冻结切片上的 COCO `AP_small`，**没有**复制全集 `AP_small` / `mAP50_95`，也没有使用 v2.5 A4 或 probe APS=0.0。切片评测记录：`outputs/v26_r0/aps_lowlight.json`。冻结文件：[`R0_BASELINE_FREEZE.json`](./R0_BASELINE_FREEZE.json)。

## 明确拒绝

- 把 v2.5 A4 的 `mAP50_95 last@20` 写成 `APS_lowlight`
- 把 v2.5-D probe `APS=0.0` 写成 R0
- 用全集 APS / mAP 冒充切片指标
- 让 LLM 改切片或把 A4 当 Planner 答案

## 命令

```text
scientist-lab freeze-v26-r0
scientist-lab freeze-v26-r0 --no-probe
scientist-lab v26-r0-run --output-dir outputs/v26_r0
scientist-lab v26-r0-run --output-dir outputs/v26_r0 --execute --confirm-human-gate --require-live-ready
scientist-lab freeze-v26-r0 --bind-metrics outputs/v26_r0
```

`--execute` 必须同时带 `--confirm-human-gate` 和 `--require-live-ready`。这是战役一次性 Human Gate，不是第五个 Agent。`live_ready=false` 时拒绝 GPU，也拒绝伪造指标。

## 何时才能开 V26.5

1. `dfine-cuda-doctor` 的 `live_ready=true` — **已满足**
2. 上面的 R0 GPU 命令跑完，有有限的 `APS_lowlight` — **已满足**
3. `freeze-v26-r0 --bind-metrics <run_dir>` 把状态写成 `metrics_bound` — **已满足**
4. 之后 Live LLM 多轮必须绑同一 `dataset_id` + `slice_id` + fingerprint 家族
