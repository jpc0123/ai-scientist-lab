# Decision — Harvest only matching contract / run_id

日期：2026-08-28  
战役：v2.6  
裁决：`continue`（执行路径 fail-closed 修补）  
Human Gate：本决议不自动 resume GPU。KEEP ≠ Claim。

## Verdict

**`try_harvest_existing` 只能重连「本轮合同」对应的 scientist-exec，禁止盲吃任意 Exited 容器或最新 metrics。**

匹配主键：`node_id`（= run_id）。无 node_id 时退化为 seed + fusion/neck/input。  
去掉 `outputs/` 按 mtime 盲扫。dest 残留 metrics 仅当合同匹配才算本轮产物。

## Reason

Campaign `exp_rgbt_dfine_v26_lowlight_20260827T144442Z` round7+ 全部 `reattached=True` 复用 `exec_fb1a7078f954`（实为 F3@seed44），导致 APS_lowlight 假重复。根因是 harvest 为 orphan waiter 设计，却在每轮 execute 前无差别启用。

## Evidence paths

- `src/scientist_lab/runners/exec_reattach.py`
- `src/scientist_lab/adapters/dfine/cuda_runner.py`
- `tests/unit/test_exec_reattach.py`
- 本文件

## Next direction

重启后端后，新轮次必须 `containers.run`。历史 round7+ 指标视为无效，勿作 Claim。若续跑需抬 `max_extra_rounds` 并重训未真实执行的 HOW。
