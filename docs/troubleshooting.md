# 常见问题与排错

## 启动脚本报错

| 现象 | 原因 | 怎么做 |
|------|------|--------|
| `-ExecutionPolicy` 无法识别 | 漏写了开头的 `powershell` | 用完整命令：`powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1` |
| `字符串缺少终止符` | 旧脚本编码问题 | 更新仓库后再启；或用「手动分窗启动」（见 [installation-windows.md](./installation-windows.md)） |
| 路径找不到脚本 | 不在 `scientist-lab` 目录 | 先 `cd "D:\AI Scientist_tiao\scientist-lab"` |

干净重启：

```powershell
cd "D:\AI Scientist_tiao\scientist-lab"
powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1
```

## 前端显示「未连接」

1. 确认 API 在跑：`http://127.0.0.1:8787/api/v1/health`（`version` 应为 `v2.0.x`）
2. 确认 Web：`http://127.0.0.1:5173`
3. 用 `scripts\doctor_windows.ps1` 或 `.\.venv\Scripts\scientist-lab.exe workbench status`

## 端口被占用

修改 `config/scientist-lab.yaml` 中的端口，或先：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1
```

## Digits 跑失败 / Docker

`demo-create` 本身不跑实验。真实 `run` 需要 Docker 与 `digits-mlp` 相关镜像/环境。先跑：

```text
scientist-lab system-doctor
```

## 重启后实验停在 running

安全策略：**不会自动重跑昂贵实验**。应：

```text
scientist-lab recover --dry-run
scientist-lab recover --apply
```

无进程的 running 会被标为 `interrupted`。

## 演示项目已存在

```text
scientist-lab demo-create digits --force
```

或 API：`POST /api/v1/demo/create`，body `{"kind":"digits","force":true}`。

## 补丁不能改主目录

设计如此。只能沙箱应用；真正合入走 Merge Center（隔离 worktree），禁止任意 Shell / 自动 push。
