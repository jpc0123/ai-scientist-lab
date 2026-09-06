# 升级说明

## 推荐路径（Windows）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\upgrade_windows.ps1
```

或重新运行安装脚本后再启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1
```

## 升级前

1. `scientist-lab workbench stop`（或 stop 脚本）
2. 备份数据库路径（见 `config/scientist-lab.yaml` / health 返回的 `db_path`）
3. `git status` 确认本地改动

## 升级后

1. `scientist-lab system-doctor`
2. 打开前端确认版本（health `version`，如 `v2.0.8`）
3. 如有 interrupted 执行：`scientist-lab recover --dry-run`

## 版本基线

- Tag 基线：`v1.9.0`
- 工作台分支：`feat/v2.0-scientist-workbench`
- 子版本进度见 `docs/plans/第二十步制作方案.md`
