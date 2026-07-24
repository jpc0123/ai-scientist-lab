# Windows 安装与启停

## 前置条件

- Windows 10/11
- Python 3.11+（推荐）
- Node.js 18+（前端）
- （可选）Docker Desktop：跑 Digits 真实训练时需要

---

## 一键安装（只需一次）

```powershell
cd "D:\AI Scientist_tiao\scientist-lab"
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
```

脚本会创建/复用 `.venv`、安装 Python 包，并在 `web/` 下执行 `npm install`。

---

## 日常：启动与停止

所有命令都在 `scientist-lab` 目录下执行。

### 干净重启（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1
```

### 只启动

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1
```

### 只停止

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1
```

### 命令写法提醒

| 正确 | 错误 |
|------|------|
| `powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1` | 漏写开头的 `powershell` |
| 先 `cd` 到 `scientist-lab` | 在别的目录执行相对路径 |

启动后打开：**http://127.0.0.1:5173**  
健康检查：**http://127.0.0.1:8787/api/v1/health**（`version` 应为 `v2.0.x`）

仓库根目录若有 `START_WORKBENCH.ps1`，也可从根目录调用。

---

## 诊断

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\doctor_windows.ps1
.\.venv\Scripts\scientist-lab.exe system-doctor
```

---

## 手动分窗启动（脚本不好用时）

窗口 1 — 后端：

```powershell
cd "D:\AI Scientist_tiao\scientist-lab"
.\.venv\Scripts\scientist-lab.exe serve --host 127.0.0.1 --port 8787
```

窗口 2 — 前端：

```powershell
cd "D:\AI Scientist_tiao\scientist-lab\web"
npm run dev
```

再打开 http://127.0.0.1:5173 。

---

## 配置

默认配置见 `config/scientist-lab.yaml`（端口、数据目录等）。修改后需重启工作台生效。
