# Scientist Lab Web Console（怎么用）

## 最快上手（推荐）

1. 打开 PowerShell，进入仓库：

```powershell
cd D:\AI Scientist_tiao\scientist-lab
powershell -ExecutionPolicy Bypass -File .\scripts\start_web_console.ps1
```

2. 浏览器打开：**http://127.0.0.1:5174**
3. 默认进入 **「实验闭环」**：打开本机 `.run/` / fixtures，一屏看 Protocol → Gate → Run → Evidence → Rubric → Memory → Next Plan

## 手动开两个窗口

**窗口 A（后端）：**

```powershell
cd D:\AI Scientist_tiao\scientist-lab
.\.venv\Scripts\scientist-lab.exe serve --host 127.0.0.1 --port 8787
```

**窗口 B（前端）：**

```powershell
cd D:\AI Scientist_tiao\scientist-lab\web
npm run dev
```

前端 Vite 在 **5174**（5173 留给其它本地项目）。`/api` 代理到 FastAPI `:8787`。

## 第一次建议点哪里

1. **实验闭环** → Formal C1 pack（0.0163 vs 0.0326）与 v2.5 probe pack（APS=0 不是声称）
2. 需要时再去 **总览** 生成演示补丁，或 **模型配置** 查看 Key（不要先 `--live`）  

合并意图**不会**改你的主代码目录。

## 构建后由后端托管

```powershell
cd web
npm run build
cd ..
.\.venv\Scripts\scientist-lab.exe serve --port 8787
```

然后打开 http://127.0.0.1:8787
