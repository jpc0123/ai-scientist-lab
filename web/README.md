# Scientist Lab Web Console（怎么用）

## 最快上手（推荐）

1. 打开 PowerShell，进入仓库：

```powershell
cd D:\AI Scientist_tiao\scientist-lab
powershell -ExecutionPolicy Bypass -File .\scripts\start_web_console.ps1
```

2. 浏览器打开：**http://127.0.0.1:5173**
3. 点左侧 **「使用指南」**，或总览页 **「生成演示补丁」**

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

## 第一次建议点哪里

1. **总览** → 生成演示补丁  
2. **补丁** → 打开刚生成的条目  
3. 按顺序：批准 → 应用到沙箱 → 沙箱测试 → 记录证据 → 合并意图  

合并意图**不会**改你的主代码目录。

## 构建后由后端托管

```powershell
cd web
npm run build
cd ..
.\.venv\Scripts\scientist-lab.exe serve --port 8787
```

然后打开 http://127.0.0.1:8787
