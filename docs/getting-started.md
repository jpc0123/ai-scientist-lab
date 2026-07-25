# 快速开始（约 10 分钟）

Scientist Lab v2.0 是**单用户、本地优先、人工审批**的 AI Scientist 工作台。

| 组件 | 默认地址 | 作用 |
|------|----------|------|
| 前端（网页） | http://127.0.0.1:5173 | 看项目、审批、看结果 |
| 后端（API） | http://127.0.0.1:8787 | 真正干活的服务 |

网页会通过 `/api` 自动连到后端，一般你只需要打开 **5173**。

---

## 0. 每次操作前：先进入目录

打开 **PowerShell**，执行：

```powershell
cd "D:\AI Scientist_tiao\scientist-lab"
```

后面所有命令都在这个目录下跑。  
（若你的仓库不在 `D:\AI Scientist_tiao`，改成你的实际路径。）

---

## 1. 安装（只需做一次）

从未装过，或怀疑环境坏了，再跑：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
```

会准备 `.venv`（Python）和 `web/node_modules`（前端）。

---

## 2. 启动 / 重启工作台（日常最常用）

### 推荐：先停再启（干净重启）

**整段复制**到 PowerShell（两行都要）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1
```

### 注意（容易踩坑）

1. **必须写完整**：以 `powershell` 开头，不要只写 `-ExecutionPolicy ...`
2. **目录要对**：当前目录应是 `scientist-lab`（能看到 `scripts` 文件夹）
3. 启动成功后，脚本通常会尝试打开浏览器；若没有，手动打开：  
   **http://127.0.0.1:5173**

### 怎样算启动成功？

1. 浏览器打开 http://127.0.0.1:5173 ，顶部显示已连接（不是「未连接」）
2. 再打开 http://127.0.0.1:8787/api/v1/health ，应看到类似：

```json
{ "ok": true, "version": "v2.1.0", "docker_ok": true }
```

- `version` 必须是 **`v2.1.x`**
- 若仍是 `v2.0.0` / `v1.7.1`：旧进程没停干净，再跑一遍上面的「先停再启」

### 只停止（不启动）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1
```

---

## 3. 创建演示项目

**网页**：左侧「项目」→ **Digits 快速演示** 或 **RGB-T Debug 演示**。

**命令行**（可选）：

```powershell
.\.venv\Scripts\scientist-lab.exe demo-create digits
.\.venv\Scripts\scientist-lab.exe demo-create rgbt-debug
```

演示只建项目骨架，**不会自动跑昂贵实验**。

---

## 4. 走通一条受控路径（可选）

1. 总览 →「生成演示补丁」
2. 补丁详情：批准 → 沙箱应用 → 测试 → 证据 → 合并意图
3. （进阶）Merge Center：Prepare → … → Finalize

---

## 5. 下一步

- **手动跑通（规划→执行→迭代）**：[manual-runthrough.md](./manual-runthrough.md)
- **CUDA + Vendor DFINE**：[dfine-cuda-runthrough.md](./dfine-cuda-runthrough.md)
- 完整流程：[first-project.md](./first-project.md)
- 安装细节：[installation-windows.md](./installation-windows.md)
- 排错：[troubleshooting.md](./troubleshooting.md)
- 安全边界：[security.md](./security.md)
- 升级：[upgrade.md](./upgrade.md)
