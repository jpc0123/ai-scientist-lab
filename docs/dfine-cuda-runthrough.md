# CUDA + Vendor DFINE 跑通指南（v2.3.7）

面向目标：在本机确认 **Vendor DFINE** 离线就绪，并在有 GPU 时按门禁跑通 Fast Eval。  
默认路径 **零 GPU / 零网络**；真实 CUDA 必须显式开门。

> **对外表述**：离线就绪 ≠ CUDA 已实跑；Fast Eval / path-open ≠ formal DFINE 优越性。

---

## 0. 一分钟离线演示（推荐先跑）

```powershell
cd "D:\AI Scientist_tiao\scientist-lab"
.\.venv\Scripts\python.exe scripts\demo_dfine_cuda_offline.py
```

期望：`overall=passed`，且 `gpu_used=false` / `formal_success=false`。

---

## 1. Doctor（离线 → GPU 深度）

```powershell
# 仅检查 pin / Dockerfile / 契约（不探测 Docker/GPU）
.\.venv\Scripts\scientist-lab.exe dfine-cuda-doctor --no-probe

# 探测 nvidia-smi、Docker 镜像、nvidia runtime
.\.venv\Scripts\scientist-lab.exe dfine-cuda-doctor
```

关注字段：

| 字段 | 含义 |
|------|------|
| `ok` | 离线硬错误是否为 0（缺 pin / Dockerfile 会失败） |
| `live_ready` | 是否具备实跑门禁（Docker + 镜像 + nvidia-smi） |
| `runtime.gpu` | v2.3.7 GPU 摘要（台数 / blockers） |
| `checks[].id=docker_nvidia_runtime` | 是否声明 nvidia runtime（推荐，非唯一路径） |

工作台总检也会带上离线 DFINE 项：

```powershell
.\.venv\Scripts\scientist-lab.exe system-doctor
```

查找 `id=dfine_cuda`。

---

## 2. 离线 dry-run 编排

```powershell
.\.venv\Scripts\scientist-lab.exe dfine-cuda-fast-eval --no-probe
.\.venv\Scripts\scientist-lab.exe dfine-cuda-formal-triad --no-probe
.\.venv\Scripts\scientist-lab.exe dfine-real-acceptance --no-probe
.\.venv\Scripts\scientist-lab.exe dfine-formal-path-gate project_rgbt_cuda
```

以上命令 **不会** 提交 GPU 训练。

---

## 3. 真机 CUDA（显式门禁）

前置：

1. `nvidia-smi` 正常  
2. 已构建 `scientist-rgbt-detection:v2-cuda`  
3. Docker 可拉起 GPU 容器  

```powershell
$env:RUN_REAL_DFINE_TESTS="1"
$env:RUN_REAL_CUDA="1"
# 可选：同时跑 rgb/thermal/fusion triad
# $env:ACCEPT_V23_REAL_TRIAD="1"
.\.venv\Scripts\python.exe scripts\accept_v23_real.py
```

或：

```powershell
.\.venv\Scripts\scientist-lab.exe dfine-real-acceptance --execute
```

成功也仍是 **exploratory**：`formal_success=false`，不得写成「formal DFINE 优越性已证明」。

---

## 4. 验收脚本

```powershell
python .\scripts\accept_v23.py          # 离线，应全绿
python .\scripts\accept_v23_real.py     # 默认 SKIP（exit 0）
```

规格：`docs/plans/第二十三步制作方案.md`。

---

## 5. 常见问题

| 现象 | 处理 |
|------|------|
| `live_ready=false` 但离线 `ok` | 缺镜像 / 无 GPU / Docker 未开；看 `runtime.gpu.blockers` |
| `docker_nvidia_runtime` warning | 安装 NVIDIA Container Toolkit，或确认 runner 的 `--gpus` 路径可用 |
| Claim 仍是 `unsupported` / path 未开 | Fast Eval 单次不够 triad；优越性本就不该由 Fast Eval 支撑 |
| stand-in → `blocked` | 预期：stand-in 证据不得支撑 formal DFINE Claim |
