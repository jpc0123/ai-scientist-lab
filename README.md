# scientist-lab

可追踪、可复现的科研实验执行底座（第一版 CLI 垂直闭环）。

## 结构

```text
Project → ExperimentNode → ExecutionAttempt → Artifact
```

## 快速开始

```powershell
cd scientist-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 构建 CPU 模拟实验镜像
docker build -t scientist-experiment:v1 docker

# 跑 smoke 验收
scientist-lab run examples/smoke_test_contract.json

# 查询执行记录
scientist-lab list-executions
scientist-lab show-execution <execution_id>
```

## 验收目标

一条 CLI 命令应完成：创建 Node → 创建 Attempt → 本地 CPU Docker → 生成 metrics / manifest → SQLite 持久化 → 重启可查询。
