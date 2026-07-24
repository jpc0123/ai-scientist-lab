# 手动跑通指南：规划 → 执行 → 迭代

面向目标：**接大模型规划实验，执行实验，再迭代**。  
本文默认先用 **Mock LLM** 把闭环跑通（不烧 Key）；文末说明如何换真模型。

---

## A. 开始前检查（5 分钟）

### A1. 启动工作台

1. 打开 PowerShell，先进入目录：

```powershell
cd "D:\AI Scientist_tiao\scientist-lab"
```

2. **整段复制**下面两行（先停再启，避免旧进程）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_workbench.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_workbench.ps1
```

注意：每行都以 `powershell` 开头，不要漏写。

3. 浏览器打开：http://127.0.0.1:5173  

顶部应显示已连接。更完整的启停说明见 [getting-started.md](./getting-started.md)。

### A2. 确认后端版本

打开：http://127.0.0.1:8787/api/v1/health  

应类似：

```json
{ "ok": true, "version": "v2.0.0", "docker_ok": true }
```

- `version` 必须是 **`v2.0.x`**（若是 `v1.7.1`，说明旧进程未停干净，重做 A1）
- Digits 真实训练需要 `docker_ok: true`

### A3. 页面布局

刷新前端（Ctrl+F5）。侧栏应独立滚动，主区内容不应压在侧栏上。

---

## B. 路径一：Digits 真实执行（推荐第一次）

目的：先确认「实验能跑、结果能看」。

### B1. 创建演示项目（Web）

1. 左侧 **项目**
2. 点 **Digits** / Digits 快速演示
3. 进入 `demo_digits_v20`（此时只有骨架，还没训练）

### B2. 跑实验（CLI）

同一台机器另开 PowerShell，进入 `scientist-lab`：

```powershell
.\.venv\Scripts\scientist-lab.exe run examples\digits_real_contract.json
```

可选对比组：

```powershell
.\.venv\Scripts\scientist-lab.exe run examples\digits_real_contract_02.json
```

成功标志：终端出现 `status` 为 `completed`，并有 `execution_id`。

### B3. Web 看结果

| 步骤 | 页面 | 做什么 |
|------|------|--------|
| 1 | **实验中心** | 打开刚完成的执行，看指标 / 日志 |
| 2 | **比较工作台** | 比较 `node_003` 与 `node_004`（若两组都跑了） |
| 3 | （可选）证据 / Claim | 基于比较构建证据 |

到这里：你已经完成「执行」半边。

---

## C. 路径二：规划 → 审批 → 合同 → 迭代（AI Scientist 主线）

目的：体验 **Planner / Critic / 人工审批 / 实验树**。  
本路径默认 **Mock LLM**（不连公网模型），但流程与真模型相同。

### C1. 准备 RGB-T 规划用项目素材

在 `scientist-lab`：

```powershell
.\.venv\Scripts\scientist-lab.exe demo-create rgbt-debug
```

然后（若尚未有协议节点，可用验收同款引导；最简方式是用已有 examples）：

```powershell
.\.venv\Scripts\scientist-lab.exe create-protocol examples\rgbt_protocol.json
```

说明：正式规划常用项目 `project_rgbt_003` + 协议 `protocol_rgbt_001` + 基线节点（见 examples）。  
若你本地库是空的，可按下面「一次性播种」做。

#### 一次性播种（复制执行）

```powershell
# 1) 协议
.\.venv\Scripts\scientist-lab.exe create-protocol examples\rgbt_protocol.json

# 2) 预算
.\.venv\Scripts\scientist-lab.exe set-budget project_rgbt_003 --max-new-nodes 5 --max-gpu-hours 10

# 3) 若没有 formal 基线节点，可先看 demo-rgbt 节点；规划验收常用 formal fusion contract。
#    最稳：用 Python/验收脚本逻辑；手动时也可只跑 plan-next 看系统是否返回 plan_id。
```

更省事：直接用 CLI 触发规划（服务会尽量解析协议）：

```powershell
.\.venv\Scripts\scientist-lab.exe plan-next project_rgbt_003 --protocol-id protocol_rgbt_001
```

记下输出里的 **`plan_id`**。

### C2. Critic 审查 + 排序 + 人工批准

把 `<PLAN_ID>` 换成上一步的 id：

```powershell
.\.venv\Scripts\scientist-lab.exe review-plan <PLAN_ID>
.\.venv\Scripts\scientist-lab.exe rank-candidates <PLAN_ID>
```

从 rank 输出里抄 **第一个 candidate_id**：

```powershell
.\.venv\Scripts\scientist-lab.exe approve-candidate <PLAN_ID> <CANDIDATE_ID>
.\.venv\Scripts\scientist-lab.exe generate-contract <PLAN_ID> <CANDIDATE_ID>
```

生成的 contract 路径一般在：

`outputs/<project_id>/plans/<plan_id>/..._contract.json`

### C3. Web 对照看规划

1. **规划中心**：应能看到刚生成的 plan  
2. **审批中心**：待批候选 / 相关条目  
3. 需要时在审批页点批准（与 CLI 二选一即可，不要重复乱批）

### C4. 实验树迭代（有限搜索）

若已有基线节点 `rgbt_formal_node_003` 与协议：

```powershell
.\.venv\Scripts\scientist-lab.exe tree-create project_rgbt_003 --root-node-id rgbt_formal_node_003 --protocol-id protocol_rgbt_001
```

记下 `tree_id`：

```powershell
.\.venv\Scripts\scientist-lab.exe tree-plan-next <TREE_ID>
.\.venv\Scripts\scientist-lab.exe tree-approve <TREE_ID> <TOP_CANDIDATE_ID>
```

这会创建 **Iteration**。  
真实环境里下一步是批准 Iteration 并执行；离线练习时可用 Web **迭代会话 / 实验树** 看状态。

推进：

```powershell
.\.venv\Scripts\scientist-lab.exe tree-advance <TREE_ID>
```

Web：**实验树** 打开该树，看结构与状态。

### C5. 证据与报告（可选）

有比较结果后：

```powershell
.\.venv\Scripts\scientist-lab.exe build-evidence --baseline-node-id <A> --candidate-node-id <B>
.\.venv\Scripts\scientist-lab.exe build-claim-matrix <PROJECT_ID>
.\.venv\Scripts\scientist-lab.exe report-build <PROJECT_ID>
```

Web：**证据 / Claim / 报告**。

---

## D. 你真正关心的闭环（对照表）

```text
1. 规划     plan-next（Mock 或真 LLM）
2. 审查     review-plan（Critic）
3. 人工闸门 approve-candidate / 审批中心
4. 出合同   generate-contract
5. 执行     scientist-lab run <contract.json>   ← 当前 Web 无一键跑
6. 看结果   实验中心 / 比较
7. 再规划   再 plan-next，或 tree-plan-next / tree-advance
8. 证据     Evidence / Claim / Report
```

**没有跑偏**：这就是设计好的 AI Scientist 环。  
缺的是「Web 一键执行」和「默认真模型」，不是方向错了。

---

## E. 换成真实大模型（有 Key 时再开）

1. 准备 OpenAI-compatible 的 `API_KEY` / `BASE_URL`（按你本地 Provider 文档）  
2. 确认配置里 **不要** 默认强制 Mock（见 `config/scientist-lab.yaml` 的 `llm` / `security`）  
3. 规划时显式指定真实 provider，例如：

```powershell
.\.venv\Scripts\scientist-lab.exe plan-next project_rgbt_003 --protocol-id protocol_rgbt_001 --provider openai
```

（具体 provider 名以你环境已注册的为准；失败时先看报错是否缺 Key / 未启用网络。）

原则：

- 真模型 **显式启用**
- 实验仍要 **人工批准** 后才执行
- 不会因为重启就自动重跑昂贵实验

---

## F. 建议你今天手动验收的最短清单

打勾即可：

1. [ ] health = `v2.0.0`，页面不重叠  
2. [ ] Digits demo 创建成功  
3. [ ] `run digits_real_contract.json` 完成  
4. [ ] 实验中心能打开该执行  
5. [ ] `plan-next` 得到 `plan_id`（Mock 即可）  
6. [ ] `review-plan` → `approve-candidate` → `generate-contract` 成功  
7. [ ] 规划中心 / 审批中心能看到对应记录  

完成 1–4：执行半边 OK。  
完成 5–7：规划半边 OK。  
两边都 OK：**主线闭环可继续接真模型。**

---

## G. 常见卡点

| 现象 | 处理 |
|------|------|
| System Doctor / 接口 not_found | 后端仍是旧版，重做 A1 |
| Digits run 失败 | 开 Docker Desktop，再 `system-doctor` |
| plan-next 说缺节点/协议 | 先 `create-protocol`，或先跑 Digits/RGB-T demo |
| 前端重叠 | 硬刷新；确认已拉到最新 `web` 构建 |

有问题把：**health JSON**、**失败的那条 CLI 完整输出** 发出来即可继续排。
