# 第一个科研项目

## 路径 A：内置演示（最快）

1. 启动工作台，打开「项目」
2. 创建 **Digits** 或 **RGB-T Debug**
3. 进入项目详情，查看协议与节点
4. Digits（可选真实跑）：

```text
scientist-lab run examples/digits_real_contract.json
```

5. 在「比较」对比节点 →「证据 / Claim」→「报告」

RGB-T Debug **不要求**真实 CUDA + DFINE；用于走通双模态协议、Smoke/Fast Eval 结构与 Claim Gate。真实检测需另配镜像与数据。

## 路径 B：六步向导

1. 「项目」→「新建项目」
2. 科研问题 → 任务类型 → 数据集 → 运行环境 → 协议 → 确认
3. 状态进入 `ready` 后可规划 / 执行 / 审批

状态机：

```text
draft → configuring → ready → running → reviewing → completed → archived
```

## 推荐菜单顺序

| 区 | 做什么 |
|----|--------|
| 总览 | 看待办与失败 |
| 项目 | 生命周期入口 |
| 实验 | 筛选执行、看详情 |
| 规划 / 审批 | Plan / Candidate / Iteration |
| 实验树 | 有限搜索树推进 |
| 证据 | Evidence / Claim Matrix |
| 报告 | Research Report / Audit |
| 系统 | Doctor / Recover |

## CLI 对照

```text
scientist-lab project-list
scientist-lab project-show <project_id>
scientist-lab demo-create digits
scientist-lab system-doctor
scientist-lab recover --dry-run
```
