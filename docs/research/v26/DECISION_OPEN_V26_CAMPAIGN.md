# Decision — Open v2.6 LLM Autonomous Detection Campaign

日期：2026-08-18  
裁决：`branch`  
不解冻 MVP。tag `mvp-freeze-m1-m4-claimgate-c1` @ `130b02c` 只读。

## Verdict

开启 **v2.6 LLM Autonomous Detection Campaign**：低照度 RGB-T 小目标上的 Live LLM 多轮迭代，假设来自文献 + 实验 Evidence + Memory，并做一次跨模型策略迁移。LiteratureRetriever 是受控工具（Semantic Scholar 默认 Provider），不是新 Agent。不再把下一主线定为 Formal E、轨迹训练或继续扩 Web。

## Action

`branch`：从 v2.5-D Demo 验收点转入 v2.6。优先级 **P0 Live LLM API → P1 Semantic Scholar LiteratureRetriever → P2 低照度切片 → P3 3～5 轮 GPU → P4 transfer**。

## Reason

v2.5-D 只证明 LLM 能进入闭环。比赛更强的证明是：真实 LLM 在一个足够垂直、可验证的检测问题上连续根据反馈设计实验并取得可量化提升，且同一科研策略能经 Adapter 迁到第二个模型。低照度使 RGB-T 融合的科学问题变得可讲、可测。

## Rejected alternatives

- 只做 Live LLM smoke / 再证明一次「能进闭环」
- 先开独立 Formal E（把 v2.5-D probe 升格，或再做 C1 扩规模）
- 轨迹训练飞轮 / Bounded Tree / 继续扩 Web
- 让 Planner 继续选 neck/fusion/hyperparameter 菜单
- 把 A4 当作 LLM 已发现的方法来展示
- 一次上三个 detector，或把 FDPN 源码直接塞进第二模型
- 要求每一轮指标单调上涨
- 让 LLM 自己切低照度子集
- 为论文检索新建 Agent 群或知识图谱
- 用摘要级文献替代本机实验进 ClaimGate

## Evidence paths

- `docs/V25D_DEMO_ACCEPTANCE.md`
- `docs/research/v25/gate_k/FINAL_METHOD_FREEZE.json`（历史对照，默认对 Planner 隐藏）
- `docs/V26_LLM_AUTONOMOUS_DETECTION.md`
- [新版构建方案.md](../../../新版构建方案.md) §十四
- `设计架构.md`（Architecture Freeze，不改）

## Next direction

P2 已落地（`low_light_subset_v1` 规则冻结 + `APS_lowlight` 指标合同 + HOW 目录 F0/F1/F3/N0/N1）。Dataset Workspace 已作为 GPU 前的基础设施插入（`data/registry` + `data/slices` + Web `/data`）。**V26.4 R0 已 `metrics_bound`（2026-08-19）**：HOW=F1 + `low_light_subset_v1`；`APS_lowlight=0.0045926865160844455`（pycocotools 切片 val，非全集 APS/mAP，非 v2.5 / probe）。下一步 **V26.5**：同一 dataset/slice/fingerprint 家族上开 3～5 轮 Live LLM。
