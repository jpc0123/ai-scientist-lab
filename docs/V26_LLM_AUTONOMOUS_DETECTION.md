# v2.6 LLM Autonomous Detection Campaign

日期：2026-08-18  
分支：`feat/post-mvp-v25`（或后续 `feat/v26-llm-autonomous-detection`）  
不解冻 MVP。tag `mvp-freeze-m1-m4-claimgate-c1` @ `130b02c` 只读。  
不改 [设计架构.md](../../设计架构.md) §0–§17 冻结正文。HOW 插件 / 文献工具 / Dataset Workspace 已按 Freeze 规则 **追加** 到架构 §18–§19，不是第五个 Agent，也不是文献 Agent 群。

战略配对：[新版构建方案.md](../../新版构建方案.md) §十四～十五。  
决策：[`docs/research/v26/DECISION_OPEN_V26_CAMPAIGN.md`](./research/v26/DECISION_OPEN_V26_CAMPAIGN.md)

---

## 0. 为什么现在开这一仗

v2.5-D 已证明：**LLM 决策能进入 Scientist Lab 真路径，Gate / Rubric / ClaimGate 仍在。**  
那只是工程闭环。下一阶段不再做 Live LLM smoke，也不先做 Formal E / 轨迹训练 / 继续扩 Web。

比赛版研究目标收紧为四件事同时成立：

1. **大模型真的在做科研决策**（Live LLM 根据 **文献 + 实验 Evidence + Strategy Memory** 提出假设，而不是选 A/B/C 菜单，也不是只靠模型参数里的记忆）
2. **科研迭代真的带来可量化指标提升**（允许中间负结果；禁止要求单调上涨）
3. **协议 / Adapter 能跨检测模型复用**（同一 Plan 语义，HOW 由各模型 Adapter 实现）
4. **假设可追溯到真实论文检索**（每轮留下 query / paper_refs / used_by_plan；文献不能替代本机实验 Evidence）

A2 / A4 / P01 是 **历史研究知识与 benchmark**，**禁止**包装成「LLM 自己发现的」。A4 对 Planner **默认隐藏**。

---

## 1. 科研问题（垂直一层）

**问题：** 在 RGB 视觉质量下降的低照度场景中，AI Scientist 能否通过自主实验找到更有效的 RGB-T 融合策略，从而提高小目标检测性能？

因果链（讲给评委听的故事，不是声称已经证明）：

```text
低照度 → RGB 退化 → Thermal 具有互补信息
  → 融合是否、何时、如何发生，变得可检验
  → 主指标落在 low-light small-object AP，而不是泛泛 mAP
```

不是「怎样提高目标检测」。

---

## 2. 比赛定位（升级叙事，不改 MVP）

**MVP 仍是：** 可控自主实验系统。

**比赛版可以写成：**

面向低照度 RGB-T 小目标检测的 LLM 驱动自主科研系统：大模型根据 **受控文献检索 + 真实实验 Evidence + Strategy Memory** 提出假设和实验计划；Scientist Lab 在 Protocol、Human Gate、Adapter、Rubric、Literature Evidence Gate 与 ClaimGate 约束下执行真实 GPU 实验；失败后可定向再检索文献并修正假设。验证其在 D-FINE 上的自主优化能力，并测试科研策略向另一检测模型迁移的通用性。

人机边界不变，只多一个受控工具：

```text
Human 冻结科研宪法
LiteratureRetriever 按需取 PaperRecord（不是 Agent）
LLM 自主提出 WHAT / WHY
Adapter 控制 HOW
Gate 控制能不能跑
Rubric 控制 KEEP / DISCARD
ClaimGate 只看 ExperimentEvidence
```

---

## 3. 三个战役目标（只保留这些）

| ID | 目标 | 停线 |
|----|------|------|
| **G1** | Live LLM 完成 ≥3 轮真实 Evidence-driven 实验迭代；使用文献的轮次可追溯 provenance | mock LLM / 无 paper_refs 的「我看过论文」不算 |
| **G2** | 最终 `APS_lowlight` 明确高于冻结的 D-FINE baseline | 禁止把 A4 数字冒充为本战役 LLM 成果 |
| **G3** | 将最终 strategy transfer 到第二 detector，同方向指标提升 | 不要求同一 Python 模块塞进两个模型 |

成功标准（全部满足才算战役成功；否则 stop / 收窄，不无限刷）：

1. ≥3 个 **LLM-generated** experiment rounds（Live API，不是 FakeProvider）
2. 至少一次真实负结果导致下一轮策略改变（DISCARD → 可追踪改向）
3. 每轮都有：`evidence_refs` · hypothesis · candidate_experiments（2～4）· selected · `decision_summary` · `memory_refs`
4. 使用了文献的轮次必须有 `literature_query_id`、`queries`、`paper_refs`、`used_by_plan`；无 provenance 的论文不得进 Plan
5. 最终 best `APS_lowlight` > baseline `APS_lowlight`
6. 最终结果至少在 **2 个 seed** 上确认（资源允许再 3 seeds）
7. ClaimGate **只按 ExperimentEvidence** 给性能声称；LiteratureEvidence 不得替代本机 APS

**不要求每一轮都涨。** 更好的展示是：

```text
R0 Baseline APS_lowlight = b
  → R1 Early Fusion  ↑  KEEP
  → R2 Aggressive Gating  ↓  DISCARD（LLM 学到 gating 过严）
  → R3 Fusion + Multi-scale  ↑  KEEP
best > baseline
```

失败 → 理解失败 → 改变假设 → 再提高，才证明 AI Scientist。

---

## 4. 数据：两层，切片规则先冻结

不要把整个 RGBT-Tiny 一锅端。

```text
RGBT-Tiny
├── General Set     整体检测表现（APS_all / mAP50-95_all）
└── Condition Slice low_light_subset_v1   低照度专项
```

规则：

- 若原始数据有 day/night 或场景标签 → **直接用标签**，并在 Protocol 里写死字段与取值。
- 若没有 → **不得声称「官方低照度标签」**。用确定性图像统计切一版 `low_light_subset_v1`（例如 RGB luminance / brightness percentile / contrast），哈希进 Dataset Freeze。
- 切片规则必须：**固定、版本化、可复现、不随实验结果改写**。
- **禁止 LLM 挑选对自己有利的子集。** 改切片走 Protocol Amendment，不是本轮 Human Gate。

General Set 仍评，用来区分「真的解决低照度小目标」还是「总体偶然涨点」。

---

## 5. 指标合同

| 角色 | 指标 |
|------|------|
| **Primary** | `APS_lowlight`（低照度切片上的 small-object AP） |
| Secondary（切片） | `mAP50-95_lowlight` · `AP50_lowlight` · `Recall_small_lowlight` |
| Secondary（全集） | `APS_all` · `mAP50-95_all` |

ClaimGate：声称 low-light small-object 改善，但证据只有全集 mAP → **BLOCKED**。  
mAP 不得冒充 APS。切片指标必须来自冻结 evaluator，不得手改 JSON。

---

## 6. LLM 做科研，而不是选菜单

**禁止**给 Planner 三个按钮：`neck` / `fusion` / `hyperparameter`。

必须输入：

```text
Research Goal · Protocol · Current Model
Literature Summary（经 Literature Evidence Gate）
Previous Experiments · Metrics · Failure Analysis
Strategy Memory · Allowed Compute · Available Adapter Capabilities
```

第一轮先形成科研问题再检索，而不是直接问「下一步做什么」。例如：低照度下 RGB-only APS 为何下降、thermal 可能在哪帮忙、现有工作通常在哪一阶段融合、哪些方法针对 small object。检索词由 Planner 构造，系统只返回结构化 `PaperRecord`（title / year / abstract / DOI / citations），**禁止整篇 PDF 塞进上下文**。

必须输出（结构化，进 Plan 附件；非法 JSON fail closed）：

1. 当前瓶颈是什么  
2. 原假设是否被支持  
3. 提出 **2～4** 个候选科研假设  
4. 每个假设预计改善哪项指标  
5. 风险  
6. 成本  
7. selected 下一轮  
8. 为什么（`decision_summary`）

示例（叙事，不是预定剧本）：

```text
Observation: RGB-only 在 low-light 上 APS 很低
Hypothesis: thermal 可补偿退化的 RGB
Candidates: early fusion / feature-level fusion / modality weighting
Selected: early fusion
```

下一轮拿真实 Evidence 回来再改假设。这才是科研迭代。

LLM 仍不得：改代码发明算子、升 `budget_class`、写 Memory、覆盖 Rubric / ClaimGate、改切片规则、把 LiteratureEvidence 送进 ClaimGate。

---

## 7. 受限方法空间（LLM = WHY/WHAT，Adapter = trusted HOW）

第一阶段只允许 **已实现并测过** 的 HOW 组合。未注册的操作 = `MATERIALIZE_REJECTED`。

### 已在 D-FINE Adapter / 训练路径里存在的能力

| 族 | ID | HOW | 状态 |
|----|----|-----|------|
| Fusion | **F0** | RGB only（`rgb` + `none`） | 已有 |
| Fusion | **F1** | `early_concat` | 已有 |
| Fusion | **F3** | `gated_multiscale` | 已有（更重；须标成本） |
| Neck | **N0** | standard HybridEncoder | 已有 |
| Neck | **N1** | `fdpn` | 已有（代码存在；本战役对 Planner **默认不泄露「A4 答案」**） |

### 必须先注册、测过，才能进目录的能力

| 族 | ID | 说明 |
|----|----|------|
| Fusion | **F2** | `weighted_fusion`（模态加权）。当前训练路径 **未** 列为 supported HOW |
| Training | **T0** | baseline 训练配方 |
| Training | **T1** | small-object loss weighting（若实现） |
| Training | **T2** | sampling / low-light 重采样（若实现；不得改切片定义） |

LLM 可组合已注册项，例如 `F1`、`F2`、`F1+N1`、`F2+N1`。  
**每一个组合都要先有 Adapter 翻译和一次 smoke**，禁止 LLM 写 Python。

A4（`F1+N1`）可作为 **隐藏对照**：战役结束后比较 LLM 路径是否接近该方向；比较表是事后分析，不是 Planner 输入。

---

## 8. 两阶段主实验

### Phase 1 — Discovery（D-FINE，G1 + G2）

```text
冻结 Protocol + low_light_subset_v1 + baseline R0
  → LiteratureRetriever（首轮问题导向检索）
  → Literature Evidence Gate → Literature Summary
  → Live LLM Planner（文献 + 实验 + Memory → 2～4 假设）
  → Gate → Human Gate（战役开始时一次性许可本战役预算，不每轮改宪法）
  → Adapter HOW → 真 GPU
  → ExperimentEvidence → Rubric
  → 必要时定向再检索（负结果改 query，不是开头搜一次就结束）
  → Memory → 至少一轮 DISCARD 改向
  → ≥3 轮后选出 Best Method S*
```

预算建议：probe 只用于工程冒烟；**G2 数字必须来自与 baseline 匹配的 run_level**（同一 Frozen Fingerprint 家族；切片规则相同）。  
不要用 16/8 smoke 宣布 G2 成功。

### Phase 2 — Transfer（第二模型，G3）

第二模型 **只选一个**：优先 **RT-DETR**（与 D-FINE 同属 DETR 系，变量更好控）。若 YOLO 工程明显更现成、明显更快，可改 YOLO，但必须在 Protocol 里写死选择理由。

不要一次上三个模型。不要把 FDPN 源文件塞进第二个 detector。

```text
Common Plan（科研语义）
  research_action: multimodal_fusion
  fusion_stage: early
  objective: improve_lowlight_small_object_AP
        ↓
┌─────────────┬─────────────┐
│ D-FINE HOW  │ RT-DETR HOW │
└─────────────┴─────────────┘
```

Transfer 不问 LLM「从零再探索 10 轮」，而问：

> 这些 evidence-linked lessons 里，哪些策略可复用？哪些 HOW 必须交给新 Adapter 重实现？

迁移前可再检索第二模型相关文献（如 `RT-DETR RGB thermal fusion`），用来判断 **策略可复用 / HOW 必须重实现**，不能用来跳过 RT-DETR Adapter 上的真实实验。

若 `RT-DETR baseline APS_lowlight < RT-DETR + transferred strategy`，证明的是 **科研经验可迁移**，不是某个 D-FINE trick。

Transfer Planner **仍是同一个 Planner Agent**（换 transfer prompt / 输入包），不是第五个 Agent。

---

## 9. 工程切入顺序（P0–P4，不要零散开发）

文献检索提前到切片之前接入认知层；切片仍必须在 GPU 战役前冻结。

| 优先级 | 步 | 内容 | 完成判据 |
|--------|----|------|----------|
| **P0** | V26.0 | 真实 LLM API 接入（env 配 key；fail closed） | 无 key 不得假装 Live；默认 rules 仍冻 |
| **P1** | V26.1 | `LiteratureRetriever`：Semantic Scholar 为默认 live Provider；配置可加 arXiv / Crossref / OpenAlex | **已落地（2026-08-18）**：`search` / `get_paper` / `references|citations` / cache + provenance + Literature Evidence Gate |
| **P2** | V26.2 | 低照度切片 + `APS_lowlight` + HOW 目录冻结 | **已落地（2026-08-19）**：`low_light_subset_v1` 规则哈希；无官方 night 标签；LLM 改切片 → Protocol Amendment；ClaimGate 拒绝用全集 mAP/APS 冒充 `APS_lowlight`；HOW 目录 F0/F1/F3/N0/N1，F2/T1/T2 未注册 |
| — | V26.2b | Dataset Workspace / Registry（基础设施，非 Agent） | **已落地（2026-08-19）**：`data/registry` + `data/slices`；Contract = `dataset_id` + `slice_id` + fingerprint；Web `/data` 只读查看/绑定，不改切片名单；图像不进 Git |
| — | V26.3 | 缺的 trusted HOW（决定 F2/T1/T2 做或不做） | 未注册不得出现在 capabilities |
| — | V26.4 | D-FINE R0 baseline（切片指标） | **metrics_bound（2026-08-19）**：HOW=F1 early_concat + standard neck；绑 `rgbt_tiny_v1` + `low_light_subset_v1`；`APS_lowlight=0.0045926865160844455`（pycocotools，切片 val 100 图）；run=`outputs/v26_r0` / `exec_31eff20c4e0c`；未从全集 APS/mAP 或 v2.5 / probe APS=0.0 填数 |
| **P3** | V26.5 | 3～5 轮 Live LLM + 文献 + GPU | **R1–R5 GPU 已落地**（ClaimGate C0 BLOCKED，**不是 G2**）。R5 F3 seed45 `APS_lowlight=0.035942673873977496`，pack `outputs/v26_r5`。Human Gate A：`max_rounds` 仅 5→6。live 文献不得进 ClaimGate。见 [`docs/research/v26/DECISION_HUMAN_GATE_A.md`](./research/v26/DECISION_HUMAN_GATE_A.md) · [`docs/research/v26/V26_5_ROUND5.md`](./research/v26/V26_5_ROUND5.md) |
| — | V26.6 | 2-seed 确认 best | G2 可判定（本战役 `max_claim_strength=C0`，**未开**） |
| **P4** | V26.7–V26.8 | 第二 detector **Transfer Probe**（不是 generalization validation） | **STOP / INCONCLUSIVE（2026-08-21）**：F1 `APS_lowlight=0.013241256515861267`；F3 `0.014949471669213135`（+0.001708），但 AP50_lowlight / 全集 mAP50 下降，成本 ~×1.7。未 BAN F3。ClaimGate C0/BLOCKED。无 `max_rounds` 2→3，无新 GPU。见 [`docs/research/v26/DECISION_STOP_P4.md`](./research/v26/DECISION_STOP_P4.md) |

明确不做（本战役）：Formal E 升格 v2.5-D probe、轨迹训练飞轮 / SFT / DPO / RL、Bounded Tree、扩 Web 产品（`/data` 是 Dataset Workspace 控制台，不是产品扩面）、SOTA 榜、第三个 detector、让 LLM 改切片、文献知识图谱、第五个 Agent。

---

## 10. 对外可说 / 不可说

**可说（战役结束后按 ClaimGate）：**

- 低照度专项协议下，Live LLM 完成了若干轮 Evidence-driven 迭代
- 假设引用了真实论文检索（可指出 `literature_query_id` / `used_by_plan`）
- 存在负结果并改变了后续假设；必要时做了定向再检索
- best `APS_lowlight` 相对本战役 baseline 的差值（若 G2 成立）
- 同一科研策略经第二 Adapter 迁移后的同方向变化（若 G3 成立）

**不可说：**

- A4 是 LLM 发现的
- v2.5-D probe APS=0.0 证明 fusion 无效或有效
- 单调上涨等于科研成功
- Scientist Lab 已把 FDPN 变成与模型无关的即插即用层
- 未注册 HOW 已被 LLM「实现」
- 摘要级文献证明了本仓库里的方法有效 / 「领域共识」
- 论文说 FDPN 有效 ⇒ 我们的 FDPN 有效

---

## 11. 最终展示板（目标形态）

```text
              Baseline   Final
D-FINE        b.bbb      f.fff  ↑或如实写未达 G2
RT-DETR       b.bbb      f.fff  ↑或如实写未达 G3

Round 1 literature queries + hypothesis
Round 2 failure + 定向再检索 + lesson
Round 3 revision
```

旁注：KEEP/DISCARD、LiteratureEvidence、ClaimGate 分栏，避免混读。

---

## 12. LiteratureRetriever（受控工具，不是新系统）

假设来源必须是三者并读，缺一不可把「接了 GPT」说成 AI Scientist：

```text
真实文献（PaperRecord） + 历史实验 Evidence + Strategy Memory
```

**不是新 Agent。** 与 LLM Gateway 一样：Planner / Reviewer 按需调用的确定性工具。上层不关心论文库。

### 12.1 主链

```text
Research Goal
  → Literature Retrieval → LiteratureEvidence（Gate）
  → LLM Planner（+ ExperimentEvidence + Memory）
  → Hypothesis → Candidate Experiments → Plan
  → Gate / Human Gate → Adapter → GPU
  → ExperimentEvidence → Rubric → LLM Reviewer
  → 必要时 Literature Re-search
  → Evidence-linked Lesson → Strategy Memory → Next Hypothesis
```

现有 Protocol / Gate / Adapter / ExperimentEvidence / Rubric / Memory / ClaimGate **全部不用推翻**。

### 12.2 Provider（配置 API，可多家）

```text
LiteratureProvider
  ├── SemanticScholarProvider     # P1 默认；用户提供 API
  └── Future（同一 search() 合同）
        arXiv / Crossref / OpenAlex
```

配置走环境变量 / 本地 config（**key 不进 git**），例如 `SEMANTIC_SCHOLAR_API_KEY`、可选 `ARXIV_*`。无 key 时 fail closed，不得用模型内部记忆假装检索成功。

调用合同：

```text
literature.search(query, year_from=2022, limit=20) → PaperRecord[]
literature.get_paper(paper_id) → PaperRecord
literature.references(paper_id) / literature.citations(paper_id)
```

`PaperRecord` 只含结构化字段：`paper_id, title, year, authors, abstract, doi, venue, citation_count, url, source, retrieval_query, retrieved_at`。不要塞全文。

### 12.3 第一版只做 4 个功能

1. `search_papers`：关键词 / 自然语言  
2. `get_paper`：按 provider paper id 取 metadata / abstract  
3. `references` / `citations`：追相关工作  
4. **本地 cache + provenance**：记下本轮搜了什么、看了哪些、Plan 用了哪些  

CLI（默认 fake，不打网；`--live` 才用 Semantic Scholar）：

```text
scientist-lab literature-search --query "low-light RGB-T small object detection"
scientist-lab literature-search --live --query "RGB thermal fusion low illumination" --year-from 2022 --limit 20
scientist-lab literature-get --paper-id S2:...
scientist-lab literature-refs --paper-id S2:... --kind references
scientist-lab freeze-lowlight-subset --rule-only
scientist-lab freeze-lowlight-subset
```
```

环境变量：`SEMANTIC_SCHOLAR_API_KEY`（必填于 `--live`）。Key 不进 git。无 key 的 `--live` fail closed。

### 12.4 Literature Evidence Gate（硬规则，不是 Agent）

进 Planner 前至少有：`paper_id, title, year, source, retrieval_query, abstract, URL|DOI|arXiv, retrieved_at`。

| 有什么 | 最多能说 |
|--------|----------|
| 只有 metadata | 这篇论文存在 |
| 只有 abstract | 概括摘要支持的内容 |
| 没有全文 | **不得**声称正文证明了某细节 |
| 单篇论文 | **不得**声称领域共识 |
| LiteratureEvidence | **不得**替代 ExperimentEvidence，也不得进 ClaimGate 性能声称 |

论文说「FDPN 对小目标有效」只能推出 **Hypothesis**，不能推出「我们这里的 FDPN 有效」。

### 12.5 每轮 provenance（答辩用）

```json
{
  "literature_query_id": "litq_004",
  "round_id": "round_02",
  "queries": ["RGB thermal fusion low illumination"],
  "paper_refs": ["S2:xxx", "S2:yyy"],
  "used_by_plan": ["S2:xxx"]
}
```

Round 2 的 hypothesis 必须能指回真实 `paper_refs`。这比「我们接了 Semantic Scholar」强。

### 12.6 再检索，而不是开头搜一次

负结果之后 Reviewer 可以触发定向 query（例如从「要不要 fusion」改成「fusion stage / adaptive weighting」）。Transfer 阶段可为第二模型补检索，但仍须跑该 Adapter 的 GPU 实验。
