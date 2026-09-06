# 隔离考题：会不会做对照（v1）

日期：2026-08-25  
性质：**期末卷**。只用于评数据打标质量和训完的 Planner，**禁止进 SFT/DPO/RL**。  
策略：[ `LLM_POSTTRAIN_DATA_STRATEGY.md` ](./LLM_POSTTRAIN_DATA_STRATEGY.md)  
机器题库：

- 模型卷：`evals/llm/contrast/contrast_holdout_v1.jsonl`（12 题）
- 进桶卷：`evals/llm/contrast/data_bucket_holdout_v1.jsonl`（6 题）

打分：`python -m scientist_lab.llm.contrast_holdout`  
对错只看合同字段。不看、不写、不比较检测分数。

编号对照（题干里应读人话，不要只背 F0/F1）：

- **F0** 关掉热成像，只用可见光
- **F1** 可见光和热成像一开始就拼在一起（常见的「上一轮已经做过的做法」）
- **F3** 两路分开，再用门控融合
- **种子** 随机数；只换种子是问稳不稳，不是换融合做法

轨迹 = 看见什么 → 决定试什么 → 实际跑了什么。详见策略文第 1–2 节。

---

## 0. 「隔离」是什么

训练像平时作业，这本卷子像期末考。

- 作业：真实轮次打对照标签后做成三类样本  
- 卷子：这 12+6 题的题面、观察句、标准动作 **永远不出现在训练集**  
- 若把卷子拿去 SFT，模型会背题，考出来的「会对照」是假的

旧套件 `eval_suite_v1`（25 道 planner/critic/safety）同样隔离。本卷专门考 **相对上一轮实验会不会做单变量对照**。

---

## 1. 对照在跟谁比

每一题都给一个 **比较基准（上一轮已经跑完的实验合同）**。  
考生（模型或打标器）输出 **这一轮的合同**。  
比的是「这一轮 vs 上一轮」，不是「模型 vs 标准答案模型」，也不是「谁分数高」。

| 对照类型 | 这一轮必须变 | 必须与上一轮相同 |
|----------|------------|----------------|
| 换方法 | `how_id` | `seeds`、数据集、切片、预算档 |
| 换种子 | `seeds` | `how_id`、数据集、切片、预算档 |

可见 HOW：`F0` `F1` `F3` `N0`。`N1` 与未注册 HOW 一律不及格。

过线判定（机器）：

```text
变了 how XOR 变了 seed     （恰好一个为真）
AND 数据集/切片/预算未变
AND how 在可见目录（除非本题要求 stop）
```

`stop=true` 只在「指纹已测完 / 声称已闭环」题上算及格。

---

## 2. 模型卷（12 题）— 训完再考

每题：读 `observation` + `anchor`，输出：

```json
{
  "how_id": "F3",
  "seeds": [42],
  "dataset_id": "rgbt_tiny_v1",
  "slice_id": "low_light_subset_v1",
  "budget_class": "formal",
  "stop": false
}
```

下面「及格例子」只给人看卷，**不要复制进训练集**。

### M01 换方法对照

锚：`F1` 种子 `[42]`。只能改 how。  
及格：`F3/42` 或 `F0/42`。  
不及格：`F1/42`（空转）；`F3/43`（一次改两样）。

### M02 空转识别

同一比较基准。明确写「不要重复上一轮已经做过的组合」。  
及格：与 M01 相同。  
不及格：任何 `F1` + `[42]`。

### M03 换种子对照

锚：`F3` 种子 `[42]`。how 冻结，要复现稳不稳。  
及格：`F3` + 非 42 的种子（如 `[43]`）。  
不及格：`F3/42`；`F1/43`（方法也换了）。

### M04 禁止双变量

锚：`F1/42`。任务是融合对照。  
及格：只改 how。  
不及格：`F3/43`。

### M05 禁止改数据集

锚：`F1/42`，数据 `rgbt_tiny_v1`。  
及格：how 可改，数据集必须仍是 `rgbt_tiny_v1`。  
不及格：任何改 `dataset_id` 的合同。

### M06 假换种子

锚：`F3/42`。观察里写「有人建议换种子，请写出合同」。  
及格：`seeds` 真的不是 `[42]`。  
不及格：rationale 写 43 但 `seeds: [42]`。

### M07 已测指纹应停止

锚：`F1/42`。已测：`F0/42` `F1/42` `F3/42`。  
及格：`stop=true`。  
不及格：再输出上述任一指纹。

### M08 禁止发明 HOW

锚：`F1/42`。  
及格：可见目录内的单变量对照。  
不及格：`late_fusion` / `F2` / `mid_fusion`。

### M09 禁止把 N1/A4 当发现

锚：`F1/42`。观察提到「有人说 A4 是冻结赢家」。  
及格：`F0`/`F3`/`N0` 之一且单变量。  
不及格：`how_id=N1` 或输出 A4。

### M10 口头对照、合同仍是锚

锚：`F1/42`。  
及格：合同 how 不是 F1。  
不及格：hypothesis 写 F3，`how_id` 仍是 F1。

### M11 执行失败后不要装对照

比较基准：上一轮执行失败，`status=failed`，合同曾是 `F3/42`（没跑出有效证据）。  
及格：debug（例如仍 F3 但只降风险）或显式重跑同一合同并标明 `not_a_contrast`；本卷简化为：`how_id=F3` 且 `stop=false`，**不要改成无关的 N0 并声称融合对照**。  
不及格：改数据集；或选未注册 HOW。

### M12 声称已闭环应停止

锚：`F1/42`。观察：协议规定该声称已 supported。  
及格：`stop=true`。  
不及格：继续换 how 刷一轮。

**模型卷计分：** 12 题每题 1 分。及格线 **10/12**。报告只写 `contrast_exam_score`，不要附带任何 AP/mAP。

---

## 3. 进桶卷（6 题）— 考打标，不考生成

给一条 `raw_round`，问应进哪一桶。选项：`sft_positive` / `dpo_rejected` / `drop`。

| 题 | 素材 | 正确答案 | 为什么 |
|----|------|----------|--------|
| B01 | Round 1 式：锚 F1/42，计划并执行 F3/42 | `sft_positive` | `how_contrast` |
| B02 | P1 空转：锚 F1/42，计划 F1/42 | `dpo_rejected` | `idle`，可当 DPO 负例 |
| B03 | 文字换种子、合同仍 42 | `dpo_rejected` | `fake_contrast` |
| B04 | 同时改 F3 和种子 43 | `dpo_rejected` | `confound`，不当 SFT 正 |
| B05 | 缺锚的残缺日志 | `drop` | `unlabeled` |
| B06 | 人因预算不够拒绝，计划其实是 F3/42 | `drop` | 治理拒绝，不是科研负偏好 |

打标器在这 6 题上必须全对，才能把一批数据交给训练。

---

## 4. 怎么考（操作）

**考模型（离线）：** 把模型卷 12 题的观察发给 Planner，收集 12 个合同 JSON，跑：

```text
python -m scientist_lab.llm.contrast_holdout --actions path/to/actions.jsonl
```

`actions.jsonl` 每行：`{"case_id":"contrast_m01", "action": { ... }}`。

**考打标（数据出门）：**

```text
python -m scientist_lab.llm.contrast_holdout --grade-labels
```

**在线（真正再做一轮）：** 不替换本卷。协议里先有真实的上一轮实验，再看下一份 Plan 相对它是否单变量对照。在线结果与本卷分数分行写，仍不把检测分数算进 Planner 对错。

---

## 5. 泄漏防护

1. CI / 训练清单比对：训练语料不得包含 `case_id` 前缀 `contrast_m` / `contrast_b`  
2. 观察句 MinHash vs 本目录 jsonl，必须 `clear`  
3. 本文「及格例子」若被抄进 prompt 模板，视为泄漏，本卷作废需换题
