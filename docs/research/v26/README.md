# v2.6 Campaign

Opened after v2.5-D Demo acceptance. Does **not** reopen A4 as an LLM discovery claim.

- Decision: [`DECISION_OPEN_V26_CAMPAIGN.md`](./DECISION_OPEN_V26_CAMPAIGN.md)
- HOW plugin (write Python / fusion only): [`DECISION_OPEN_HOW_PLUGIN.md`](./DECISION_OPEN_HOW_PLUGIN.md)
- LLM HOW lifecycle (protocol frozen; LLM add/author/accept overlay): [`DECISION_LLM_HOW_LIFECYCLE.md`](./DECISION_LLM_HOW_LIFECYCLE.md)
- Construction direction: [`../../V26_LLM_AUTONOMOUS_DETECTION.md`](../../V26_LLM_AUTONOMOUS_DETECTION.md)

Question: can an LLM-driven AI Scientist improve **low-light RGB-T tiny-object** detection using **literature + experiment evidence + memory**, under Protocol / Adapter / Gate / Rubric / ClaimGate, then transfer the strategy to a second detector?

Goals: G1 Live ≥3 rounds with citable literature provenance · G2 `APS_lowlight` > baseline · G3 transfer lift.

Priority: P0 live LLM API · P1 Semantic Scholar LiteratureRetriever · **P2 done** · Dataset Workspace inserted · **V26.4 R0 metrics_bound** (`APS_lowlight=0.0045926865160844455`) · **V26.5 R1–R5 GPU done**（ClaimGate BLOCKED/C0；**不是 G2**）· **Human Gate A**（[`DECISION_HUMAN_GATE_A.md`](./DECISION_HUMAN_GATE_A.md)）：`max_rounds` 仅 5→6；ClaimGate 仍 C0。R5 = F3 seed 45 pack `outputs/v26_r5`，`APS_lowlight=0.035942673873977496`。**P4 GO**（[`DECISION_P4_GO.md`](./DECISION_P4_GO.md)）：RT-DETR Adapter + transfer Protocol；不是 D-FINE R6。文献 live 不得进 ClaimGate。

P2 freeze: [`LOW_LIGHT_SUBSET_V1.md`](./LOW_LIGHT_SUBSET_V1.md) · [`HOW_CATALOG_V26.json`](./HOW_CATALOG_V26.json)

Dataset Workspace: [`DATASET_WORKSPACE.md`](./DATASET_WORKSPACE.md) · `data/registry/` · `data/slices/`

R0: [`R0_BASELINE.md`](./R0_BASELINE.md) · [`R0_BASELINE_FREEZE.json`](./R0_BASELINE_FREEZE.json)

V26.5 Round 1: [`V26_5_ROUND1.md`](./V26_5_ROUND1.md) · `outputs/v26_r1/`

V26.5 Round 2: [`V26_5_ROUND2.md`](./V26_5_ROUND2.md) · `outputs/v26_r2/`

V26.5 Round 3: [`V26_5_ROUND3.md`](./V26_5_ROUND3.md) · `outputs/v26_r3/`（F3 seed 43 `APS_lowlight=0.04881493259150456`，与 Round 1 F3 seed 42 同方向高于 R0；**不是 G2 成功声明**）

V26.5 Round 4: [`V26_5_ROUND4.md`](./V26_5_ROUND4.md) · `outputs/v26_r4/`（F3 seed 44 `APS_lowlight=0.04204268629433699`，三 seed 同方向高于 R0；**不是 G2 成功声明**）

V26.5 Round 5: [`V26_5_ROUND5.md`](./V26_5_ROUND5.md) · `outputs/v26_r5/`（F3 seed 45 `APS_lowlight=0.035942673873977496`，四 seed 同方向高于 R0；ClaimGate BLOCKED/C0；**不是 G2 成功声明**）

STOP 前决策：[`DECISION_STOP_MAX_ROUNDS.md`](./DECISION_STOP_MAX_ROUNDS.md) · Human Gate A：[`DECISION_HUMAN_GATE_A.md`](./DECISION_HUMAN_GATE_A.md) · live probe `outputs/v26_literature_live/` `provider=semantic_scholar` `litq_687d9c49f056`（未进 ClaimGate）

P4 RT-DETR **Transfer Probe**：**STOP / INCONCLUSIVE**（[`DECISION_STOP_P4.md`](./DECISION_STOP_P4.md)）。不是 generalization validation。Adapter=`rtdetr` · packs `outputs/v26_p4_r0`（F1）/ `outputs/v26_p4_r1`（F3）。ClaimGate C0/BLOCKED。未 BAN F3。C0 分析：[`V26_P4_C0_ANALYSIS.md`](./V26_P4_C0_ANALYSIS.md)。

**P0 无人值守两轮（2026-08-24）**：**产品门 ACCEPT / 科学 INCONCLUSIVE**（[`DECISION_P0_ACCEPT.md`](./DECISION_P0_ACCEPT.md)）。战役 `p0_20260824T111526Z`，一次 Human Gate，`gpu_rounds=2`，HOW=F1×2 seed 42，`APS_lowlight` 0.00642 / 0.01034。Rubric REPLICATE。ClaimGate C0。KEEP ≠ Claim。无第三轮 GPU。

**P1 Planner 质量（2026-08-24）**：**INCONCLUSIVE**（[`DECISION_P1_PLANNER_QUALITY.md`](./DECISION_P1_PLANNER_QUALITY.md)）。规则 `p0_20260824T094123Z` vs LLM `p0_20260824T111526Z`。两臂 R1 都空转 F1 seed 42，Rubric Δ=`null`。不是 APS 赛。KEEP ≠ Claim。

基模后训练（不训检测器）：

- 三类样本制作策略：[`LLM_POSTTRAIN_DATA_STRATEGY.md`](./LLM_POSTTRAIN_DATA_STRATEGY.md)
- 隔离考题（对照卷，禁止进训练）：[`LLM_HOLDOUT_EXAM_CONTRAST_V1.md`](./LLM_HOLDOUT_EXAM_CONTRAST_V1.md)
- 六元组只读导出：`scientist-lab export-trajectory --run-dir <run> [--parent-dir <anchor>]`（`trajectory_step.schema.json`，不改账本、不跑 GPU）
