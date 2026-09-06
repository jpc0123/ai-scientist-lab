# Proof package — Live B invent capability (P3) + E8 vs F1

Date: 2026-09-01  
Status: **KEEP evidence package**（**不是 Claim / 不是 G2 / 不是论文结论**）  
Scope: 证明「系统能 invent→author→smoke，且该插件在 formal 8ep 下相对 catalog F1 有可比增益迹象」

---

## 1. 要证明什么 / 不证明什么

| 这是 | 这不是 |
|------|--------|
| **Invent 能力**：LLM invent 意图 → 人审 → LLM 写插件 → smoke_ok → overlay | 人手写插件（P5）能力 |
| **机制诚实**：真 HxW thermal spatial gate（非 GAP 假空间） | P1 式 channel gate 冒充 spatial |
| **同预算对照**：formal **epochs=8**、同 low-light 切片下 P3 vs F1 | 2ep 尖峰；换数据集；ClaimGate SUPPORTED |
| KEEP / 探索性比较 | G2 成功 Claim；Nature 级可发表结论 |

**KEEP ≠ Claim。** 本包不得直接写入 ClaimGate 成功声明。

---

## 2. 链路证据（Invent capability）

### 2.1 Campaign

- Invent 战役：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260830T171900Z/`（`stage=B_invent`，`completed`）
- 候选：`howc_invent_P3_llm_prove_invent_author`
- 字段（`how_pending.json`）：
  - `draft_origin=invent`
  - `plugin_authored_by=llm`
  - `plugin_worker_id=llm`
  - `smoke_ok=true`
  - `human_decision=register`
  - `fusion_method=plugin:P3`

### 2.2 插件产物

- 路径：`experiment_apps/rgbt_detection_real/models/how_plugins/P3/plugin.py`
- 类：`ThermalSpatialAttentionFusion`（`FeatureFusion`）
- 机制要点：对 thermal 特征做 **1×1 → sigmoid** 得到 **空间注意力图**，再作用于 RGB 后 concat+reduce（真空间维，非 GAP→标量门）

### 2.3 明确排除（不算本证明）

| HOW | 为何不算 invent 能力证明 |
|-----|-------------------------|
| P1 | GAP 通道门，假 spatial |
| P5 | `plugin_authored_by=human_file` / harness 落地 |
| 目录 F0/F1/F3/A4/N1 | 预置 catalog，无 invent |

---

## 3. GPU 对照证据（E8 vs F1）

### 3.1 Campaign / 协议

- Longtrain 战役：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260831T081633Z/`
- Formal knobs：`_FORMAL_TRAIN_KNOBS.epochs=8`（`src/scientist_lab/adapters/dfine/how.py`）
- 主指标：`APS_lowlight`（低光切片；评测规模与协议冻结一致）
- 终态：`failed`（Windows `campaign.json` 原子替换 PermissionError）——**不影响已归档合法 run 的可比性**
- **不要 resume 本场**做 invent；本场仅作 P3 vs F1 证据库

### 3.2 合法对照表（仅 epochs=8）

**F1 = early_concat（catalog）**

| run | APS_lowlight |
|-----|--------------|
| run_plan_v26_r0_dfine_f1 | 0.02148 |
| run_plan_r1_fab6e1578b | 0.01103 |
| run_plan_r2_56f3e22e91 | 0.01128 |
| **n=3 mean** | **≈ 0.01460** |

**P3 = plugin:P3（LLM-authored）**

| run | APS_lowlight |
|-----|--------------|
| run_plan_r4_5ef485419c | 0.07371 |
| run_plan_r5_030f83a31c | 0.04790 |
| run_plan_r6_6f6d1a0132 | 0.04505 |
| run_plan_r7_cf78cb3b1f | 0.04993 |
| run_plan_r8_067013fd76 | 0.04517 |
| **n=5 mean ± pstdev** | **≈ 0.05235 ± 0.01083** |

### 3.3 必须剔除

| run | 原因 |
|-----|------|
| run_plan_r3_c704dec973 | 计划为 P3@8，产物为 `fusion=none` / `epochs=2`（伪 harvest）；**不得计入 P3** |

### 3.4 读数（非 Claim）

- 在同 formal 8ep、同切片下，合法 P3 均值约 **0.052**，F1 均值约 **0.015**；P3 各 seed 均高于 F1 均值。
- 仍有 seed 方差；切片小；**探索性 KEEP 证据**，不是可发表 Claim。
- 2ep 时代结果与本表 **不可直接混比**。

---

## 4. 结论（证明包判定）

1. **Invent 能力（链路）**：以 P3 为主证据 —— **成立（KEEP）**。  
2. **Invent 产物可用性（E8 vs F1）**：合法多 seed 下 P3 相对 F1 **有增益迹象（KEEP）**。  
3. **Claim / G2**：**未成立**；禁止把本包当作论文主结论。

---

## 5. 收口状态（2026-09-01）

- E8 **已归档，勿 resume**（见 `DECISION_LIVE_B_DUAL_INVENT_CLOSEOUT.md`）。  
- **第二份 invent（P4）已立**：`PROOF_LIVE_B_P4_INVENT_CAPABILITY.md`；invent-second 战役亦勿 resume。  
- **两份 invent 已立（P3 + P4）**；KEEP ≠ Claim。

## Evidence index

- Invent：`…20260830T171900Z/how_pending.json`（P3 llm）  
- Plugin：`…/how_plugins/P3/plugin.py`  
- E8 runs：`…20260831T081633Z/runs/run_plan_*`  
- Decisions：`DECISION_LIVE_B_INVENT.md`，`DECISION_LIVE_B_LONGTRAIN_E8.md`，`DECISION_LIVE_B_P3_E8_RESUME.md`，`DECISION_LIVE_B_DUAL_INVENT_CLOSEOUT.md`  
- 合法 P3 示例产物：`outputs/project_rgbt_cuda_001/exec_4a1959fd827f/`（plugin:P3, epochs=8）  
- 姊妹包：`PROOF_LIVE_B_P4_INVENT_CAPABILITY.md`
