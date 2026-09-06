# Proof package — Live B invent capability (P4, second) + vs F1

Date: 2026-09-01  
Status: **KEEP evidence package**（**不是 Claim / 不是 G2 / 不是论文结论**）  
Prior: `PROOF_LIVE_B_P3_INVENT_CAPABILITY.md`（第一份 invent）  
Scope: 证明「invent 可重复：机制 ≠ P3 的第二条 LLM invent→author→smoke，且 formal 8ep 下相对本场 F1 有增益迹象」

---

## 1. 要证明什么 / 不证明什么

| 这是 | 这不是 |
|------|--------|
| **第二份 invent**：新机制 ≠ P3 thermal concat gate | 再训 / 续跑 E8；复用 P3 代码 |
| LLM invent → 人审 → LLM author → smoke_ok → overlay | harness / `human_file`（P5） |
| 本场 formal **epochs=8** 下 P4 vs 本场 F1 R0 | 与 E8 场 F1 均值直接混比 Claim |
| KEEP / 可重复性 | ClaimGate SUPPORTED / G2 |

**KEEP ≠ Claim。** 本包不得直接写入 ClaimGate 成功声明。

---

## 2. 链路证据（Invent capability #2）

### 2.1 Campaign

- Invent-second 战役：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260901T025804Z/`  
  - `stage=B_invent`，终态 **`completed`**，`gpu_rounds=9`  
  - `llm_may_invent_how=true`，`plugin_worker=llm`
- 候选：`howc_invent_P4_round_2`
- 字段（`how_pending.json`）：
  - `draft_origin=invent`
  - `plugin_authored_by=llm`
  - `plugin_worker_id=llm`
  - `smoke_ok=true`
  - `human_decision=register`
  - `status=registered`
  - `fusion_method=plugin:P4`
  - `mechanism=Cross-Modal Spatial Gating with Thermal Confidence Weighting`

### 2.2 插件产物（与 P3 可区分）

- 路径：`experiment_apps/rgbt_detection_real/models/how_plugins/P4/plugin.py`
- 类：`SpatialConfidenceFusion`（`FeatureFusion`）
- 机制要点（相对 P3）：
  - **双向** HxW 门控：thermal→mask 门控 RGB；RGB→mask 门控 thermal
  - 门控后 **相加**（可选 RGB residual），**不做** P3 的 concat+reduce
- Overlay：`registered_overlay['P4']`

### 2.3 同场其它 invent（不算本包主证据）

| HOW | 状态 | 说明 |
|-----|------|------|
| P6–P10 | invent+smoke 后 **rejected** | 人审/策略拒绝；不计入第二份「已立」主证据 |
| Catalog F1 / F3 | 非 invent | F3 仅 1 个 seed，不作 P4 主对照 |

---

## 3. GPU 对照（本场 P4@8ep vs F1）

### 3.1 协议

- Formal knobs：`epochs=8`（与 E8 臂相同 Adapter formal）
- 主指标：`APS_lowlight`
- **不要 resume** 本场；证据已齐则归档

### 3.2 合法对照表（仅 epochs=8）

**F1 = early_concat（本场 R0）**

| run | APS_lowlight |
|-----|--------------|
| run_plan_v26_r0_dfine_f1 | 0.01052 |
| **n=1 mean** | **≈ 0.01052** |

（弱于 E8 场 F1 n=3 对照；读数时注明 **本场 F1 单 seed**。）

**P4 = plugin:P4（LLM-authored）**

| run | APS_lowlight |
|-----|--------------|
| run_plan_r2_56f3e22e91 | 0.05752 |
| run_plan_r3_c704dec973 | 0.05581 |
| run_plan_r4_5ef485419c | 0.07200 |
| run_plan_r5_030f83a31c | 0.02248 |
| run_plan_r6_6f6d1a0132 | 0.06945 |
| run_plan_r7_cf78cb3b1f | 0.02190 |
| live / last P4（若归档于 `run/`） | 0.04393 |
| **archived P4 n=6 mean** | **≈ 0.04994**（约 0.022–0.072） |

**同场其它（非主对照）**

| run | fusion | APS_lowlight |
|-----|--------|--------------|
| run_plan_r1_fab6e1578b | gated_multiscale (F3) | 0.06994 | n=1 |

### 3.3 读数（非 Claim）

- 相对本场 F1（≈0.011），P4 多数 run 更高；均值约 **0.050**。
- **seed 方差大**（最低 ≈0.022）；F1 仅 n=1 —— 增益是 **KEEP 迹象**，不是稳固 Claim。
- 与 E8 场 P3 均值（≈0.052）数量级相近，但 **两场 F1 基线不同，禁止把两场 delta 直接拼成跨战役 Claim**。

---

## 4. 结论

1. **Invent 可重复性（链路 #2）**：P4 —— **成立（KEEP）**。  
2. **相对本场 F1 的可用性**：有增益迹象（KEEP）；方差与单 seed F1 限制强度。  
3. **Claim / G2**：**未成立**。

---

## 5. 归档与「两份 invent 已立」

见 `DECISION_LIVE_B_DUAL_INVENT_CLOSEOUT.md`：

- **两份 invent 已立**：P3（第一份）+ P4（第二份，机制可区分）。  
- **勿 resume** E8 `…081633Z` 与 invent-second `…20260901T025804Z`。

## Evidence index

- Campaign：`.run/autonomous/exp_rgbt_dfine_v26_lowlight_20260901T025804Z/`  
- Plugin：`…/how_plugins/P4/plugin.py`  
- Prior P3 pack：`PROOF_LIVE_B_P3_INVENT_CAPABILITY.md`  
- Decision：`DECISION_LIVE_B_INVENT_SECOND.md`，`DECISION_LIVE_B_DUAL_INVENT_CLOSEOUT.md`
