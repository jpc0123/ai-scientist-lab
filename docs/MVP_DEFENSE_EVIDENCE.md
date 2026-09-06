# MVP 答辩证据链索引

日期：2026-08-17  
Freeze tag：`mvp-freeze-m1-m4-claimgate-c1` → SHA `130b02cfbb5521829e959d10b99d17fd5fff28ab`  
本文件只索引路径与已冻结口径。**.run/ 大文件不进 git**；现场答辩打开磁盘产物即可。

权威冻结段落仍以 `docs/MVP_FREEZE.md` 与 `docs/ARCHITECTURE_TO_CODE_AUDIT.md` 为准。本页不改数字、不升级声称。

## 声称纪律（写死）

- 产品是可控自主实验系统，不是更好的检测器。
- KEEP ≠ Claim；DISCARD ≠「模块无效」。
- probe ≠ formal；mAP ≠ APS；early_concat ≠ FDPN；不得 C2。
- Manager 单臂 BLOCKED 是正确行为，不能拿它当 C1 失败。
- 配对 `claim_gate_c1.json` 的 C1 SUPPORTED 与 `scientific_outcome=INCONCLUSIVE` 可并存：投影只看 `primary_delta`，不是第二条声称状态机。
- 160×160 / 2ep staging，不是论文协议。不得把 v2.5 / LLM / Web / dataset 战役说成 MVP。

## 1. Protocol / Plan（在 git freeze 内）

| 角色 | 路径 |
|------|------|
| Formal C1 Protocol | `schemas/examples/research_protocol_rgbt_dfine_formal_c1_v1.json` |
| Formal-01 Plan（baseline，hyperparameter → rgb+none） | `schemas/examples/experiment_plan_formal_01_rgb_none.json` |
| Formal-02 Plan（candidate，fusion → early_concat） | `schemas/examples/experiment_plan_formal_02_early_concat.json` |
| C1 声称模板 | `schemas/examples/claim_c1_fusion_aps.json` |
| Fingerprint schema / 例 | `schemas/frozen_fingerprint.schema.json`、`schemas/examples/frozen_fingerprint_rgbt_dfine_v1.json` |
| ClaimGate schema | `schemas/claim_gate_result.schema.json` |
| Probe 协议（禁止科学声称，对照用） | `schemas/examples/research_protocol_rgbt_dfine_v1.json` |

## 2. Formal-01 / Formal-02 磁盘产物（.run/，gitignore）

根目录：`.run/formal_c1_aps_early_concat/`

| 臂 | handle | metrics | result / review |
|----|--------|---------|-----------------|
| Formal-01 | `.run/formal_c1_aps_early_concat/baseline/handle.json` | `.run/formal_c1_aps_early_concat/baseline/run/metrics.json` | `baseline/result.json`、`baseline/review.json` |
| Formal-02 | `.run/formal_c1_aps_early_concat/candidate_resume/handle.json` | `.run/formal_c1_aps_early_concat/candidate_resume/run/metrics.json` | `candidate_resume/result.json`、`candidate_resume/review.json` |

中断未完成臂（不要当 C1）：`.run/formal_c1_aps_early_concat/candidate/`。

### 同一 Frozen Fingerprint（五哈希一致；HOW 只在 notes）

两臂 `fingerprint_id=FP-RGBT-DFINE-FORMAL-C1-V1`：

| 字段 | 值 |
|------|----|
| `dataset_split_hash` | `b5ee87ee748bf6b2df266be1764a940b0068bc0738c9260bd1a34dd7f1414471` |
| `evaluator_hash` | `0acf11ecccd6b25230dc8252ce021abcc71e4a160f57ce7d089fcf12ed9bfc6e` |
| `metric_definition_hash` | `f2deff7f692e69506f6d21b0fa908487f523fc506b8d23913f4788aa2e4345bc` |
| `baseline_config_hash` | `eec836882acf99ed429359327b80870e7427dd35f2dbb7f52224fa4f3fb93f13` |
| `data_manifest_hash` | `542de08936488c621fd5cb9781f91d84ef40426b8bfe07dc4561be4d47b8fec7` |

口径 APS（四位）：baseline **0.0163**，early_concat **0.0326**。  
handle 内 raw：`APS=0.016313298031160568` / `0.03256971555632778`（`AP_small` 同值）。

run_id：`run_plan_formal_01_rgb_none` / `run_plan_formal_02_early_concat`。

## 3. ClaimGate 配对 vs Manager 单臂

| 产物 | 路径 | 口径 |
|------|------|------|
| 配对 C1（`--baseline-run-dir`） | `.run/formal_c1_aps_early_concat/claim_gate_c1.json` | **SUPPORTED**；`keep_is_not_claim=true`；reason 写明 KEEP/DISCARD 未决定声称 |
| Manager 自动单臂 | `.run/formal_c1_aps_early_concat/candidate_resume/claim_gate.json` | **BLOCKED**；`baseline comparison lacks matched Frozen Fingerprint (C1)` — 正确 |
| 单臂 baseline 文件 | `.run/formal_c1_aps_early_concat/candidate_resume/baseline_metrics.json` | 无数指纹，不能当配对 C1 |

实现：`src/scientist_lab/core/claim_gate.py`；CLI `--baseline-run-dir`：`src/scientist_lab/cli.py`。  
Manager `_attach_claim_gate` 对仅 `baseline_metrics` 写 `matched_fingerprint=False`。

## 4. HOW 差异（不是 FDPN）

源码：`src/scientist_lab/adapters/dfine/how.py`（`_NECK_HOW` / `_FUSION_HOW`；`invented_operators=[]`；`neck_type=standard`）。

| 臂 | HOW 身份 | notes / 产物要点 |
|----|----------|------------------|
| Formal-01 | `input_mode=rgb`，`fusion_method=none` | `module=hyperparameter` fallback，与 neck HOW 同身份；HOW signature `f80f627b…` |
| Formal-02 | `input_mode=rgbt`，`fusion_method=early_concat` | `staging_mode=early_concat_blend`，`fusion_applied=false`；HOW signature `27905f8b…` |

哈希不含 HOW，故两臂 Frozen Fingerprint 可比。early_concat 是已有 staging 混合，**不是**新 fusion 网络，不得升 C2。  
单测：`tests/unit/test_dfine_adapter_m2.py` — `test_materialize_neck_vs_fusion_how_differs`、`test_neck_fusion_frozen_fingerprint_equivalent_how_notes_differ`。

## 5. pytest 口径

Audit 终态：**676 passed / 12 skipped**。本封版材料只加 docs，未改代码，未重跑全量。

关键测试（git freeze 内）：

| 主题 | 路径 |
|------|------|
| ClaimGate / KEEP≠Claim / probe≠formal / mAP≠APS | `tests/unit/test_claim_gate_m4.py`（`test_keep_is_not_supported`、`test_only_formal_matched_pair_supports_c1`、`test_probe_c1_and_c2_not_supported`、`test_metric_spec_bound_map_is_not_aps`） |
| HOW / Fingerprint | `tests/unit/test_dfine_adapter_m2.py` |
| Memory → Next Plan | 见下一节 |

## 6. Memory → Next Plan

源码：`src/scientist_lab/core/next_plan.py`、`src/scientist_lab/core/memory_writer.py`、`src/scientist_lab/core/invariants.py`（`assert_memory_refs_resolvable`）。

| 测试文件 | 代表用例 |
|----------|----------|
| `tests/unit/test_n_plus_one_memory_refs_m2.py` | `test_recovered_valid_run_next_plan_with_memory_refs_passes`、`test_next_plan_missing_memory_refs_fails`、`test_discard_negative_evidence_lesson_is_citable` |
| `tests/unit/test_planner_m3.py` | `test_planner_emits_resolvable_plan_citing_discard_lesson`、`test_round_ge1_without_memory_is_refused` |
| `tests/unit/test_manager_multround_m4.py` | `test_three_stub_rounds_cite_memory_and_switch_module` |
| `tests/unit/test_git_memory_m2.py` | MemoryWriter 拒无证据 lesson；Trace 持久化 |

## 7. Architecture-to-Code Audit PASS

路径：`docs/ARCHITECTURE_TO_CODE_AUDIT.md`。终态 **PASS**，checklist 9/9。  
冻结摘要：`docs/MVP_FREEZE.md`。  
权威架构（工作区，不在本 tag 内改写）：`d:\AI Scientist_tiao\设计架构.md`、`d:\AI Scientist_tiao\新版构建方案.md`。

## 8. Git 保护（现场核对命令）

```text
git rev-parse mvp-freeze-m1-m4-claimgate-c1^{}
# 期望：130b02cfbb5521829e959d10b99d17fd5fff28ab

git rev-parse freeze/mvp-m1-m4-claimgate-c1
# 期望：同上

git check-ignore -v .run/formal_c1_aps_early_concat/claim_gate_c1.json
# 期望：命中 .gitignore 的 .run/
```

不要 `git tag -f`，不要 amend freeze commit，不要把 `.run/`、datasets、v2.5、web、llm 打进该 tag。
