# Architecture-to-Code Audit — MVP Freeze

源码仓库（磁盘为准，不是「当前会话没有源码仓库」）：`d:\AI Scientist_tiao\scientist-lab`。  
权威 Freeze 文本仍在工作区 `d:\AI Scientist_tiao\设计架构.md` 与 `d:\AI Scientist_tiao\新版构建方案.md`（本文件不整篇改写那两份）。  
日期：2026-08-17。停止扩功能。本文件先收口 **Conditional PASS**，核对 checklist 全过后升 **PASS**。

## 终态

**PASS**（源码级最终 Audit；checklist 9/9 全过；pytest 676 passed / 12 skipped；未重跑 GPU / probe / formal。）

MVP 两条核心能力已落地，边界未弱化：

1. 自主科研行为闭环：Gate → Run → Evidence → Rubric → Memory → Next Plan
2. 科学声称闭环：ClaimGate 限制 Evidence 最多支持的 Claim 强度

已接受 Formal C1：同一 Frozen Fingerprint，baseline APS=0.0163，early_concat APS=0.0326，C1 **SUPPORTED**。

### 能力边界（必须保留）

- 160×160 / 2 epoch 是 staging/formal-C1，不是论文协议
- early_concat 是 staging 融合 HOW，不是 FDPN，不得升 C2
- probe ≠ formal evidence
- mAP ≠ APS
- KEEP/DISCARD ≠ ClaimGate SUPPORTED/INCONCLUSIVE/BLOCKED
- `scientific_outcome` 是只读投影，不得与 ClaimGate 混读，不得形成平行科研判断状态机
- 科研负结果不是 error；FAILED/INVALID 才是工程/协议错误

### 明确不进 MVP

新 GPU probe/formal、`trajectory_step`、训练飞轮、preference/SFT/DPO/RL、Bounded Tree、LLM Planner/Reviewer、FDPN C2、论文级 SOTA。

## Checklist（对着代码 / 测试 / 产物核对）

| # | 项 | 结论 | 路径 |
|---|----|------|------|
| 1 | Frozen Fingerprint：哈希不含 HOW；neck vs fusion 可比；C1 两臂 hashes 一致 | **PASS** | `src/scientist_lab/adapters/dfine/fingerprint.py` `_HASH_FIELDS` 不含 HOW；HOW 只进 `notes`。测试 `tests/unit/test_dfine_adapter_m2.py` `test_neck_fusion_frozen_fingerprint_equivalent_how_notes_differ`、`test_formal_how_full_train_drops_probe_subset`。C1 产物两臂五哈希相同：`dataset_split_hash=b5ee87ee…` / `evaluator_hash=0acf11ec…` / `metric_definition_hash=f2deff7f…` / `baseline_config_hash=eec83688…` / `data_manifest_hash=542de089…`（`.run/formal_c1_aps_early_concat/baseline/handle.json` vs `candidate_resume/handle.json`）。notes 不同：baseline `module=hyperparameter; rgb+none`，candidate `module=fusion; rgbt+early_concat`。 |
| 2 | Evidence UID / run_id 绑定 | **PASS** | Freeze 证据身份是 `run_id`（无游离 `evidence_uid`）。`schemas/experiment_result.schema.json` / `experiment_run.schema.json` 必填 `run_id`。`src/scientist_lab/core/result_parser.py` 缺 `run_id` 即 `KeyError`。Adapter `run_{plan_id}`。C1：`run_plan_formal_01_rgb_none` / `run_plan_formal_02_early_concat` 贯穿 handle / result / experiment_run / ClaimGate `evidence_refs`。 |
| 3 | ClaimGate refs：reason + evidence_refs；KEEP ≠ SUPPORTED；Manager 仅 baseline_metrics 时 BLOCKED；配对 `--baseline-run-dir` 才是 C1 | **PASS** | `src/scientist_lab/core/claim_gate.py` 输出必有 `reason` + `evidence_refs` + `keep_is_not_claim=true`。`invariants.assert_keep_is_not_claim`。测试 `test_output_has_reason_and_evidence_refs`、`test_keep_is_not_supported`、`test_only_formal_matched_pair_supports_c1`。CLI `--baseline-run-dir`：`src/scientist_lab/cli.py`。Manager `_attach_claim_gate` 对 `baseline_metrics` 写 `matched_fingerprint=False`。只读产物：`claim_gate_c1.json` **SUPPORTED**（KEEP 在 refs 里但 reason 写明 KEEP/DISCARD did not decide）；`candidate_resume/claim_gate.json` **BLOCKED**（`baseline comparison lacks matched Frozen Fingerprint`）——正确。 |
| 4 | probe vs formal metadata（budget_class / run_level）；16/8 不得当 C1 | **PASS** | Plan `budget_class` → ClaimGate `run_level`。probe C1 BLOCKED：`test_probe_c1_and_c2_not_supported`、`test_probe_vs_formal_distinguished`。formal HOW 去掉 16/8：`adapters/dfine/how.py`、`cuda_runner.py` `full_train` pop subset。C1 产物 `dfine_subset.json`：`execution_mode=full_train`，`max_train_images=null`，staged **3342/650**（不是 16/8）。 |
| 5 | metric-spec 强绑定：APS ≠ mAP50 ≠ mAP50-95；声称 small-object 只有 mAP → BLOCKED | **PASS** | `metrics_parser.py`：APS 只取 `APS`/`AP_small`，不从 mAP 发明。`claim_gate.py` `_has_only_map_for_aps`。测试 `test_metric_spec_bound_map_is_not_aps`。C1 `result.json` 同时保留 APS / mAP50 / mAP50_95，声称用 APS。 |
| 6 | Adapter HOW：neck=rgb+none；fusion=rgbt+early_concat（已有 staging），不发明 FDPN | **PASS** | `adapters/dfine/how.py` `_NECK_HOW` / `_FUSION_HOW`；`invented_operators=[]`；`neck_type=standard`。测试 `test_materialize_neck_vs_fusion_how_differs`。C1 Formal-01 `modification_scope=hyperparameter` 走 fallback，仍是 rgb+none（与 neck HOW 同身份、不同 `primary_module`）；Formal-02 fusion=early_concat，`staging_mode=early_concat_blend`，`fusion_applied=false`。不是 FDPN。 |
| 7 | Memory → Next Plan memory_refs | **PASS** | `core/next_plan.py` `memory_refs_from_run`；`invariants.assert_memory_refs_resolvable`；GateEngine Round≥1 REJECT 空/假 refs。测试 `tests/unit/test_n_plus_one_memory_refs_m2.py`、`test_planner_m3.py`、`test_manager_multround_m4.py`。 |
| 8 | tests：pytest（pythonpath src,worker）；不为刷绿改 rubric | **PASS** | `.venv` 补装已声明依赖 `jsonschema`/`referencing`（环境缺口，未改 DecisionRubric）。唯一红项是 HEAD 遗留断言：`test_dfine_vendor.py` 仍 match `not implemented`，而 `dfine_s.py` 已拒绝 `fusion_method=fdpn` 为 `not a fusion switch`（与 `test_fdpn_neck.py` 一致）。只改测试正则对齐现有拒绝语义，**未改 rubric**。终态：**676 passed, 12 skipped**。 |
| 9 | working tree：`.run/` gitignore；不把 GPU 产物提交进 freeze | **PASS** | `.gitignore` 第 16 行 `.run/`。`git check-ignore -v .run/formal_c1_aps_early_concat/claim_gate_c1.json` 命中。C1 GPU 产物只读、不入库。 |

## C1 产物只读核对（未重跑）

| 产物 | 期望 | 实测 |
|------|------|------|
| `.run/formal_c1_aps_early_concat/baseline/run/metrics.json` | APS=0.0163 | `AP_small=0.016313298031160568`；canonical APS 同值 |
| `.run/formal_c1_aps_early_concat/candidate_resume/run/metrics.json` | APS=0.0326 | `AP_small=0.03256971555632778`；canonical APS 同值 |
| `.run/formal_c1_aps_early_concat/claim_gate_c1.json` | SUPPORTED | `status=SUPPORTED`，`run_level=formal`，`keep_is_not_claim=true`，`scientific_outcome=INCONCLUSIVE`（投影未覆盖 ClaimGate） |
| Manager 自动 `candidate_resume/claim_gate.json` | BLOCKED（无配对指纹） | `status=BLOCKED`，`reason=baseline comparison lacks matched Frozen Fingerprint (C1)` |

`scientific_outcome=INCONCLUSIVE` 与 ClaimGate `SUPPORTED` 并存是正确的：投影只看 `primary_delta`（证据阶段常为 None），不是第二条声称状态机。

## 缺口

无。Audit 升 **PASS**。允许 Git Freeze（commit + annotated tag）。不 push。

---

# Historical increment log (M1–M4)

> Non-architecture expansion. Maps Freeze SSOT to existing v2.x code.  
> Date: 2026-08-13. Freeze: Learning Auditability + Trajectory Sidecar.

## Decision

| Legacy | Freeze mapping | Action |
|--------|----------------|--------|
| `scientist_lab.core.state_machine` | Three-field run_state / evidence_status / review_decision | **SSOT — keep** |
| `scientist_lab.research_loop.state_machine` | v2.1 LLM session loop | **Isolate** until M3 Manager; do not drive autonomous Round |
| `scientist_lab.search.state_machine` | v2.0 search tree | **Isolate**; Bounded Tree is M4+ |
| `scientist_lab.domain.contracts.ExperimentContract` | Old workbench contract | **Adapt later**; Freeze contract is `schemas/experiment_contract.schema.json` |
| `tasks/rgbt_detection/adapter.py` (`RGBTDetectionAdapter`) | Execution HOW for existing workbench | **Wrap from** `adapters.dfine.DFINEAdapter` in M2 execute, do not merge objects yet |
| `patching/` Diff safety | Pre-execution Compliance | **Reuse ideas** in GateEngine M2/M3; do not replace Freeze Gate |
| Claim / Evidence v2.3 | ClaimGate / EvidenceValidator | **Reuse services**; bind to Freeze Result schema |

## Enum drift

- Freeze `run_state` includes MATERIALIZED / GATED / MEMORY_WRITTEN. Legacy research_loop uses different names. **Do not alias silently.**
- `scientific_outcome` is a **projection**, not a fourth competing state machine.
- Signal level `S0–S3` vs autonomy `A1–A4`: not implemented in runtime yet; Export Schema later.

## Trajectory

- **No** `trajectory_step.schema.json` in this increment (Export Schema comes after domain + events).
- Canonical facts: `research_event.schema.json` + `EventAppender`.
- MemoryWriter consumes events and writes `research_memory.json` / `strategy_memory.json` / `research_trace.json`. It does not own the event stream.

## Next code

1. **Done (M2 start):** DFINEAdapter.execute (default dry_run) / parse_metrics / fingerprint compute + EventAppender tool/execution events.  
2. **Done:** GateEngine BLOCK on fingerprint CHANGED; REJECT frozen edits; HUMAN_REQUIRED for formal. `make_cuda_live_runner` maps Freeze contract → legacy Fast Eval; `--execute` only. CLI: `dfine-adapter-run`.  
3. **Done:** ResultParser → EvidenceValidator → DecisionRubric on VALID only. INVALID/INCOMPLETE cannot KEEP/DISCARD. ExceptionHandler maps OOM/timeout → `RETRY_EXECUTION`.  
4. **Done:** GitManager dual pointers (`experiment_sha` / `best_sha`); DISCARD restores worktree, never `reset --hard`. MemoryWriter consumes events, refuses lessons without evidence, projects Research Trace. Does not invent lessons from metrics.  
5. **Done (REPLAY / M3 start):** Recovered artifacts (`metrics.json` + `checkpoint_selection.json`) through Gate→Parse→Evidence yield **VALID**. Rules-first Reviewer (`scientist_lab.core.reviewer`, role import `scientist_lab.agents.reviewer`) runs only on VALID; emits KEEP/DISCARD/REPLICATE/VALIDATE + evidence-linked lessons/strategy + `decision_summary`; negative rubric (`suggest_discard_threshold`) → DISCARD + `negative_evidence`. GitManager `apply_review` only after `review_decision`. MemoryWriter persists Reviewer structure and Trace edges (`derived_from` / `used_by` / `supported_by`). Fixture: `schemas/examples/recovered_k2c44_*` (K2-C seed44 COCO AP_small and AP all, not APS invented from mAP).  
6. **Done (N→N+1 citability, still no Planner):** Deterministic `build_candidate_next_plan` (not LLM) cites written `lesson_ids` / `strategy_ids` / `evidence_runs`. GateEngine + `assert_memory_refs_resolvable` REJECT empty or fake refs once MemoryWriter has content. `MemoryWriter.record_plan_citation` records Plan N+1 Trace edges (`used_by` / `derived_from` / `supported_by`). DISCARD / `negative_evidence` remains citable. Adapter still requires `proposed_changes` (no invented modules). Tests: `tests/unit/test_n_plus_one_memory_refs_m2.py`.  
7. **Done (Planner WHAT/WHY, rules-first):** `scientist_lab.core.planner` + role import `scientist_lab.agents.planner`. Reuses `build_candidate_next_plan`; Round≥1 must cite written memory; prefers DISCARD/`negative_evidence`; emits structured `decision_summary` and `plan_proposal` (`actor_role=planner`, `phase=planning`). No LLM, no Protocol edits, no CLI. Legacy MockPlanner isolated at `scientist_lab.agents.legacy_planner`. Helper `propose_and_gate_next` (Plan→materialize→Gate) is not a Manager loop. Tests: `tests/unit/test_planner_m3.py`.  
8. **Done (thin Manager):** `scientist_lab.core.manager` + role import `scientist_lab.agents.manager`. Reads three-field state + `next_orchestration_action`; `step()` is one action; `run_until(max_steps=...)` has a hard cap. Default dry-run / REPLAY. INVALID/INCOMPLETE/NOT_APPLICABLE skip Reviewer. `stop_rules` can STOP. NEED_PLAN without memory does not forge refs. Does not drive legacy `research_loop` / `search`. Tests: `tests/unit/test_manager_m3.py`.  
9. **Done (single REAL GPU round wiring):** Manager `NEED_EXECUTION` + `execute=True` uses injected `live_runner` or lazy `make_cuda_live_runner` (ExperimentService delayed). `require_live_ready` + CUDA doctor `live_ready=false` → `BLOCKED`, no GPU, no forged `metrics.json`. `execute=False` never calls the runner. Gate REJECT never ignites. CLI: `scientist-lab manager-run --protocol --plan --output-dir [--execute] [--require-live-ready] [--max-steps]`; doctor fact written to events + `cuda_doctor.json`; not ready + `--require-live-ready` exits non-zero. `python -m scientist_lab.cli manager-run` lazy-imports docker/ExperimentService. `dfine-adapter-run` kept. Tests: `tests/unit/test_manager_execute_m3.py`. No `trajectory_step.schema.json`, no training flywheel.  
10. **Done (REAL 前置：解开 docker 导入遮蔽):** 仓库镜像配方目录从 `docker/` 改名为 `dockerfiles/`，避免 CWD 把该目录当成 Python namespace package 挡住 PyPI `docker`（`from docker.errors import ...`）。CUDA doctor / vendor audit / Dockerfile `COPY` 路径已跟上。测试：`tests/unit/test_docker_import_unshadow_m4.py`（无 PyPI SDK 时 skip 导入断言，但断言不再存在名为 `docker/` 的配方目录）。  
11. **Done (M4 编排层 3 轮，stub，无 GPU):** Manager `run_until` + `max_extra_rounds=2` 连续 3 轮：Plan → Gate → Execute(stub) → VALID → Reviewer → Memory → Next Plan。一轮 DISCARD（负 APS）后 Planner 换模块；下一 Plan `memory_refs` 指向上一轮 lesson/strategy；Trace `used_by` / `derived_from` 可检查。`stop_rules.max_rounds` 与 `run_until(max_steps=)` 仍是硬上限。无 Bounded Tree。测试：`tests/unit/test_manager_multround_m4.py`。  
12. **REAL fast_eval 试跑（诚实失败，未伪造指标）：** 本机 doctor `live_ready=true`（RTX 5070 Ti、镜像、nvidia runtime）。`manager-run --execute --require-live-ready` 已过 Gate `APPROVED` 并点火；orchestrator 在 1200s 报 `timed_out`（容器内仍在 Epoch 0，约 1119/1671 step）。**未写出 `metrics.json`，未发明 APS，未 KEEP/DISCARD。** 随后 legacy `contract.json` 曾覆盖 Freeze 契约，Manager `_need_parse` `KeyError: run_id`（exit 1）。已修：REAL 产物进 `run/`；sync 跳过 Freeze 控制文件；`timed_out`→`timeout`；超时后 `FAILED` + idle，提示 `RETRY_EXECUTION` 而非立刻再点火。根因：Freeze 映射已带 `max_train_images=16` / `max_val_images=8`，但 vendored `stage_coco_for_dfine` / `train_dfine.py` 拷贝全量 COCO，忽略 subset。  
13. **Done（probe subset HOW，未改 Protocol dataset 身份）：** `stage_coco_for_dfine` 接受 `max_train_images` / `max_val_images`（`None` = 全量）；按 `file_name` 稳定排序取前 N 并过滤 annotations。`train_dfine.resolve_stage_image_caps` **仅** `fast_eval` / `smoke` / `smoke_train` / `debug` 缩小；`full_train` 保持全量。写出 `dfine_subset.json`。测试：`tests/unit/test_dfine_stage_subset.py`。  
14. **Done（REAL probe 闭环，GPU，VALID，未伪造指标）：** 权威一次：`.run/real_fast_eval_m4c`（exec `exec_b54bc9c66574`），wall ~48s，CLI `exit_code=0`，`metrics_forged=false`。doctor `live_ready=true`（RTX 5070 Ti；`image_registry_key` 仅 warning）。Gate `APPROVED` 后点火。`dfine_subset.json` staged train=16 / val=8；log `[0/8]`（非全量 1671）。`resource_usage.device=cuda`，peak ~364MB。真实 `metrics.json`：1 epoch、`pretrained=false`、APS=0.0（parse 只用 APS/`AP_small`，未把 mAP 写成 APS）。Trace：`gate_decision=APPROVED` → `execution=completed` → `evidence_check=VALID` → `review_decision=REPLICATE`（无 baseline，`primary_delta=None`，合法；非 KEEP/DISCARD）→ MemoryWriter 写出 `LESSON-run_plan_round1_neck_hr-001` / `STRATEGY-run_plan_round1_neck_hr-001`。`run_state=MEMORY_WRITTEN`，`scientific_outcome=INCONCLUSIVE`。**Git 指针未按决策更新**：Manager 路径未调用 `GitManager`（`REPLICATE` 本也不会改 `best_sha`）。零 APS 是 probe 预算下的实测，不是 SOTA。先前 timeout（`.run/real_fast_eval_trace1`）与 m4b CPU/收集失败不覆盖本次。仍不要 `trajectory_step`。  
15. **Done（GitManager 接到 Freeze Manager）：** `NEED_PARSE` 尽量 `record_experiment`（不改 `best_sha`）；`NEED_REVIEW` 之后 `apply_review`。**KEEP** 才把 `best_sha` 设为 `experiment_sha`；**DISCARD** 工作树切回 `best_sha`（`switch --detach`，禁止 `reset --hard`），负结果 commit 仍可解析；**REPLICATE / VALIDATE / PENDING** 不改 `best_sha`。无 VALID 不 apply KEEP/DISCARD。`project_dir` / `git_root` 没有自己的 `.git` 则 skip 并写 `tool_call git_manager` + `git_pointers.json`，不崩溃、不向上找父仓库（避免 `.run/` 误提交 Scientist Lab 本体）。测试：`tests/unit/test_manager_git_m4.py`。  
16. **Done（M4 真 2 轮 GPU fast_eval + N→N+1 Trace）：** CLI 增加 `--max-extra-rounds`（默认 0）与可选 `--baseline-metrics`。`.run/real_fast_eval_m4_rounds2` wall ~92s，`exit_code=0`，`metrics_forged=false`，`live_ready=true`。两轮都 Gate `APPROVED` 后点火（`dry_run=false`）：exec `exec_1c9d1ba3ed94` / `exec_e8858e39430c`，各 16/8 子集、`device=cuda`。APS=0.0（`AP_small`，未把 mAP 写成 APS）；相对 m4c baseline `primary_delta=0.0` → 两轮 Reviewer 均为 **KEEP**（合法，非伪造）。N+1 Plan `plan_round2_from_run_plan_round1_neck_hr` 的 `memory_refs` 指向 `LESSON-run_plan_round1_neck_hr-001` / `STRATEGY-run_plan_round1_neck_hr-001`；Trace 有 `used_by` / `derived_from`。Round 1 产物归档到 `runs/run_plan_round1_neck_hr/`。`.run/` 不是 git repo：Git **skip**（`best_sha_updated=false`），未向上误提交。零 APS / KEEP@delta=0 是 probe 事实，不是 SOTA。测试：`test_cli_parses_max_extra_rounds`。仍不要 `trajectory_step`。  
17. **Done（真 GPU 负结果改向，3 轮 fast_eval）：** `.run/real_fast_eval_m4_rounds3` wall ~124s，`exit_code=0`，`metrics_forged=false`。baseline APS=0.6 是 **synthetic control**（写入 `baseline_metrics.json` 并标注；不是本轮 GPU 指标；选它是为了让实测 APS=0.0 得到 delta=-0.6 < discard_if=-0.5）。三轮均 Gate `APPROVED` 后点火（cuda，16/8）：exec `exec_ef974841f0b8` / `exec_53f2663a181b` / `exec_3a2bfc79d224`。APS 均为 0.0（`AP_small`，未伪造）。三轮 Reviewer 均为 **DISCARD**（delta=-0.6）。Round 2 Plan 引用负结果 `LESSON-run_plan_round1_neck_hr-001`，`modification_scope` **neck→fusion**；Round 3 Plan 引用 Round 2 负结果 lesson，**fusion→neck**。第三轮 DISCARD 后 `stop_rules.max_consecutive_discards=3` 触发 STOP，Round 3 未写 Memory（协议硬停，未改 rubric）。Git skip（`.run/` 非 repo）。HOW 仍 `fusion_method=none`：Planner 换了 WHAT 模块，Adapter 未发明 fusion 实现。零 APS 不是 SOTA。测试：`test_reviewer_discard_when_aps_zero_vs_control_baseline`。仍不要 `trajectory_step`。  
18. **Done（Adapter HOW 落实 Planner 模块变更；未发明算子）：** `DFINEAdapter.materialize_contract` 按 `modification_scope[0]` 填已有 train/config 旋钮（`adapters/dfine/how.py`），经 `materialization.legacy_parameters` 进入 CUDA Fast Eval。**neck HOW：** `input_mode=rgb`、`fusion_method=none`、`neck.type=standard`（现有 HybridEncoder；**不**因 Plan 写「high-res」就发明/启用 FDPN）。**fusion HOW：** `input_mode=rgbt`、`fusion_method=early_concat`（仓库已有 staging 混合；**不是**新 fusion 网络）。`gated_multiscale` 在仓库存在，probe HOW **未选用**（更重 dual-stream，Adapter 不假装 SOTA）。Frozen Fingerprint **哈希不变**（HOW 只进 `notes` / instrumentation / `how_signature`，不进 `_HASH_FIELDS`），故 Gate 仍只看契约、neck→fusion 不会 BLOCK。单测：`test_materialize_neck_vs_fusion_how_differs`、`test_neck_fusion_frozen_fingerprint_equivalent_how_notes_differ`、`test_instrumentation_carries_how_signature`；套件 **111 passed**（原 108 + 3）。**REAL 验证（各 1 轮，`--max-extra-rounds 0`，不为 APS 拉长）：** `.run/real_how_neck` wall ~51s，exec `exec_b0d7618fbe43`，cuda ~24s / 363MB，`staging_mode=single_modality`；`.run/real_how_fusion` wall ~67s，exec `exec_b48f714b1e6c`，cuda ~39s / 364MB，`staging_mode=early_concat_blend`。二者 `exit_code=0`，`metrics_forged=false`，Gate `APPROVED`，VALID，APS=0.0（`AP_small`，未把 mAP 写成 APS），`parameter_count` 同为 10228145（early_concat 是输入混合，不是新骨干）。容器实际吃到不同 HOW：`_legacy_fast_eval_contract.json` + `dfine_subset.json` + `model_summary.json`。loss/optimizer/hyperparameter/augmentation 仍无独立已有旋钮映射（HOW 身份仅 `primary_module`）。零 APS 不是 SOTA。仍不要 `trajectory_step`。
19. **Done（ClaimGate v1 确定性规则层；KEEP ≠ Claim）：** `scientist_lab.core.claim_gate` 在 Evidence 之后、声称之前判定 `SUPPORTED | PARTIALLY_SUPPORTED | INCONCLUSIVE | BLOCKED`，带 `reason` + `evidence_refs`。不是 Agent，不覆盖 Reviewer KEEP/DISCARD，不替代 `scientific_outcome`。硬编码：probe ≠ 正式科研证据；mAP ≠ APS；无 baseline 不得 C1 outperform；无 ablation 不得 C2（含 FDPN 正向贡献）；无 matched SOTA 不得 C4；INVALID/FAILED 不得 claim；KEEP ≠ SUPPORTED；DISCARD ≠ 模块无效。Schema：`claim.schema.json` / `claim_gate_result.schema.json`。CLI：`scientist-lab claim-gate --run-dir … [--claim]`（无 GPU）。Manager 在 Reviewer 之后写 `claim_gate.json` 并发 `claim_gate` 事件。Freeze 不变量已追加到架构文档 §16 与构建方案「十三」。测试：`tests/unit/test_claim_gate_m4.py`；套件 **127 passed**（原 111 + 16）。未开 formal GPU。现有 probe `.run/real_how_fusion` 上 C1「early_concat outperform rgb+none APS」判定 **BLOCKED**（无 formal baseline）。
20. **Done（Formal C1 协议/HOW/ClaimGate 配对；Human Gate=开始）：** 新增 Formal Protocol `research_protocol_rgbt_dfine_formal_c1_v1`（`allow_scientific_claims=true`，`max_claim_strength=C1`，`full_training=auto`）与两份 Plan：Formal-01 `hyperparameter`→rgb+none；Formal-02 `fusion`→early_concat（staging blend，**不是 FDPN**）。Probe 协议仍禁止科学声称。Adapter `budget_class=formal` → `execution_mode=full_train`、去掉 16/8 子集、timeout **4h/轮**（14400s，禁止再用 1200s 把全量训死当科学失败）。Orchestrator 允许 `full_train`；C1 科学声称仍由 Freeze ClaimGate 判定，orchestrator `formal_success` 仍为 false。ClaimGate：`--baseline-run-dir` 配对 fingerprint；双方 APS=0 或 candidate 不高于 baseline → **INCONCLUSIVE**（不得写成模块无效或 SOTA）。**最小 formal 预算（Laptop RTX 5070 Ti）：** 全量 `rgbt_tiny_v1` train+val（非 16/8）、160×160、epochs=2、batch=2、`pretrained=true`、seed=42。限制：不是 640/20ep 论文协议；early_concat ≠ FDPN。输出目录 `.run/formal_c1_aps_early_concat/`（gitignore）。测试套件 **131 passed**。K2C 历史 640 formal 合同 fingerprint 不匹配，**未 import**。
21. **Done（Formal C1 GPU 对照 + 配对 ClaimGate）：** Formal-01 `.run/formal_c1_aps_early_concat/baseline` 复用（未重跑）：3342/650，`rgb+none`，APS=**0.0163**（`AP_small`），VALID + REPLICATE，C0 observational SUPPORTED。Formal-02 中断目录 `candidate/` 保留；续跑 `.run/formal_c1_aps_early_concat/candidate_resume`，exec `exec_b194798b00b6`，wall ~23min，exit 0，`full_train` 3342/650，`rgbt+early_concat` / `early_concat_blend`，APS=**0.0326**，VALID + **KEEP**（delta=+0.0163）。Frozen hashes 与 Formal-01 一致（HOW 只在 notes）。配对 ClaimGate `claim_gate_c1.json`：**C1 SUPPORTED**（candidate APS 高于 matched formal baseline；KEEP 未决定声称）。Manager 自动 `claim_gate.json` 仍是 **BLOCKED**（只有 `baseline_metrics.json` 无数指纹）——正确。限制：160×160 / 2ep，不是 640/20ep；`fusion_applied=false`（staging 混合，不是新网络）；`parameter_count` 同 10228145；不得 C2/C3/C4；`scientific_outcome` 仍 INCONCLUSIVE（证据阶段 `primary_delta=None` 投影，ClaimGate 不覆盖）。CLI `claim-gate --output` 修了 `main()` 内二次 `from pathlib import Path` 导致的 UnboundLocalError。仍不要 `trajectory_step` / FDPN / SOTA。

