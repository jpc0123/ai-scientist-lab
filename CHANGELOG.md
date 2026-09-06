# Changelog

## [Unreleased]

### Added
- HOW 生命周期（2026-08-26）：一次 Human Gate 冻结协议。战役内 LLM 决定是否新增 HOW、沙箱写 `plugin.py`、是否接入 overlay；Planner 只选可物化 HOW。`max_rounds` 6→12，`protocol_version` 1→2。`/loop` 默认最多 12 枪 GPU。不 merge 主树。KEEP ≠ Claim。决策 [`docs/research/v26/DECISION_LLM_HOW_LIFECYCLE.md`](docs/research/v26/DECISION_LLM_HOW_LIFECYCLE.md)
- HOW 插件 Harness 走阿里云兼容协议（2026-08-26）：容器改用 `dsh-llm-pi-ai` + `openai-completions`，`supportsDeveloperRole=false` / `max_tokens`。不是只能用 DeepSeek 模型。不开 GPU。KEEP ≠ Claim。
- HOW 插件工人接口（2026-08-26）：`author_how_patch` 的写 Diff 段可替换。默认仍是 `LlmPatchPluginWorker`（RealPatchPlanner）。`FakePluginWorker` 复制示例插件，无网络、无 Docker、无 GPU。人再注册。KEEP ≠ Claim。
- HOW 插件双入口（2026-08-26）：`/loop` 可让现有 LLM 写 `plugin.py`，也可上传/粘贴人写的文件。同一 `author_how_patch` 沙箱 smoke，人再注册。不接 DeepSeek Harness 运行时。不开 GPU。KEEP ≠ Claim。决策 [`docs/research/v26/DECISION_OPEN_HOW_PLUGIN.md`](docs/research/v26/DECISION_OPEN_HOW_PLUGIN.md)
- HOW 候选冻结：LLM 检索后只能写入 `how_pending.json` 草稿；人审 `register` 且 Adapter 能映射才进目录。未批准 / 无映射不能 materialize。文献不能进 ClaimGate。KEEP ≠ Claim。无 GPU。
- **P0 ACCEPT（2026-08-24）**：live 战役 `p0_20260824T111526Z` `status=completed` `gpu_rounds=2`（F1 seed 42 ×2，`APS_lowlight` 0.00642 / 0.01034）。Rubric REPLICATE。ClaimGate C0 observational。KEEP ≠ Claim。不是 G2。无第三轮 GPU。决策 [`docs/research/v26/DECISION_P0_ACCEPT.md`](docs/research/v26/DECISION_P0_ACCEPT.md)
- **P1 Planner 质量 INCONCLUSIVE（2026-08-24）**：同一协议 / RGBT-Tiny，规则 `p0_20260824T094123Z` vs LLM `p0_20260824T111526Z`。比规划质量不比 APS。两臂 R1 空转 F1 seed 42，Rubric Δ=`null`。不作废前误判 `winner=rules`。决策 [`docs/research/v26/DECISION_P1_PLANNER_QUALITY.md`](docs/research/v26/DECISION_P1_PLANNER_QUALITY.md)
- Planner 对照修复（2026-08-25）：规则臂（人工特选）在 R0/bootstrap 或负 Δ 后下一枪换注册 HOW（F1→F3→F0）。禁止空转同一 HOW+同一 seed。冻结 R0 `aps_lowlight.json` 可注入战役 baseline。KEEP ≠ Claim。无新 GPU。
- LLM 路线（2026-08-25）：`/loop` 开始只走 Live LLM；不再把 LLM 选的 HOW 改写成 F3。同 HOW 复现仍 bump seed。规则臂先停用。KEEP ≠ Claim。无新 GPU。
- Web `/llm-config` 可配置 Semantic Scholar Key（独立于 LLM Key，写入 `runtime/literature_secrets.env`，GET 不回显）
- v2.6 V26.4 R0 协议锚：HOW=F1 + `low_light_subset_v1` + `APS_lowlight`；CLI `freeze-v26-r0` / `v26-r0-run`；拒绝伪造 GPU 数字
- V26.4 R0 GPU 已跑并绑定：`status=metrics_bound`，`APS_lowlight=0.0045926865160844455`（pycocotools 切片 val），run=`outputs/v26_r0` / `exec_31eff20c4e0c`
- V26.5 R1 Live Reviewer 已在已有 GPU 证据上闭合：`outputs/v26_r1/semantic_review.json`，`hypothesis_status=not_a_claim`；Rubric KEEP / ClaimGate BLOCKED / `APS_lowlight=0.02135704762627717` 未改。未重跑 GPU。
- V26.5 Round 2 GPU pack `outputs/v26_r2/`：HOW=F0 RGB-only，`APS_lowlight=1.3452432199741713e-06`（相对 R0 -0.004591341272864471，相对 Round 1 F3 -0.021355702383057194）。Rubric KEEP/INCONCLUSIVE（未越 discard 阈）。ClaimGate BLOCKED/C0。Live Reviewer `not_a_claim`。不是 G2。
- V26.5 Round 3 GPU pack `outputs/v26_r3/`：HOW=F3 seed **43**（合同已对齐，非静默 42 重跑），`APS_lowlight=0.04881493259150456`（相对 R0 +0.044222246075420114）。与 Round 1 F3 seed 42 同方向高于 R0。Rubric KEEP。ClaimGate BLOCKED/C0。不是 G2 成功声明。
- V26.5 R3 Live Reviewer 已在已有 GPU 证据上闭合：`outputs/v26_r3/semantic_review.json`，`hypothesis_status=not_a_claim`。根因是嵌套 `run_id` 被 LLM 截成 `...31eff20c`；前缀规范化后 persist，外键仍 fail-closed。Rubric KEEP / ClaimGate BLOCKED / `APS_lowlight` 未改。未重跑 GPU。
- V26.5 Round 4 GPU pack `outputs/v26_r4/`：HOW=F3 seed **44**，`APS_lowlight=0.04204268629433699`（相对 R0 +0.03744999977825254）。三 seed 同方向高于 R0。Rubric KEEP。ClaimGate BLOCKED/C0。不是 G2 成功声明。
- V26.5 R4 Live Reviewer 同修复后已写出 `outputs/v26_r4/semantic_review.json`（`not_a_claim`）。ClaimGate 仍 BLOCKED。不是 G2。
- V26.5 Round 5：Next Plan F3 seed 45 曾因 `max_rounds=5` 在 GPU 前 STOP（`metrics_forged=false`）。Human Gate A 后 GPU 已落地：pack `outputs/v26_r5/`，HOW=F3 seed **45**，`APS_lowlight=0.035942673873977496`（相对 R0 +0.03134998735789305）。四 seed 同方向高于 R0。Rubric KEEP。ClaimGate BLOCKED/C0。不是 G2 成功声明。
- **P4 STOP（2026-08-21）**：用户拒绝 `max_rounds` 2→3。P4 归档为 INCONCLUSIVE **Transfer Probe**（不是 generalization validation，不是失败验证）。Strategy Memory：`LESSON-V26-P4-PROBE-INCONCLUSIVE-001`，`STRATEGY-V26-P4-GATED-MULTISCALE-001` action=`keep`。未 BAN F3。ClaimGate 仍 C0/BLOCKED。无新 GPU。
- **P4 C0 分析（2026-08-21）**：无新 GPU。`outputs/v26_p4_analysis/summary.json`。F3−F1 `APS_lowlight` +0.0017082151533518684，但 AP50_lowlight 与全集 mAP50 下降；相对 D-FINE 同 seed Δ 约为 0.102。INCONCLUSIVE。不是 G2/G3 / transfer success。
- **P4 GO（2026-08-21）**：Adapter=`rtdetr`；Protocol=`research_protocol_rgbt_rtdetr_transfer_v26`（`max_rounds=2`，C0）。F1 `outputs/v26_p4_r0` `APS_lowlight=0.013241256515861267`（exec=`exec_5cdd236bde02`）。F3 `outputs/v26_p4_r1` `APS_lowlight=0.014949471669213135`（exec=`exec_6b36673902b9`，KEEP，ClaimGate BLOCKED/C0）。不是 G2/G3。
- V26.5 R5 Live Reviewer 已在已有 GPU 证据上闭合：`outputs/v26_r5/semantic_review.json`，`hypothesis_status=not_a_claim`。Rubric KEEP / ClaimGate BLOCKED / `APS_lowlight` 未改。未重跑 GPU。
- Semantic Scholar live 已接通（pack `provider=semantic_scholar`，`litq_a4d7bff94296`；探查 `litq_687d9c49f056`）。文献不能进 ClaimGate。
- **Human Gate A（2026-08-20）**：用户批准 Protocol Amendment，`stop_rules.max_rounds` 仅 5→6。ClaimGate 仍 C0。不宣称 G2。P4 / RT-DETR 推迟。
- Planner contract 将非法 `expected_effect.direction`（如 `decrease_relative_to_F3`）夹紧到 `increase|decrease|stabilize|unclear`。不发明 HOW。

## [2.3.0] - 2026-07-25

CUDA + **Vendor DFINE** readiness and gated Fast Eval path on top of v2.2 (offline-first; no silent stand-in→formal claims).

### Added
- Vendor pin / stage / adapter surface (`vendor_audit`, pinned `third_party/DFINE`)
- CUDA doctor with GPU-depth probes (`nvidia-smi` inventory, Docker nvidia runtime summary)
- Fast Eval orchestrator (default dry-run; `--execute --require-live-ready` for live)
- Vendor Evidence annotation (non-stand-in); metrics feedback; exploratory-only
- CUDA formal triad planner (`rgb` / `thermal` / `fusion`; default dry-run)
- Formal path Claim Gate (`claim_formal_dfine_path`); path-open ≠ superiority
- Gated live pipeline `dfine_real_acceptance` + `scripts/accept_v23_real.py` (default SKIP)
- Offline demo `scripts/demo_dfine_cuda_offline.py`; guide `docs/dfine-cuda-runthrough.md`
- CLI: `dfine-cuda-doctor` / `dfine-cuda-fast-eval` / `dfine-cuda-record-evidence` /
  `dfine-cuda-formal-triad` / `dfine-formal-path-gate` / `dfine-real-acceptance`
- System Doctor check `dfine_cuda` (offline)
- Offline `scripts/accept_v23.py`; acceptance notes under `docs/acceptance/v2.3/`

### Changed
- Package / CLI / Web / API health versions aligned to **2.3.0** (`API_VERSION=v2.3.0`)

### Notes
- Zero GPU / zero network by default; live CUDA needs env gates + doctor `live_ready`
- Stand-in evidence keeps `claim_formal_dfine` **blocked**; Fast Eval never yields formal superiority **supported**
- Tag target: `v2.3.0` (includes v2.2 Diff workbench; create tag only after explicit user request)
- Deferred / gated: live `accept_v23_real` GPU evidence; multi-seed formal superiority claims

## [2.2.0] - 2026-07-25

Real-provider **restricted Unified Diff** loop on top of v2.1 (CodeContext → Diff → safety → approve → sandbox → evidence → replay).

### Added
- CodeContextBundle / AllowedSourceFile / SourceSnapshot / PathPolicy.for_code_context / SHA + size budget
- RealPatchPlanner + PATCH_PROPOSAL_SCHEMA (no silent mock/fake/replay fallback)
- DiffSafety: secrets, prompt injection, dangerous APIs, line budget, path deletes, executables
- Patch real_only mode + LimitingProvider call/token/cost budget
- Approval content seal (source/context/patch/proposal SHA); stale approve invalidates
- Sandbox apply + fixed test registry (`smoke` / `syntax` / `unit` / `mock_experiment`)
- PatchEvidence feedback into Evidence / Claim drafts / Planner / Tree notes
- Patch Replay Bundle export + MockTransport offline replay (`patch-export-replay` / `patch-replay`)
- Web Patch workbench: `/code-contexts*`, `/patches/propose-real`, check-seal, export-replay; Patches UI
- Offline `scripts/accept_v22.py`; gated live `scripts/accept_v22_real.py` (default SKIP)

### Changed
- Package / CLI / Web / API health versions aligned to **2.2.0** (`API_VERSION=v2.2.0`)

### Notes
- Never applies Diff to the main workspace; no auto-merge / commit / push
- Live Provider Diff needs explicit `--allow-network` / Web checkbox + env gates
- Tag target: `v2.2.0` (baseline `v2.1.0`); create tag only after explicit user request
- Deferred: CUDA + Vendor DFINE formal runs (v2.3); `accept_v22_real` live evidence may be SKIP

## [2.1.0] - 2026-07-24

Real-LLM multi-round research closed loop on top of the v2.0 workbench.

### Added
- `research_loop/` session model, state machine, repository, and service
- Real-only Planner/Critic provider gate (`fallback_allowed=false`; no silent mock/replay)
- Human approve → Digits execute (ban mock entrypoint); Evidence/Claim feedback backfill
- Round-2 PlanningContext + `FeedbackUseVerifier` (six deterministic dimensions)
- Replay Bundle export (scrubbed) + offline `scripts/accept_v21.py` (17/17)
- Gated live acceptance `scripts/accept_v21_real.py` (default SKIP / zero network)
- Web Console Real Loops (`/real-loops`) with REAL / MOCK / REPLAY display mode
- CLI: `real-loop-create|show|check|plan|review|approve|reject|execute|record-feedback|next-round|verify-feedback|export|…`
- API: `/api/v1/real-loops*`

### Changed
- Package / CLI / Web / API health versions aligned to **2.1.0** (`API_VERSION=v2.1.0`)

### Notes
- Default LLM remains **mock**; live closed-loop needs explicit gates
- Human approval preserved; no auto-approve / auto-push / auto Diff
- Tag target: `v2.1.0` (baseline `v2.0.0`)
- Deferred historically: real-provider Diff (**done in 2.2.0**); CUDA + Vendor DFINE (v2.3)

## [2.0.0] - 2026-07-24

Scientist Lab AI Scientist Workbench — unified local product surface on top of the v1.x controlled research stack.

### Added
- `ResearchProject` lifecycle and six-step create wizard
- Unified Dashboard (todos, health, activity)
- Experiment center filters, detail, and compare UI
- Planning / approval / experiment-tree workbench surfaces
- Evidence / Claim / Report / Audit centers in Web Console
- System Doctor, security posture, and restart Recovery (no auto expensive re-runs)
- Windows install / start / stop / doctor / upgrade scripts and `workbench` CLI helpers
- Digits and RGB-T debug demos; getting-started / installation / security docs
- Project export / import bundles
- Acceptance: `scripts/accept_v20.py` (30 checks), `docs/acceptance/v2.0/`

### Changed
- Package / CLI / Web / API health versions aligned to **2.0.0** (`API_VERSION=v2.0.0`)
- Product positioning: local-first, human-approval AI Scientist workbench

### Notes
- Default LLM remains **mock**; real providers stay explicitly gated
- No arbitrary Shell; no auto Git push; Claim Gate preserved
- Tag target: `v2.0.0` (baseline `v1.9.0`)
- Deferred (not blocking this release): CUDA + Vendor DFINE strong acceptance; real-provider Diff generation (v1.6.7)

## [1.9.0] - 2026-07-23

Controlled Git merge, rollback (`git revert`), and local Release Candidate.

- Whitelist `GitAdapter` (no push / reset --hard / force / rebase)
- Worktree apply → approve → commit → finalize `--no-ff`
- Acceptance: `scripts/accept_v19.py` (20/20)

## [1.8.0] - 2026-07-23

Workspace views and export-only Release management.

- Release create / freeze / export / discard
- Report / Audit build-export via `/api/v1`
- Chat workspace UI
- Acceptance: `scripts/accept_v18.py` (8/8)

## [1.7.0] - 2026-07-22

Web Console MVP (`/api/v1` + React console).

- Dashboard, executions, trees, plans, patch/report/audit surfaces
- Acceptance: `scripts/accept_v17.py` (22/22)

## [1.6.0] - 2026-07-22

Restricted source patching and sandbox validation.

- PatchProposal → verify → approve → sandbox apply/test → evidence → merge intent
- Deferred: optional real-provider Diff generation (v1.6.7)
- Acceptance: `scripts/accept_v16.py` (12/12)

## [1.5.0] - 2026-07-22

LLM quality governance and multi-model evaluation.

### Added
- Versioned eval suite (`eval_suite_v1`, 25 planner/critic/safety cases)
- Rule-based Planner / Critic / Safety graders (safety = hard fail)
- `llm-eval-run` for mock / fake / replay / real (real gated; CI offline)
- `LLMModelProfile` registry (`llm-profile-register/list/show/select`)
- Evaluation scorecards under `outputs/.../llm_evals/`
- Quality Gate (`llm-eval-verify`) and regression compare (`llm-eval-compare`)
- Static profile ranking (`llm-profile-rank`; no auto-default by cost)
- Planning gate: `--require-quality-gate` → `profile_not_qualified` unless bypassed
- Acceptance: `scripts/accept_v15.py`, `docs/acceptance/v1.5/`

### Notes
- Default provider remains **mock**
- Unqualified / bypassed profiles are not formally approval-eligible
- After `v1.5.0`, only bugfixes on this line; source-patch work goes to v1.6+

## [1.4.0] - 2026-07-22

First OpenAI-compatible cloud LLM provider (explicitly gated; default remains mock).

### Added
- `OpenAICompatibleProvider` + `HttpTransport` / `MockTransport` (offline tests)
- Config / key boundary (`LLMConfig`, `OpenAICompatibleConfig`, `redact_secrets`)
- Retry, budget, concurrency, one-shot schema repair
- `llm-eval --suite` with real-provider gates (live HTTP not required for CI)
- Planner / Critic / `tree-plan-next` support for `--provider real --allow-network`
- Context sanitizer; fail-closed `real_provider_failed` (no silent mock fallback)
- Acceptance: `scripts/accept_v14.py`, `docs/acceptance/v1.4/`

### Notes
- Default planner/critic path remains **mock**
- Live cloud calls need triple gate: provider=real + `--allow-network` + `LLM_ALLOW_NETWORK`
- After `v1.4.0`, only bugfixes land on this line; new work goes to v1.5+

## [1.3.0] - 2026-07-22

Offline LLM provider architecture and evaluation.

### Added
- Unified `LLMProvider` layer: Fake, Replay, audit records, JSON Schema validation
- `ProviderPlanner` / `ProviderCritic` adapters (`mock` | `fake` | `replay`)
- Quality evaluation (`llm-eval`) comparing Mock / Fake / Replay (Real deferred)
- Token / cost / latency limits (`LimitingProvider`) and usage summary (`llm-usage`)
- Finite tree expansion via `tree-plan-next --provider`
- Acceptance: `scripts/accept_v13.py`, `docs/acceptance/v1.3/`

### Notes
- Default planner/critic path remains **mock**
- No real cloud LLM calls in this release
- After `v1.3.0`, only bugfixes land on this line; new work goes to v1.4+

## [1.2.0] - 2026-07-22

Research reporting and audit bundle (deterministic templates; no LLM reporting).

## [1.1.0] - 2026-07-21

Finite Best-First experiment tree (MockPlanner).

## [1.0.0] - 2026-07-21

Controlled experiment planning (MockPlanner / MockCritic).
