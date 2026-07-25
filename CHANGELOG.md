# Changelog

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
