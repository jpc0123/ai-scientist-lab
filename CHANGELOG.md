# Changelog

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
