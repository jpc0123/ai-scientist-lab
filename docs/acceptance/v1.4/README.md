# v1.4 Acceptance

- First OpenAI-compatible cloud provider (offline-gated)
- Default provider remains **mock**; CI needs no API key
- Triple gate for live HTTP: `--provider real` + `--allow-network` + `LLM_ALLOW_NETWORK`
- Failures surface as `real_provider_failed` (no silent mock fallback)
- MockTransport covers Planner / Critic / retry / budget / schema repair

Checks: 17/17

Run: `python scripts/accept_v14.py`
