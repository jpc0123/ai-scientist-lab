# v1.3 Acceptance

- Offline LLM Provider (Fake / Replay / Audit / Schema)
- Planner/Critic adapters (default mock)
- Quality eval + token/cost/latency limits
- Finite tree plan-next with `--provider mock|fake|replay`
- Real cloud LLM: false

Checks: 9/9

Run: `python scripts/accept_v13.py`
