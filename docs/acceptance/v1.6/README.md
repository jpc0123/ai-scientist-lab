# v1.6 Acceptance

Restricted source patching with sandbox-only apply and human merge intent.

## Covered

- Mock `PatchProposal` (Unified Diff) + parser + fingerprint / duplicate detection
- PathPolicy / PatchVerifier (allow-list, no binary, no escapes, no secrets)
- CLI: propose / show / verify / approve / reject
- Sandbox apply only (`patch-apply-sandbox`) — never main tree
- Controlled sandbox tests (`patch-test-sandbox`: smoke | syntax | mock_experiment)
- `PatchEvidence` + human `merge` / `discard` intent (no auto main apply)

## Deferred

- v1.6.7 optional real LLM provider for diff generation (gated)

## Run

```powershell
cd D:\AI Scientist_tiao\scientist-lab
.\.venv\Scripts\python.exe scripts\accept_v16.py
.\.venv\Scripts\python.exe -m pytest tests/unit/test_patching_v16.py -q
```

Checks: see `v16_acceptance_report.json` after run.
