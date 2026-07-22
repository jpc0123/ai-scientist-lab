# v1.7 Acceptance

Scientist Lab Web Console MVP — local single-user research workbench.

## Covered

- `/api/v1` health + system summary + path policy
- Executions / Trees / Plans list contracts
- Patch controlled lifecycle (approve → sandbox → test → evidence → merge intent)
- Evidence / Claims / Reports / Audits list surfaces
- OpenAPI core routes; no arbitrary shell API
- Frontend `web/dist` build present

## Run

```powershell
cd D:\AI Scientist_tiao\scientist-lab
cd web; npm run build; cd ..
.\.venv\Scripts\python.exe scripts\accept_v17.py
.\.venv\Scripts\python.exe -m pytest tests/unit/test_api_v1.py -q
```

Checks: see `v17_acceptance_report.json` after run.
